///==========================================================================
/// File: ParallelEdtLowering.cpp
/// Complete implementation of parallel EDT lowering pass that transforms
/// arts.edt<parallel> operations into task graphs before datablock lowering.
///
/// This pass implements parallel EDT lowering that must execute before
/// datablock operations are lowered to ensure proper dependency management.
///==========================================================================

#include "ArtsPassDetails.h"
#include "arts/Analysis/ArtsAnalysisManager.h"
#include "arts/ArtsDialect.h"
#include "arts/Codegen/ArtsCodegen.h"
#include "arts/Passes/ArtsPasses.h"
#include "arts/Utils/ArtsUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "arts/Utils/ArtsDebug.h"
#include "polygeist/Ops.h"
ARTS_DEBUG_SETUP(parallel_edt_lowering);

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arts;

namespace {

//===----------------------------------------------------------------------===//
// ParallelEdtLowering - Lowers parallel EDTs into explicit task EDTs.
//
// This class implements OpenMP static scheduling semantics for ARTS.
// When an arts.edt<parallel> contains an arts.for, it transforms it into:
// - For static without chunk_size: Block distribution of contiguous iteration
//   ranges
// - For static with chunk_size: Round-robin (cyclic) distribution of chunks
// - Inter-node: Round-robin across nodes via route for cyclic, direct route for
//   block distribution
// - Intra-node: All tasks with route=0 for free worker assignment
// - Inner arts.edt<task> operations that execute chunks or blocks of work
//===----------------------------------------------------------------------===//
class ParallelEdtLowering {
public:
  ArtsCodegen *AC;
  Location loc;

  struct ChunkedForInfo {
    Value lowerBound;
    Value upperBound;
    Value loopStep;
    Value chunkSize;
    Value totalWorkers;
    Value totalIterations;
    Value totalChunks;
    Value workerFirstChunk;
    Value workerChunkCount;
    Value workerIterationCount;
  };

  struct LoopInfo {
    Value lowerBound;
    Value upperBound;
    Value loopStep;
    Value chunkSize;
    Value totalWorkers;
    Value totalIterations;
    Value totalChunks;
  };

  struct ChunkedDepsResult {
    SmallVector<Value> deps;
    SmallVector<DbAcquireOp, 4> chunkedAcquires;
  };

  struct ReductionInfo {
    Value originalAcc;
    MemRefType originalType;
    MemRefType elementBufferType;
    Type scalarType;
    DbAllocOp partialAlloc;
    DbAllocOp originalAlloc;
  };

  struct ReductionAccess {
    Value buffer;
    SmallVector<Value> indices;
  };

  ParallelEdtLowering(ArtsCodegen *AC, Location loc) : AC(AC), loc(loc) {}

  //===----------------------------------------------------------------------===//
  /// Lower a parallel EDT.
  ///
  /// Transforms parallel EDTs containing arts.for operations into worker EDTs.
  /// Example: arts.edt<parallel> { arts.for %i = 0 to 10 { ... } }
  /// becomes: multiple arts.edt<task> workers each handling a portion of the
  /// loop.
  //===----------------------------------------------------------------------===//
  LogicalResult lowerParallelEdt(EdtOp op) {
    if (!op.getRegion().hasOneBlock()) {
      ARTS_ERROR("Parallel EDT must have exactly one block");
      assert(false);
      return failure();
    }

    OpBuilder::InsertionGuard IG(AC->getBuilder());
    AC->setInsertionPoint(op);

    /// Create arts.epoch wrapper
    EpochOp epochOp = createEpochWrapper();
    Block &epochBody = epochOp.getRegion().front();
    AC->setInsertionPointToEnd(&epochBody);

    arts::ForOp containedForOp = findContainedForOp(op);
    if (containedForOp) {
      ARTS_INFO("Lowering parallel EDT with contained arts.for");
      auto [lb, ub, isInterNode] = determineParallelismStrategy(op);
      activeReductions.clear();
      activeLoopCount = Value();
      activeLoopOp = nullptr;

      if (containedForOp->hasAttr("chunk_size")) {
        /// Cyclic distribution: prepare chunks and workers
        LoopInfo info;
        info.lowerBound = containedForOp.getLowerBound()[0];
        info.upperBound = containedForOp.getUpperBound()[0];
        info.loopStep = containedForOp.getStep()[0];
        info.totalWorkers = ub;
        computeTotalIterationsAndChunkSize(containedForOp, info);
        prepareReductionBuffers(containedForOp, info.totalChunks);
        lowerStaticCyclic(op, containedForOp, lb, ub, isInterNode);
        activeLoopCount = info.totalChunks;
      } else {
        /// Block distribution: create one worker per iteration
        Value totalIterations = computeTotalIterations(containedForOp);
        Value slotCount = totalIterations;
        prepareReductionBuffers(containedForOp, slotCount);
        lowerStaticBlock(op, containedForOp, lb, slotCount, isInterNode);
        activeLoopCount = slotCount;
      }

      if (!activeReductions.empty()) {
        /// Create reduction aggregator EDT
        if (!activeLoopOp || failed(createReductionAggregator(
                                 activeLoopCount, activeLoopOp, op)))
          return failure();
        activeReductions.clear();
        activeLoopOp = nullptr;
        activeLoopCount = Value();
      }
    } else {
      ARTS_INFO("Lowering simple parallel EDT");
      lowerSimpleParallelEdt(op);
    }

    /// Yield from epoch
    AC->create<YieldOp>(loc);

    /// Store dependencies before erasing the original parallel EDT
    SmallVector<Value> originalDeps = op.getDependencies();

    /// Erase the original parallel EDT first
    op.erase();

    /// Now erase any unused original DbAcquireOp operations
    SmallVector<Operation *> toErase;
    if (containedForOp) {
      for (Value dep : originalDeps) {
        if (auto acq = dep.getDefiningOp<DbAcquireOp>()) {
          if (llvm::all_of(acq->getResults(),
                           [](Value v) { return v.use_empty(); }))
            toErase.push_back(acq);
        }
      }
    }
    for (auto *e : toErase)
      e->erase();
    return success();
  }

private:
  void prepareReductionBuffers(ForOp forOp, Value slotCount);
  void appendReductionDeps(Value offset, SmallVector<Value> &deps);
  LogicalResult initializePrivateAccumulators(ArrayRef<Value> privateArgs,
                                              Block &body);
  Value createZeroValue(Type type);
  Value combineValues(Type type, Value lhs, Value rhs);
  Value remapDependencyValue(Value value, ValueRange sourceArgs,
                             ValueRange mappedValues);
  void remapDependencyList(ValueRange deps, ValueRange sourceArgs,
                           ValueRange mappedValues,
                           SmallVectorImpl<Value> &out);
  ReductionAccess computeReductionAccess(const ReductionInfo &info,
                                         Value buffer, Value linearIndex,
                                         Value zeroIndex, Value strideOne);
  LogicalResult createReductionAggregator(Value loopCount,
                                          Operation *insertAfter,
                                          EdtOp sourceOp);

  SmallVector<ReductionInfo> activeReductions;
  Value activeLoopCount;
  Operation *activeLoopOp = nullptr;

  /// Compute total iterations of an arts.for operation
  Value computeTotalIterations(ForOp forOp) {
    Value oneIndex = AC->createIndexConstant(1, loc);
    Value lower = forOp.getLowerBound()[0];
    Value upper = forOp.getUpperBound()[0];
    Value step = forOp.getStep()[0];
    Value range = AC->create<arith::SubIOp>(loc, upper, lower);
    Value adjustedRange = AC->create<arith::AddIOp>(
        loc, range, AC->create<arith::SubIOp>(loc, step, oneIndex));
    return AC->create<arith::DivUIOp>(loc, adjustedRange, step);
  }

  EpochOp createEpochWrapper() {
    auto epochOp = AC->create<EpochOp>(loc);
    auto &region = epochOp.getRegion();
    if (region.empty())
      region.push_back(new Block());
    return epochOp;
  }

  //===----------------------------------------------------------------------===//
  ///
  /// Lowering parallel EDT with arts.for using block distribution
  /// Block distribution example with 4 workers (threads=4):
  ///
  /// Input IR (no chunk_size):
  ///   arts.edt<parallel> {
  ///     arts.for %i = 0 to 100 step 1 {
  ///       /// loop body computing 100 iterations
  ///     }
  ///   }
  ///
  /// Output IR:
  ///   Total iterations: 100, Workers: 4
  ///   Worker 0: iterations 0-24   (25 iterations)
  ///   Worker 1: iterations 25-49  (25 iterations)
  ///   Worker 2: iterations 50-74  (25 iterations)
  ///   Worker 3: iterations 75-99  (25 iterations)
  ///
  ///   scf.for %worker = 0 to 4 step 1 {
  ///     arts.edt<task route=0> deps[...] {
  ///       scf.for %local_i = 0 to worker_chunk_count step 1 {
  ///         %global_i = worker_first_chunk * 1 + %local_i
  ///         /// cloned loop body with %global_i
  ///       }
  ///     }
  ///   }
  ///
  /// For uneven division (101 iterations):
  /// Worker 0: 26 iterations, Worker 1: 26 iterations
  /// Worker 2: 25 iterations, Worker 3: 24 iterations
  void lowerStaticBlock(EdtOp op, ForOp containedForOp, Value lb, Value ub,
                        bool isInterNode) {

    /// Create scf.for loop over workers
    Value one = AC->createIndexConstant(1, loc);
    scf::ForOp workerLoop = AC->create<scf::ForOp>(loc, lb, ub, one);
    OpBuilder::InsertionGuard IG(AC->getBuilder());
    AC->setInsertionPointToStart(workerLoop.getBody());

    Value workerId = workerLoop.getInductionVar();
    Value route = createRouteValue(workerId, isInterNode);

    /// Compute worker's iteration bounds
    ChunkedForInfo chunkInfo = computeChunkBounds(containedForOp, workerId, ub);

    /// Create dependencies for this worker
    auto chunkResult =
        createBlockDeps(op, containedForOp, workerId, ub, chunkInfo);
    SmallVector<bool> chunkedDepFlags(chunkResult.deps.size(), false);
    for (DbAcquireOp acquire : chunkResult.chunkedAcquires) {
      auto it = llvm::find(chunkResult.deps, acquire.getPtr());
      if (it != chunkResult.deps.end())
        chunkedDepFlags[it - chunkResult.deps.begin()] = true;
    }

    appendReductionDeps(workerId, chunkResult.deps);

    /// Create arts.edt<task> for this worker
    auto newEdt = AC->create<EdtOp>(
        loc, EdtType::task, EdtConcurrency::intranode, route, chunkResult.deps);
    setupEdtRegion(newEdt, op);

    AC->setInsertionPointToEnd(&newEdt.getRegion().front());
    SmallVector<BlockArgument> chunkedArgs;
    auto newEdtArgs = newEdt.getRegion().front().getArguments();
    for (auto [idx, arg] : llvm::enumerate(newEdtArgs))
      if (idx < chunkedDepFlags.size() && chunkedDepFlags[idx])
        chunkedArgs.push_back(arg);

    SmallVector<Value> privateAccArgs;
    if (!activeReductions.empty()) {
      Block &edtBody = newEdt.getRegion().front();
      auto dstArgs = edtBody.getArguments();
      size_t reductionArgStart = dstArgs.size() - activeReductions.size();
      for (unsigned i = 0; i < activeReductions.size(); ++i)
        privateAccArgs.push_back(dstArgs[reductionArgStart + i]);
      if (failed(initializePrivateAccumulators(privateAccArgs, edtBody)))
        return;
    }

    IRMapping mapper;
    auto srcArgs = op.getRegion().front().getArguments();
    auto dstArgs = newEdt.getRegion().front().getArguments();
    for (auto [srcArg, dstArg] : zip(srcArgs, dstArgs))
      mapper.map(srcArg, dstArg);

    /// Clone arts.for loop body into worker EDT
    Block &loopBody = containedForOp.getRegion().front();
    Value loopInductionVar = loopBody.getArgument(0);

    /// Create scf.for for worker's iterations
    Value zeroIndex = AC->createIndexConstant(0, loc);
    Value oneIndex = AC->createIndexConstant(1, loc);
    auto assignedLoop = AC->create<scf::ForOp>(
        loc, zeroIndex, chunkInfo.workerIterationCount, oneIndex);

    Block &assignedBody = assignedLoop.getRegion().front();
    Value localIndex = assignedLoop.getInductionVar();
    AC->setInsertionPointToStart(&assignedBody);

    /// Convert to global iteration number and loop index
    Value baseIter = AC->create<arith::MulIOp>(loc, chunkInfo.workerFirstChunk,
                                               chunkInfo.chunkSize);
    Value iterationNumber =
        AC->create<arith::AddIOp>(loc, baseIter, localIndex);

    IRMapping iterationMapper = mapper;
    Value actualLoopIndex = AC->create<arith::AddIOp>(
        loc, chunkInfo.lowerBound,
        AC->create<arith::MulIOp>(loc, iterationNumber, chunkInfo.loopStep));
    iterationMapper.map(loopInductionVar, actualLoopIndex);
    if (!activeReductions.empty()) {
      for (auto [idx, info] : llvm::enumerate(activeReductions))
        iterationMapper.map(info.originalAcc, privateAccArgs[idx]);
    }
    mapReductionIdentityValues(iterationMapper, containedForOp);

    /// Clone operations from arts.for body
    for (Operation &inner : loopBody.without_terminator()) {
      Operation *cloned = AC->clone(inner, iterationMapper);
      for (auto [oldRes, newRes] :
           llvm::zip(inner.getResults(), cloned->getResults()))
        iterationMapper.map(oldRes, newRes);
    }
    replaceUndefWithZero(assignedBody);

    /// Fix arts.db_ref indices for chunked datablocks
    SmallVector<DbRefOp> chunkedDbRefs;
    for (Operation &inner : assignedBody)
      if (auto dbRef = dyn_cast<DbRefOp>(&inner))
        if (dbRef.getSource().isa<BlockArgument>()) {
          auto barg = dbRef.getSource().cast<BlockArgument>();
          if (llvm::is_contained(chunkedArgs, barg))
            chunkedDbRefs.push_back(dbRef);
        }
    for (DbRefOp dbRef : chunkedDbRefs) {
      AC->setInsertionPoint(dbRef);
      Value globalIndex = iterationMapper.lookup(loopInductionVar);
      Value localIndexForDb =
          AC->create<arith::SubIOp>(loc, globalIndex, baseIter);
      dbRef.getIndicesMutable().assign(ValueRange{localIndexForDb});
    }

    /// Release chunked datablocks
    AC->setInsertionPointToEnd(&newEdt.getRegion().front());
    for (BlockArgument barg : chunkedArgs)
      AC->create<DbReleaseOp>(loc, barg);

    if (!privateAccArgs.empty()) {
      AC->setInsertionPointToEnd(&newEdt.getRegion().front());
      for (Value accArg : privateAccArgs)
        AC->create<DbReleaseOp>(loc, accArg);
    }

    AC->setInsertionPointToEnd(&newEdt.getRegion().front());
    if (!isa<YieldOp>(newEdt.getRegion().front().back()))
      AC->create<YieldOp>(loc);
    if (!activeReductions.empty())
      activeLoopOp = workerLoop.getOperation();
  }

  //===----------------------------------------------------------------------===//
  /// Create the dependencies for a given worker.
  //===----------------------------------------------------------------------===//
  ChunkedDepsResult createBlockDeps(EdtOp op, ForOp containedForOp,
                                    Value workerId, Value totalWorkers,
                                    const ChunkedForInfo &chunkInfo) {
    ChunkedDepsResult result;
    ValueRange srcArgs = op.getRegion().front().getArguments();
    ValueRange srcDeps = op.getDependencies();

    for (Value dep : srcDeps) {
      if (auto acq = dep.getDefiningOp<DbAcquireOp>()) {
        bool isReductionDep = llvm::any_of(activeReductions, [&](auto &info) {
          return acq.getSourcePtr() == info.originalAlloc.getPtr();
        });
        if (isReductionDep)
          continue;
        Value sourceGuid =
            remapDependencyValue(acq.getSourceGuid(), srcArgs, srcDeps);
        Value sourcePtr =
            remapDependencyValue(acq.getSourcePtr(), srcArgs, srcDeps);

        /// Calculate offset into datablock for this worker's chunk
        Value chunkOffset = AC->create<arith::MulIOp>(
            loc, chunkInfo.workerFirstChunk, chunkInfo.chunkSize);

        /// Calculate theoretical chunk size this worker should get
        Value chunkSizeElems = AC->create<arith::MulIOp>(
            loc, chunkInfo.workerChunkCount, chunkInfo.chunkSize);

        /// Clamp chunk size to available elements (handle edge case where
        /// worker's chunk extends beyond total iterations)
        Value zeroIndex = AC->createIndexConstant(0, loc);
        Value totalI = chunkInfo.totalIterations;
        Value needClamp = AC->create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, totalI, chunkOffset);
        Value diff = AC->create<arith::SubIOp>(loc, totalI, chunkOffset);
        Value remaining =
            AC->create<arith::SelectOp>(loc, needClamp, zeroIndex, diff);
        chunkSizeElems =
            AC->create<arith::MinUIOp>(loc, chunkSizeElems, remaining);

        /// Create arts.db_acquire for this worker's chunk of the datablock
        SmallVector<Value> offsets{chunkOffset};
        SmallVector<Value> sizes{chunkSizeElems};
        auto newAcquire =
            AC->create<DbAcquireOp>(loc, acq.getMode(), sourceGuid, sourcePtr,
                                    SmallVector<Value>{}, offsets, sizes);
        ARTS_DEBUG("  Created chunked DbAcquire "
                   << newAcquire << " block=" << newAcquire->getBlock());
        result.chunkedAcquires.push_back(newAcquire);
        result.deps.push_back(newAcquire.getPtr());
        continue;
      }

      /// Remap non-datablock dependencies (scalars, etc.)
      result.deps.push_back(remapDependencyValue(dep, srcArgs, srcDeps));
    }

    return result;
  }

  //===----------------------------------------------------------------------===//
  /// Compute the chunk bounds for a given worker.
  ///
  /// Block distribution example with 4 workers, loop 0 to 100:
  /// - Total iterations = (100 - 0) / 1 = 100
  /// - chunksPerWorker = 100 / 4 = 25
  /// - remainderChunks = 100 % 4 = 0
  //
  /// - Worker 0: firstChunk = 0*25 + min(0,0) = 0, count = 25+0 = 25
  /// - Worker 1: firstChunk = 1*25 + min(1,0) = 25, count = 25+0 = 25
  /// - Worker 2: firstChunk = 2*25 + min(2,0) = 50, count = 25+0 = 25
  /// - Worker 3: firstChunk = 3*25 + min(3,0) = 75, count = 25+0 = 25
  //
  /// Uneven example with 101 iterations:
  /// - Total iterations = 101, workers = 4
  /// - chunksPerWorker = 101/4 = 25, remainder = 101%4 = 1
  ///   - Worker 0: firstChunk = 0 + min(0,1) = 0, count = 25+1 = 26
  ///   - Worker 1: firstChunk = 25 + min(1,1) = 26, count = 25+1 = 26
  ///   - Worker 2: firstChunk = 50 + min(2,1) = 51, count = 25+0 = 25
  ///   - Worker 3: firstChunk = 75 + min(3,1) = 76, count = 25+0 = 25
  //===----------------------------------------------------------------------===//
  ChunkedForInfo computeChunkBounds(ForOp containedForOp, Value workerId,
                                    Value totalWorkers) {
    ChunkedForInfo info;

    info.lowerBound = containedForOp.getLowerBound()[0];
    info.upperBound = containedForOp.getUpperBound()[0];
    info.loopStep = containedForOp.getStep()[0];

    /// Calculate total iterations: (upper - lower) / step
    Value oneIndex = AC->createIndexConstant(1, loc);
    Value range =
        AC->create<arith::SubIOp>(loc, info.upperBound, info.lowerBound);
    Value adjustedRange = AC->create<arith::AddIOp>(
        loc, range, AC->create<arith::SubIOp>(loc, info.loopStep, oneIndex));
    info.totalIterations =
        AC->create<arith::DivUIOp>(loc, adjustedRange, info.loopStep);

    /// For block distribution, chunk_size = 1
    info.chunkSize = oneIndex;

    /// Each iteration is a "chunk"
    info.totalChunks = info.totalIterations;

    /// Divide total chunks evenly across workers
    /// chunksPerEDT = totalChunks / totalWorkers
    Value chunksPerEDT =
        AC->create<arith::DivUIOp>(loc, info.totalChunks, totalWorkers);

    /// Remainder chunks
    /// remainderChunks = totalChunks % totalWorkers
    Value remainderChunks =
        AC->create<arith::RemUIOp>(loc, info.totalChunks, totalWorkers);

    /// First few workers get an extra chunk if remainder > 0
    /// getsExtra = workerId < remainderChunks
    Value getsExtra = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                workerId, remainderChunks);
    /// extraChunk = getsExtra ? 1 : 0
    Value extraChunk = AC->castToIndex(getsExtra, loc);

    /// Total chunks this worker should handle
    /// workerChunkCount = chunksPerEDT + extraChunk
    info.workerChunkCount =
        AC->create<arith::AddIOp>(loc, chunksPerEDT, extraChunk);

    /// Total number of workers
    info.totalWorkers = totalWorkers;

    /// Calculate which chunk this worker starts with:
    ///  workerFirstChunk = workerId * chunksPerWorker + min(workerId,remainder)
    /// Base start
    Value baseStart = AC->create<arith::MulIOp>(loc, workerId, chunksPerEDT);
    /// Extra chunks this worker gets
    Value prefixExtra =
        AC->create<arith::MinUIOp>(loc, workerId, remainderChunks);
    /// First chunk this worker handles
    info.workerFirstChunk =
        AC->create<arith::AddIOp>(loc, baseStart, prefixExtra);

    /// Calculate actual iteration count
    /// rawIterCount = workerChunkCount * chunkSize
    Value rawIterCount =
        AC->create<arith::MulIOp>(loc, info.workerChunkCount, info.chunkSize);

    /// First element this worker handles
    /// startElems = workerFirstChunk * chunkSize
    Value startElems =
        AC->create<arith::MulIOp>(loc, info.workerFirstChunk, info.chunkSize);

    /// Handle edge case: if worker starts past total iterations
    Value zeroIndex = AC->createIndexConstant(0, loc);
    Value needZero = AC->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, info.totalIterations, startElems);

    /// Calculate how many iterations are actually available to this worker
    Value remaining =
        AC->create<arith::SubIOp>(loc, info.totalIterations, startElems);

    /// Clamp to zero if worker starts beyond end (shouldn't happen in practice)
    Value remainingNonNeg =
        AC->create<arith::SelectOp>(loc, needZero, zeroIndex, remaining);

    /// Final iteration count = min(theoretical count, available count)
    info.workerIterationCount =
        AC->create<arith::MinUIOp>(loc, rawIterCount, remainingNonNeg);

    return info;
  }

  //===----------------------------------------------------------------------===//
  /// Cyclic distribution example with 4 workers (threads=4):
  ///
  ///  Assigning chunks to workers in a cyclic manner.
  ///  Example:
  /// - Input IR (with chunk_size=5):
  ///   arts.edt<parallel> {
  ///     arts.for %i = 0 to 100 step 1 chunk_size=5 {
  ///       /// loop body
  ///     }
  ///   }
  ///
  /// - Output IR:
  ///  Total iterations: 100, Chunk size: 5, Total chunks: 20
  ///  -Worker 0: chunks 0,4,8,12,16  (iterations 0-4,20-24,40-44,60-64,80-84)
  ///  -Worker 1: chunks 1,5,9,13,17  (iterations 5-9,25-29,45-49,65-69,85-89)
  ///  -Worker 2: chunks 2,6,10,14,18 (iterations 10-14,30-34,50-54,70-74,90-94)
  ///  -Worker 3: chunks 3,7,11,15,19 (iterations 15-19,35-39,55-59,75-79,95-99)
  ///
  ///   scf.for %chunk = 0 to 20 step 1 {
  ///     %worker = %chunk % 4  /// cyclic assignment
  ///     arts.edt<task route=0> deps[...] {
  ///       scf.for %local_i = 0 to 5 step 1 {
  ///         %global_i = %chunk * 5 + %local_i
  ///         /// cloned loop body
  ///       }
  ///     }
  ///   }
  //===----------------------------------------------------------------------===//
  void lowerStaticCyclic(EdtOp op, ForOp containedForOp, Value lb, Value ub,
                         bool isInterNode) {

    LoopInfo info;
    info.lowerBound = containedForOp.getLowerBound()[0];
    info.upperBound = containedForOp.getUpperBound()[0];
    info.loopStep = containedForOp.getStep()[0];
    info.totalWorkers = ub;

    computeTotalIterationsAndChunkSize(containedForOp, info);

    /// Create scf.for loop over chunks
    Value zero = AC->createIndexConstant(0, loc);
    Value one = AC->createIndexConstant(1, loc);
    scf::ForOp chunkLoop =
        AC->create<scf::ForOp>(loc, zero, info.totalChunks, one);

    OpBuilder::InsertionGuard IG(AC->getBuilder());
    AC->setInsertionPointToStart(chunkLoop.getBody());

    Value chunkId = chunkLoop.getInductionVar();

    /// Set route based on inter/intra-node execution
    Value route;
    if (isInterNode) {
      /// Assign chunk to worker: workerId = chunkId % totalWorkers
      Value nodeId =
          AC->create<arith::RemUIOp>(loc, chunkId, info.totalWorkers);
      route = AC->castToInt(AC->Int32, nodeId, loc);
    } else {
      route = AC->createIntConstant(0, AC->Int32, loc);
    }

    /// Create dependencies for this chunk
    ChunkedDepsResult chunkResult = createCyclicDeps(
        op, containedForOp, chunkId, info.chunkSize, info.totalIterations);
    SmallVector<bool> chunkedDepFlags(chunkResult.deps.size(), false);
    for (DbAcquireOp acquire : chunkResult.chunkedAcquires) {
      auto it = llvm::find(chunkResult.deps, acquire.getPtr());
      if (it != chunkResult.deps.end())
        chunkedDepFlags[it - chunkResult.deps.begin()] = true;
    }

    appendReductionDeps(chunkId, chunkResult.deps);

    /// Create arts.edt<task> for this chunk
    auto newEdt = AC->create<EdtOp>(
        loc, EdtType::task, EdtConcurrency::intranode, route, chunkResult.deps);
    setupEdtRegion(newEdt, op);

    AC->setInsertionPointToEnd(&newEdt.getRegion().front());
    SmallVector<BlockArgument> chunkedArgs;
    auto newEdtArgs = newEdt.getRegion().front().getArguments();
    for (auto [idx, arg] : llvm::enumerate(newEdtArgs))
      if (idx < chunkedDepFlags.size() && chunkedDepFlags[idx])
        chunkedArgs.push_back(arg);

    SmallVector<Value> privateAccArgs;
    if (!activeReductions.empty()) {
      Block &edtBody = newEdt.getRegion().front();
      auto dstArgs = edtBody.getArguments();
      size_t reductionArgStart = dstArgs.size() - activeReductions.size();
      for (unsigned i = 0; i < activeReductions.size(); ++i)
        privateAccArgs.push_back(dstArgs[reductionArgStart + i]);
      if (failed(initializePrivateAccumulators(privateAccArgs, edtBody)))
        return;
    }

    IRMapping mapper;
    auto srcArgs = op.getRegion().front().getArguments();
    auto dstArgs = newEdt.getRegion().front().getArguments();
    for (auto [srcArg, dstArg] : zip(srcArgs, dstArgs))
      mapper.map(srcArg, dstArg);

    /// Calculate which iterations this chunk should execute
    Block &loopBody = containedForOp.getRegion().front();
    Value loopInductionVar = loopBody.getArgument(0);

    /// Calculate which iterations this chunk should execute
    /// chunkStartIter = chunkId * chunkSize
    /// Example: chunkId=0, chunkSize=5 -> chunkStartIter = 0*5 = 0 (iterations
    /// 0,1,2,3,4) Example: chunkId=4, chunkSize=5 -> chunkStartIter = 4*5 = 20
    /// (iterations 20,21,22,23,24) Example: chunkId=19, chunkSize=5 ->
    /// chunkStartIter = 19*5 = 95 (iterations 95,96,97,98,99)
    Value chunkStartIter =
        AC->create<arith::MulIOp>(loc, chunkId, info.chunkSize);
    /// chunkStartIter = chunkId * chunkSize

    /// Handle last chunk: may have fewer than chunkSize iterations
    /// Example: totalIterations=100, chunkId=19, chunkSize=5
    /// remaining = 100 - 95 = 5, so actualChunkSize = min(5, 5) = 5 ✓
    /// Example: totalIterations=102, chunkId=20, chunkSize=5 (hypothetical)
    /// remaining = 102 - 100 = 2, so actualChunkSize = min(5, 2) = 2 ✓
    Value remaining =
        AC->create<arith::SubIOp>(loc, info.totalIterations, chunkStartIter);
    /// remaining = totalIterations - chunkStartIter

    Value actualChunkSize =
        AC->create<arith::MinUIOp>(loc, info.chunkSize, remaining);
    /// actualChunkSize = min(chunkSize, remaining)

    /// Create inner loop: scf.for %localIndex = 0 to actualChunkSize step 1
    /// Example: chunkSize=5, not last chunk -> %localIndex ∈ [0,4]
    /// Example: last chunk with only 3 remaining -> %localIndex ∈ [0,2]
    auto assignedLoop = AC->create<scf::ForOp>(loc, zero, actualChunkSize, one);

    Block &assignedBody = assignedLoop.getRegion().front();
    AC->setInsertionPointToStart(&assignedBody);

    Value localIndex = assignedLoop.getInductionVar();

    /// iterationNumber = chunkStartIter + localIndex
    /// Example: chunkId=0, localIndex=0 -> 0+0=0, chunkId=4, localIndex=4 ->
    /// 20+4=24
    /// iterationNumber = chunkStartIter + localIndex
    Value iterationNumber =
        AC->create<arith::AddIOp>(loc, chunkStartIter, localIndex);

    /// Convert to actual loop index: actual_index = lowerBound +
    /// iterationNumber * step Example: for i=0 to 100 step 1: actual_index = 0
    /// + iterationNumber * 1
    IRMapping iterationMapper = mapper;
    /// actualLoopIndex = lowerBound + iterationNumber * step
    Value actualLoopIndex = AC->create<arith::AddIOp>(
        loc, info.lowerBound,
        AC->create<arith::MulIOp>(loc, iterationNumber, info.loopStep));

    /// Map the original loop induction variable to our computed actual index
    iterationMapper.map(loopInductionVar, actualLoopIndex);
    if (!activeReductions.empty()) {
      for (auto [idx, info] : llvm::enumerate(activeReductions))
        iterationMapper.map(info.originalAcc, privateAccArgs[idx]);
    }
    mapReductionIdentityValues(iterationMapper, containedForOp);

    for (Operation &inner : loopBody.without_terminator()) {
      Operation *cloned = AC->clone(inner, iterationMapper);
      for (auto [oldRes, newRes] :
           llvm::zip(inner.getResults(), cloned->getResults()))
        iterationMapper.map(oldRes, newRes);
    }
    replaceUndefWithZero(assignedBody);

    SmallVector<DbRefOp> chunkedDbRefs;
    for (Operation &inner : assignedBody) {
      if (auto dbRef = dyn_cast<DbRefOp>(&inner))
        if (dbRef.getSource().isa<BlockArgument>()) {
          auto barg = dbRef.getSource().cast<BlockArgument>();
          if (llvm::is_contained(chunkedArgs, barg))
            chunkedDbRefs.push_back(dbRef);
        }
    }

    for (DbRefOp dbRef : chunkedDbRefs) {
      AC->setInsertionPoint(dbRef);
      Value globalIndex = iterationMapper.lookup(loopInductionVar);
      Value localIndexForDb =
          AC->create<arith::SubIOp>(loc, globalIndex, chunkStartIter);
      dbRef.getIndicesMutable().assign(ValueRange{localIndexForDb});
    }
    AC->setInsertionPointToEnd(&assignedBody);

    AC->setInsertionPointToEnd(&newEdt.getRegion().front());
    for (BlockArgument barg : chunkedArgs)
      AC->create<DbReleaseOp>(loc, barg);

    if (!privateAccArgs.empty()) {
      AC->setInsertionPointToEnd(&newEdt.getRegion().front());
      for (Value accArg : privateAccArgs)
        AC->create<DbReleaseOp>(loc, accArg);
    }

    AC->setInsertionPointToEnd(&newEdt.getRegion().front());
    if (!isa<YieldOp>(newEdt.getRegion().front().back()))
      AC->create<YieldOp>(loc);
    if (!activeReductions.empty())
      activeLoopOp = chunkLoop.getOperation();
  }

  //===----------------------------------------------------------------------===//
  /// Simple parallel EDT with 4 workers (threads=4):
  ///
  /// Creates identical task EDTs for embarrassingly parallel workloads
  /// - Input IR (no loop, just computation):
  ///   arts.edt<parallel> {
  ///     /// some computation work
  ///   }
  ///
  /// - Output IR:
  ///   scf.for %worker = 0 to 4 step 1 {
  ///     arts.edt<task route=0> deps[...] {
  ///       /// cloned computation work
  ///     }
  ///   }
  //===----------------------------------------------------------------------===//
  void lowerSimpleParallelEdt(EdtOp op) {
    auto [lb, ub, isInterNode] = determineParallelismStrategy(op);
    ARTS_INFO("Lowering simple parallel EDT; bounds: " << lb << ".." << ub);

    Value one = AC->createIndexConstant(1, loc);
    /// Create worker loop:
    /// for(workerId = 0; workerId < totalWorkers; workerId++)
    scf::ForOp workerLoop = AC->create<scf::ForOp>(loc, lb, ub, one);
    OpBuilder::InsertionGuard IG(AC->getBuilder());
    AC->setInsertionPointToStart(workerLoop.getBody());

    ValueRange srcArgs = op.getRegion().front().getArguments();
    ValueRange srcDeps = op.getDependencies();
    SmallVector<Value> deps;
    remapDependencyList(srcDeps, srcArgs, srcDeps, deps);
    Value workerId = workerLoop.getInductionVar();
    Value route = createRouteValue(workerId, isInterNode);

    /// Create arts.edt with intranode concurrency and deps for each worker
    auto newEdt = AC->create<EdtOp>(loc, EdtType::task,
                                    EdtConcurrency::intranode, route, deps);
    ARTS_INFO("Created task EDT with route " << route);
    setupEdtRegion(newEdt, op);

    /// Clone the original EDT's body into each worker's task EDT
    moveOperationsToNewEdt(op, newEdt, nullptr, Value());
  }

  //===----------------------------------------------------------------------===//
  /// Setup EDT region structure
  ///
  /// Initializes the region and block structure for a new EDT operation.
  /// Creates block arguments for each dependency and prepares the EDT
  /// for receiving cloned operations from the source EDT.
  //===----------------------------------------------------------------------===//
  void setupEdtRegion(EdtOp edt, EdtOp sourceEdt) {
    Region &dst = edt.getRegion();
    if (dst.empty())
      dst.push_back(new Block());

    Block &dstBody = dst.front();
    dstBody.clear();

    auto deps = edt.getDependencies();
    for (Value dep : deps)
      dstBody.addArgument(dep.getType(), loc);
  }

  //===----------------------------------------------------------------------===//
  /// Determine parallelism strategy for EDT
  ///
  /// Analyzes EDT attributes to determine the number of workers and whether
  /// the parallelism is intra-node or inter-node. Uses concurrency pass
  /// attributes or falls back to runtime worker count. Returns: (lower_bound=0,
  /// upper_bound=worker_count, is_inter_node)
  //===----------------------------------------------------------------------===//
  std::tuple<Value, Value, bool> determineParallelismStrategy(EdtOp edtOp) {
    Value lb = AC->createIndexConstant(0, loc);
    Value ub = nullptr;

    /// Read parallelism strategy from EDT attributes set by concurrency pass
    if (auto workersAttr = edtOp.getWorkers()) {
      int numWorkers = workersAttr->getValue();
      ub = AC->createIndexConstant(numWorkers, loc);

      bool isInterNode = (edtOp.getConcurrency() == EdtConcurrency::internode);
      ARTS_INFO("Using EDT attributes: workers="
                << numWorkers << ", concurrency="
                << (isInterNode ? "internode" : "intranode"));
      return {lb, ub, isInterNode};
    }

    /// Fallback: no attributes set, use runtime workers
    ARTS_WARN("No parallelism attributes set on EDT, using runtime fallback");
    ub = AC->castToIndex(AC->getTotalWorkers(loc), loc);
    return {lb, ub, /*isInterNode=*/false};
  }

  ForOp findContainedForOp(EdtOp op) {
    ForOp containedForOp = nullptr;
    op.walk([&](ForOp forOp) {
      if (!containedForOp)
        containedForOp = forOp;
      return WalkResult::advance();
    });
    return containedForOp;
  }

  ChunkedDepsResult createCyclicDeps(EdtOp op, ForOp containedForOp,
                                     Value chunkId, Value chunkSize,
                                     Value totalIterations) {
    ChunkedDepsResult result;
    ValueRange srcArgs = op.getRegion().front().getArguments();
    ValueRange srcDeps = op.getDependencies();

    Value chunkOffset = AC->create<arith::MulIOp>(loc, chunkId, chunkSize);
    Value remaining =
        AC->create<arith::SubIOp>(loc, totalIterations, chunkOffset);
    Value chunkSizeElems =
        AC->create<arith::MinUIOp>(loc, chunkSize, remaining);

    for (Value dep : srcDeps) {
      if (auto acq = dep.getDefiningOp<DbAcquireOp>()) {
        bool isReductionDep = llvm::any_of(activeReductions, [&](auto &info) {
          return acq.getSourcePtr() == info.originalAlloc.getPtr();
        });
        if (isReductionDep)
          continue;
        Value sourceGuid =
            remapDependencyValue(acq.getSourceGuid(), srcArgs, srcDeps);
        Value sourcePtr =
            remapDependencyValue(acq.getSourcePtr(), srcArgs, srcDeps);
        SmallVector<Value> offsets{chunkOffset};
        SmallVector<Value> sizes{chunkSizeElems};
        auto newAcquire =
            AC->create<DbAcquireOp>(loc, acq.getMode(), sourceGuid, sourcePtr,
                                    SmallVector<Value>{}, offsets, sizes);
        ARTS_DEBUG("  Created cyclic DbAcquire "
                   << newAcquire << " block=" << newAcquire->getBlock());
        result.chunkedAcquires.push_back(newAcquire);
        result.deps.push_back(newAcquire.getPtr());
        continue;
      }

      result.deps.push_back(remapDependencyValue(dep, srcArgs, srcDeps));
    }

    return result;
  }

  void computeTotalIterationsAndChunkSize(ForOp containedForOp,
                                          LoopInfo &info) {
    /// Calculate total chunks: ceil(totalIterations / chunkSize)
    /// Formula: (totalIterations + chunkSize - 1) / chunkSize
    /// Example: 100 iterations, chunkSize=5 -> (100 + 5 - 1) / 5 = 104 / 5 = 20

    Value oneIndex = AC->createIndexConstant(1, loc);
    /// Calculate total iterations: (upper - lower) / step
    Value range =
        AC->create<arith::SubIOp>(loc, info.upperBound, info.lowerBound);
    Value adjustedRange = AC->create<arith::AddIOp>(
        loc, range, AC->create<arith::SubIOp>(loc, info.loopStep, oneIndex));
    info.totalIterations =
        AC->create<arith::DivUIOp>(loc, adjustedRange, info.loopStep);

    /// Get chunk_size attribute from the arts.for operation
    auto chunkAttr = containedForOp->getAttrOfType<IntegerAttr>("chunk_size");
    int64_t chunkInt = std::max<int64_t>(1, chunkAttr.getInt());
    info.chunkSize = AC->createIndexConstant(chunkInt, loc);

    /// Calculate total chunks: ceil(totalIterations / chunkSize)
    /// Formula: (totalIterations + chunkSize - 1) / chunkSize
    /// - Example: 100 iterations, chunkSize=5 ->
    ///   adjustedIterations = 100 + 5 - 1 = 104
    ///   totalChunks = 104 / 5 = 20 (ceil(100/5) = 20)
    //
    /// - Example: 101 iterations, chunkSize=5 ->
    ///   adjustedIterations = 101 + 5 - 1 = 105 t
    ///   totalChunks = 105 / 5 = 21
    ///   (ceil(101/5) = 21, since 5*20 = 100 < 101)

    /// adjustedIterations = totalIterations + chunkSize - 1
    Value adjustedIterations = AC->create<arith::AddIOp>(
        loc, info.totalIterations,
        AC->create<arith::SubIOp>(loc, info.chunkSize, oneIndex));

    /// Integer division gives ceiling
    /// totalChunks = adjustedIterations / chunkSize)
    info.totalChunks =
        AC->create<arith::DivUIOp>(loc, adjustedIterations, info.chunkSize);
  }

  //===----------------------------------------------------------------------===//
  /// Move operations from source EDT to new EDT
  ///
  /// Clones operations from the source EDT region into the target EDT region,
  /// mapping block arguments and optionally skipping certain operations.
  /// Handles the transfer of computation from one EDT to another during
  /// parallel EDT lowering transformations.
  /// Example:
  ///   Source EDT with body operations -> cloned into target EDT with mapped
  ///   arguments
  //===----------------------------------------------------------------------===//
  void moveOperationsToNewEdt(EdtOp sourceOp, EdtOp newEdt,
                              Operation *operationToSkip, Value releaseValue) {
    Block &dstBody = newEdt.getRegion().front();
    Block &srcBody = sourceOp.getRegion().front();

    IRMapping mapper;

    for (auto [srcArg, dstArg] :
         zip(srcBody.getArguments(), dstBody.getArguments())) {
      mapper.map(srcArg, dstArg);
    }

    OpBuilder bodyBuilder(newEdt);
    bodyBuilder.setInsertionPointToEnd(&dstBody);

    /// Clone operations in forward order so producers are cloned before
    /// consumers
    SmallVector<Operation *> opsToErase;
    for (Operation &op : srcBody.without_terminator()) {
      if (&op != operationToSkip) {
        opsToErase.push_back(&op);
        bodyBuilder.clone(op, mapper);
      }
    }

    /// Erase operations in reverse order - consumers before producers
    for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it)
      (*it)->erase();

    if (dstBody.empty() || !isa<YieldOp>(dstBody.back())) {
      bodyBuilder.setInsertionPointToEnd(&dstBody);
      bodyBuilder.create<YieldOp>(newEdt.getLoc());
    }
  }

  Value createRouteValue(Value workerId, bool isInterNode) {
    if (isInterNode)
      return AC->castToInt(AC->Int32, workerId, loc);
    else
      return AC->createIntConstant(0, AC->Int32, loc);
  }

  void mapReductionIdentityValues(IRMapping &mapper, ForOp forOp) {
    if (activeReductions.empty())
      return;

    Block &body = forOp.getRegion().front();
    for (Operation &op : body.without_terminator()) {
      for (Value operand : op.getOperands()) {
        if (auto undef = operand.getDefiningOp<LLVM::UndefOp>()) {
          if (mapper.contains(operand))
            continue;
          OpBuilder::InsertionGuard IG(AC->getBuilder());
          Value zero = createZeroValue(undef.getType());
          if (zero)
            mapper.map(operand, zero);
        }
      }
    }
  }

  void replaceUndefWithZero(Block &body) {
    SmallVector<LLVM::UndefOp, 4> undefs;
    for (Operation &op : body)
      if (auto undef = dyn_cast<LLVM::UndefOp>(&op))
        undefs.push_back(undef);

    for (LLVM::UndefOp undef : undefs) {
      OpBuilder::InsertionGuard IG(AC->getBuilder());
      AC->setInsertionPoint(undef);
      Value zero = createZeroValue(undef.getType());
      if (!zero)
        continue;
      undef.replaceAllUsesWith(zero);
      undef.erase();
    }
  }
};

/// Prepare reduction buffers for the given arts.for loop.
/// Clears the active reductions list and creates partial result datablocks
/// for each reduction accumulator found in the for loop. Each partial buffer
/// is allocated with the specified slot count to store intermediate results
/// from parallel worker EDTs before final aggregation.
void ParallelEdtLowering::prepareReductionBuffers(ForOp forOp,
                                                  Value slotCount) {
  assert(forOp && "ForOp cannot be null when preparing reduction buffers");
  assert(slotCount &&
         "SlotCount cannot be null when preparing reduction buffers");

  activeReductions.clear();

  ValueRange reductionAccs = forOp.getReductionAccumulators();
  if (reductionAccs.empty())
    return;

  Value zeroRoute = AC->createIntConstant(0, AC->Int32, loc);
  for (Value acc : reductionAccs) {
    auto memType = acc.getType().dyn_cast<MemRefType>();
    assert(memType && "Reduction accumulator must be a memref value");

    Operation *underlying = arts::getUnderlyingDbAlloc(acc);
    auto dbAllocOp = dyn_cast_or_null<DbAllocOp>(underlying);
    assert(dbAllocOp &&
           "Missing datablock allocation for reduction accumulator");

    Type scalarType = dbAllocOp.getElementType();
    assert(scalarType.isIntOrIndexOrFloat() &&
           "Only integer, index, or floating-point reductions are supported");

    SmallVector<Value> elementSizes(dbAllocOp.getElementSizes().begin(),
                                    dbAllocOp.getElementSizes().end());
    SmallVector<int64_t> elementShape;
    if (!elementSizes.empty())
      elementShape.assign(elementSizes.size(), ShapedType::kDynamic);
    MemRefType elementBufferType = MemRefType::get(elementShape, scalarType);

    SmallVector<Value> partialSizes{slotCount};
    auto partialAlloc = AC->create<DbAllocOp>(
        loc, ArtsMode::inout, zeroRoute, DbAllocType::heap, DbMode::write,
        scalarType, partialSizes, elementSizes);

    ReductionInfo info;
    info.originalAcc = acc;
    info.originalType = memType;
    info.elementBufferType = elementBufferType;
    info.scalarType = scalarType;
    info.partialAlloc = partialAlloc;
    info.originalAlloc = dbAllocOp;
    activeReductions.push_back(info);
  }
}

/// Append reduction dependencies to the dependency list for a worker EDT.
///
/// Each worker EDT needs write access to its designated slot in the partial
/// result datablocks for each reduction variable. This function creates
/// DbAcquire operations in "out" mode for each active reduction accumulator,
/// using the provided offset as the slot index within the partial buffers.
void ParallelEdtLowering::appendReductionDeps(Value offset,
                                              SmallVector<Value> &deps) {
  if (activeReductions.empty())
    return;

  Value oneIndex = AC->createIndexConstant(1, loc);
  for (ReductionInfo &info : activeReductions) {
    SmallVector<Value> offsets{offset};
    SmallVector<Value> sizes{oneIndex};
    DbAllocOp partialAlloc = info.partialAlloc;
    auto acquire = AC->create<DbAcquireOp>(
        loc, ArtsMode::out, partialAlloc.getGuid(), partialAlloc.getPtr(),
        SmallVector<Value>{}, offsets, sizes);
    deps.push_back(acquire.getPtr());
  }
}

/// Initialize private accumulator variables for a worker EDT.
///
/// Each worker EDT receives private accumulator arguments that correspond to
/// its slot in the partial result datablocks. This function initializes these
/// accumulators to zero so workers can start accumulating their partial
/// results. It iterates through each private accumulator argument, creates
/// a zero value of the appropriate type, and stores it in the accumulator's
/// memory location.
LogicalResult
ParallelEdtLowering::initializePrivateAccumulators(ArrayRef<Value> privateArgs,
                                                   Block &body) {
  if (privateArgs.empty())
    return success();

  OpBuilder::InsertionGuard IG(AC->getBuilder());
  AC->setInsertionPointToStart(&body);
  Value zeroIndex = AC->createIndexConstant(0, loc);
  Value oneIndex = AC->createIndexConstant(1, loc);
  for (auto [idx, arg] : llvm::enumerate(privateArgs)) {
    ReductionInfo &info = activeReductions[idx];
    Value zero = createZeroValue(info.scalarType);
    if (!zero)
      return emitOptionalError(loc, "Unsupported reduction element type for "
                                    "private accumulator initialization");
    auto ptrType = arg.getType().dyn_cast<MemRefType>();
    if (!ptrType)
      return emitOptionalError(
          loc, "Expected memref pointer for private reduction accumulator");

    ReductionAccess access =
        computeReductionAccess(info, arg, zeroIndex, zeroIndex, oneIndex);
    AC->create<memref::StoreOp>(loc, zero, access.buffer, access.indices);
  }
  AC->setInsertionPointToEnd(&body);
  return success();
}

/// Create a zero value of the specified type for reduction initialization.
///
/// This utility function creates appropriate zero constants for different
/// numeric types used in reduction operations. It handles index, integer,
/// and floating-point types, returning an appropriate zero value that can
/// be used to initialize reduction accumulators or as a starting value
/// for reduction aggregation.
///
/// Returns an empty Value() if the type is not supported for reductions.
Value ParallelEdtLowering::createZeroValue(Type type) {
  if (type.isa<IndexType>())
    return AC->createIndexConstant(0, loc);
  if (auto intTy = type.dyn_cast<IntegerType>())
    return AC->createIntConstant(0, intTy, loc);
  if (auto floatTy = type.dyn_cast<FloatType>())
    return AC->create<arith::ConstantOp>(
        loc, floatTy, AC->getBuilder().getFloatAttr(floatTy, 0.0));
  return Value();
}

/// Combine two values using the appropriate reduction operation.
///
/// Performs the core reduction operation by combining two
/// values of the same type. For integer and index types, it uses integer
/// addition. For floating-point types, it uses floating-point addition.
Value ParallelEdtLowering::combineValues(Type type, Value lhs, Value rhs) {
  if (type.isa<IntegerType>() || type.isa<IndexType>())
    return AC->create<arith::AddIOp>(loc, lhs, rhs).getResult();

  if (type.isa<FloatType>())
    return AC->create<arith::AddFOp>(loc, lhs, rhs).getResult();

  /// This should never happen as types are validated earlier
  llvm_unreachable("Unsupported reduction element type");
}

Value ParallelEdtLowering::remapDependencyValue(Value value,
                                                ValueRange sourceArgs,
                                                ValueRange mappedValues) {
  if (auto arg = value.dyn_cast<BlockArgument>()) {
    unsigned index = arg.getArgNumber();
    if (index < mappedValues.size())
      return mappedValues[index];
  }

  if (auto constOp = value.getDefiningOp<arith::ConstantOp>())
    return AC->clone(*constOp)->getResult(0);

  return value;
}

void ParallelEdtLowering::remapDependencyList(ValueRange deps,
                                              ValueRange sourceArgs,
                                              ValueRange mappedValues,
                                              SmallVectorImpl<Value> &out) {
  out.clear();
  out.reserve(deps.size());
  for (Value dep : deps)
    out.push_back(remapDependencyValue(dep, sourceArgs, mappedValues));
}

ParallelEdtLowering::ReductionAccess
ParallelEdtLowering::computeReductionAccess(const ReductionInfo &info,
                                            Value buffer, Value linearIndex,
                                            Value zeroIndex, Value strideOne) {
  assert(buffer && "Buffer cannot be null in computeReductionAccess");
  assert(info.originalType &&
         "Original type cannot be null in computeReductionAccess");

  ReductionAccess access;
  unsigned rank = info.originalType.getRank();
  SmallVector<Value> outerIdx(rank, zeroIndex);
  if (rank > 0)
    outerIdx.front() = linearIndex;
  SmallVector<Value> strides(rank, strideOne);
  Value elementPtr = AC->create<DbGepOp>(loc, AC->getLLVMPointerType(buffer),
                                         buffer, outerIdx, strides);
  access.buffer =
      AC->create<LLVM::LoadOp>(loc, info.elementBufferType, elementPtr)
          .getResult();
  unsigned innerRank = info.elementBufferType.getRank();
  access.indices.assign(innerRank, zeroIndex);
  return access;
}

/// Create reduction aggregator EDT.
///
/// Creates an arts.edt<task> that combines partial reduction results
/// from all worker EDTs back into the original accumulators.
LogicalResult ParallelEdtLowering::createReductionAggregator(
    Value loopCount, Operation *insertAfter, EdtOp sourceOp) {
  assert(loopCount && "LoopCount cannot be null when creating aggregator");
  assert(insertAfter && "InsertAfter cannot be null when creating aggregator");

  if (activeReductions.empty())
    return success();

  OpBuilder::InsertionGuard IG(AC->getBuilder());
  AC->setInsertionPointAfter(insertAfter);

  SmallVector<Value> partialPtrs, mappedAccs;
  Value zeroIndex = AC->createIndexConstant(0, loc);
  Value oneIndex = AC->createIndexConstant(1, loc);
  ARTS_DEBUG("Creating reduction aggregator with " << activeReductions.size()
                                                   << " accumulators");

  /// Create arts.db_acquire for partial results and original accumulators
  for (ReductionInfo &info : activeReductions) {
    /// Acquire partial result datablock (read-only)
    SmallVector<Value> offsets{zeroIndex};
    SmallVector<Value> sizes{loopCount};
    DbAllocOp partialAlloc = info.partialAlloc;
    auto partialAcquire = AC->create<DbAcquireOp>(
        loc, ArtsMode::in, partialAlloc.getGuid(), partialAlloc.getPtr(),
        SmallVector<Value>{}, offsets, sizes);
    partialPtrs.push_back(partialAcquire.getPtr());

    /// Acquire original accumulator (read-write)
    SmallVector<Value> origOffsets{zeroIndex};
    SmallVector<Value> origSizes{oneIndex};
    auto origAcquire = AC->create<DbAcquireOp>(
        loc, ArtsMode::inout, info.originalAlloc.getGuid(),
        info.originalAlloc.getPtr(), SmallVector<Value>{}, origOffsets,
        origSizes);
    mappedAccs.push_back(origAcquire.getPtr());
  }

  /// Create arts.edt<task> aggregator with all dependencies
  Value zeroRoute = AC->createIntConstant(0, AC->Int32, loc);
  SmallVector<Value> allDeps = partialPtrs;
  allDeps.append(mappedAccs.begin(), mappedAccs.end());

  auto reducerEdt = AC->create<EdtOp>(
      loc, EdtType::task, EdtConcurrency::intranode, zeroRoute, allDeps);
  setupEdtRegion(reducerEdt, sourceOp);
  Block &aggBlock = reducerEdt.getRegion().front();
  ValueRange aggArgsRange = aggBlock.getArguments();
  ValueRange depArgs = aggArgsRange.take_front(partialPtrs.size());
  ValueRange origArgs = aggArgsRange.drop_front(partialPtrs.size());

  /// Implement aggregation logic for each reduction variable
  AC->setInsertionPointToStart(&aggBlock);
  for (auto [idx, info] : llvm::enumerate(activeReductions)) {
    Value depArg = depArgs[idx];
    Value origArg = origArgs[idx];
    Value initial = createZeroValue(info.scalarType);
    assert(initial && "Unsupported reduction element type for aggregator");

    /// Create scf.for to sum all partial results
    auto sumLoop = AC->create<scf::ForOp>(loc, zeroIndex, loopCount, oneIndex,
                                          ValueRange{initial});
    Block &loopBody = sumLoop.getRegion().front();
    AC->setInsertionPointToStart(&loopBody);
    Value iv = sumLoop.getInductionVar();
    Value running = loopBody.getArgument(1);

    /// Load and combine partial result
    ReductionAccess access =
        computeReductionAccess(info, depArg, iv, zeroIndex, oneIndex);
    Value partialVal =
        AC->create<memref::LoadOp>(loc, access.buffer, access.indices);
    Value combined = combineValues(info.scalarType, running, partialVal);
    AC->create<scf::YieldOp>(loc, ValueRange{combined});

    /// Store final result to original accumulator
    AC->setInsertionPointAfter(sumLoop);
    Value finalVal = sumLoop.getResult(0);
    ReductionAccess origAccess =
        computeReductionAccess(info, origArg, zeroIndex, zeroIndex, oneIndex);
    AC->create<memref::StoreOp>(loc, finalVal, origAccess.buffer,
                                origAccess.indices);
    AC->setInsertionPointToEnd(&aggBlock);
  }

  /// Release all acquired datablocks
  AC->setInsertionPointToEnd(&aggBlock);
  for (Value depArg : depArgs)
    AC->create<DbReleaseOp>(loc, depArg);
  for (Value origArg : origArgs)
    AC->create<DbReleaseOp>(loc, origArg);
  if (aggBlock.empty() || !isa<YieldOp>(aggBlock.back()))
    AC->create<YieldOp>(loc);

  return success();
}

//===----------------------------------------------------------------------===//
// Parallel EDT Lowering Pass Implementation
//===----------------------------------------------------------------------===//
struct ParallelEdtLoweringPass
    : public arts::ParallelEdtLoweringBase<ParallelEdtLoweringPass> {
  explicit ParallelEdtLoweringPass() {}
  ~ParallelEdtLoweringPass() override { delete AC; }
  void runOnOperation() override;

private:
  ModuleOp module;
  ArtsCodegen *AC = nullptr;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void ParallelEdtLoweringPass::runOnOperation() {
  module = getOperation();
  AC = new ArtsCodegen(module, false);

  ARTS_INFO_HEADER(ParallelEdtLoweringPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// Lower all parallel EDTs
  SmallVector<EdtOp, 8> parallelEdts;
  module.walk<WalkOrder::PostOrder>([&](EdtOp edt) {
    if (edt.getType() == EdtType::parallel)
      parallelEdts.push_back(edt);
  });
  ARTS_INFO("Found " << parallelEdts.size() << " parallel EDTs to lower");
  for (EdtOp edt : parallelEdts) {
    ParallelEdtLowering lowerer(AC, edt.getLoc());
    if (failed(lowerer.lowerParallelEdt(edt))) {
      edt.emitError("Failed to lower parallel EDT");
      return signalPassFailure();
    }
  }

  ARTS_INFO_FOOTER(ParallelEdtLoweringPass);
  ARTS_DEBUG_REGION(module.dump(););
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace arts {

std::unique_ptr<Pass> createParallelEdtLoweringPass() {
  return std::make_unique<ParallelEdtLoweringPass>();
}

} // namespace arts
} // namespace mlir
