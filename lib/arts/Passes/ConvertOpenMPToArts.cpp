///==========================================================================
/// File: ConvertOpenMPToArts.cpp
//
// This file implements a module pass that converts OpenMP ops
// (omp.parallel, omp.master, omp.task, etc.) into Arts ops
///==========================================================================

/// Dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
/// Arts
#include "ArtsPassDetails.h"
#include "arts/ArtsDialect.h"
#include "arts/Passes/ArtsPasses.h"
#include "arts/Utils/ArtsUtils.h"
/// Others
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
/// Debug
#include "arts/Utils/ArtsDebug.h"
#include <optional>
ARTS_DEBUG_SETUP(convert_openmp_to_arts);

using namespace mlir;
using namespace arts;
//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//
/// Pattern to replace `omp.parallel` with `arts.edt` with `parallel` attribute
struct OMPParallelToArtsPattern : public OpRewritePattern<omp::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ARTS_INFO("Converting omp.parallel to arts.parallel");

    /// Create a new `arts.edt` operation with parallel type
    auto parOp = rewriter.create<EdtOp>(loc, EdtType::parallel,
                                        EdtConcurrency::internode);
    parOp.setNoVerifyAttr(NoVerifyAttr::get(rewriter.getContext()));
    Block &blk = parOp.getBody().front();

    /// Move the region's operations.
    Block &old = op.getRegion().front();
    blk.getOperations().splice(blk.end(), old.getOperations());

    /// Remove the original operation.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to replace `scf.parallel` with an `arts.edt` with `parallel`
/// attribute
struct SCFParallelToArtsPattern : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ARTS_INFO("Converting scf.parallel to arts.edt<parallel> + arts.for");
    rewriter.setInsertionPoint(op);

    /// Create `arts.edt` with `parallel` type and internode concurrency
    auto parEdt = rewriter.create<EdtOp>(loc, EdtType::parallel,
                                         EdtConcurrency::internode);
    parEdt.setNoVerifyAttr(NoVerifyAttr::get(rewriter.getContext()));
    Block &parBlk = parEdt.getBody().front();

    /// Insert `arts.for` inside the parallel EDT with same bounds/step
    rewriter.setInsertionPointToStart(&parBlk);
    Value lb = op.getLowerBound().front();
    Value ub = op.getUpperBound().front();
    Value st = op.getStep().front();

    auto artsFor = rewriter.create<arts::ForOp>(
        loc, ValueRange{lb}, ValueRange{ub}, ValueRange{st},
        /*schedule*/ nullptr,
        /*reductionAccumulators*/ ValueRange{});

    Region &dstRegion = artsFor.getRegion();
    if (dstRegion.empty())
      dstRegion.push_back(new Block());
    Block &dst = dstRegion.front();
    if (dst.getNumArguments() == 0)
      dst.addArgument(rewriter.getIndexType(), loc);

    /// Map IV and clone original body into arts.for body
    OpBuilder::InsertionGuard IG(rewriter);
    rewriter.setInsertionPointToStart(&dst);
    IRMapping mapper;
    Block &src = op.getRegion().front();
    if (!op.getInductionVars().empty())
      mapper.map(op.getInductionVars().front(), dst.getArgument(0));
    if (!src.getArguments().empty())
      mapper.map(src.getArgument(0), dst.getArgument(0));
    for (Operation &srcOp : src.without_terminator())
      rewriter.clone(srcOp, mapper);
    rewriter.create<arts::YieldOp>(loc);

    /// Terminate parallel EDT body
    rewriter.setInsertionPointToEnd(&parBlk);
    rewriter.create<arts::YieldOp>(loc);

    /// Remove original scf.parallel
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to replace `omp.master` with `arts.edt` with `single` attribute
struct MasterToARTSPattern : public OpRewritePattern<omp::MasterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::MasterOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    /// Create a new `arts.single` operation with intranode concurrency.
    auto artsSingle =
        rewriter.create<EdtOp>(loc, EdtType::single, EdtConcurrency::intranode);
    artsSingle.setNoVerifyAttr(NoVerifyAttr::get(rewriter.getContext()));

    /// Move the region's operations.
    Block &old = op.getRegion().front();
    Block &blk = artsSingle.getBody().front();
    blk.getOperations().splice(blk.end(), old.getOperations());

    /// Remove the original operation.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to replace `omp.task` with `arts.edt`
struct TaskToARTSPattern : public OpRewritePattern<omp::TaskOp> {
  using OpRewritePattern::OpRewritePattern;

  ArtsMode getDbMode(omp::ClauseTaskDepend taskClause) const {
    switch (taskClause) {
    case omp::ClauseTaskDepend::taskdependin:
      return ArtsMode::in;
    case omp::ClauseTaskDepend::taskdependout:
      return ArtsMode::out;
    case omp::ClauseTaskDepend::taskdependinout:
      return ArtsMode::inout;
    }
    llvm_unreachable("Unknown ClauseTaskDepend value");
  }

  LogicalResult collectTaskDependencies(SmallVector<Value> &deps,
                                        omp::TaskOp task,
                                        PatternRewriter &rewriter,
                                        Location loc) const {

    /// Collect the task deps clause
    auto dependList = task.getDependsAttr();
    if (!dependList) {
      ARTS_DEBUG(" - No dependencies found for task");
      return success();
    }
    ARTS_DEBUG(" - Processing " << dependList.size() << " dependencies");
    for (unsigned i = 0, e = dependList.size(); i < e; ++i) {
      /// Get dependency clause and type.
      auto depClause = dyn_cast<omp::ClauseTaskDependAttr>(dependList[i]);
      if (!depClause) {
        ARTS_ERROR("Missing ClauseTaskDependAttr for dependency " << i);
        return failure();
      }
      Value depVar = task.getDependVars()[i];

      /// Two supported forms for dependency variables:
      ///  (A) Scalar token held in a local memref (memref.alloca or
      ///  memref.alloc)
      ///      where a value is stored before the task; we extract the stored
      ///      value (expected produced by memref.load) as the dependency
      ///      source.
      ///  (B) A memref loaded from a table of memrefs (i.e., depVar defined by
      ///      memref.load and has memref type). In this case, use the loaded
      ///      memref directly as the dependency source.

      Operation *allocLikeOp = nullptr;
      if (auto a = depVar.getDefiningOp<memref::AllocaOp>())
        allocLikeOp = a.getOperation();
      else if (auto a2 = depVar.getDefiningOp<memref::AllocOp>())
        allocLikeOp = a2.getOperation();
      if (allocLikeOp) {
        /// Find the first user (except the task itself) that is a memref.store
        memref::StoreOp depStoreOp = nullptr;
        for (Operation *user : depVar.getUsers()) {
          if (user == task)
            continue;
          if ((depStoreOp = dyn_cast<memref::StoreOp>(user)))
            break;
        }
        if (!depStoreOp) {
          ARTS_ERROR("Expected a memref.store operation for dependency var");
          return failure();
        }

        /// Get the value that was stored; expect a memref.load
        Operation *valueDefOp = depStoreOp.getValueToStore().getDefiningOp();
        Value depLoadVal;
        if (auto loadOp = dyn_cast<memref::LoadOp>(valueDefOp))
          depLoadVal = loadOp.getResult();
        else {
          ARTS_ERROR("Expected a memref.load operation feeding dep store");
          return failure();
        }

        /// Clone the load operation at the beginning of the edt region and
        /// replace uses in-region so the region no longer depends on the outer
        /// SSA value.
        auto &region = task.getRegion();
        {
          OpBuilder::InsertionGuard IG(rewriter);
          rewriter.setInsertionPointToStart(&region.front());
          auto loadOp = cast<memref::LoadOp>(depLoadVal.getDefiningOp());
          auto newLoad = rewriter.create<memref::LoadOp>(
              loc, loadOp.getMemref(), loadOp.getIndices());
          replaceInRegion(region, loadOp.getResult(), newLoad.getResult());
        }

        /// Create the control dependency - will be removed in CreateDbs pass
        {
          OpBuilder::InsertionGuard IG(rewriter);
          rewriter.setInsertionPointAfter(depLoadVal.getDefiningOp());
          SmallVector<Value> emptyIndices;
          Value dbControl = rewriter.create<DbControlOp>(
              depLoadVal.getLoc(), getDbMode(depClause.getValue()), depLoadVal,
              emptyIndices);
          deps.push_back(dbControl);
        }

        /// Replace the dependency allocation with an undefined value to allow
        /// DCE of the token container.
        replaceWithUndef(allocLikeOp, rewriter);
        continue;
      }

      if (auto depMemrefLoad = depVar.getDefiningOp<memref::LoadOp>()) {
        /// This is the case where the dependency is a memref loaded from a
        /// table (e.g., memref<?xmemref<?xf64>>). Use the loaded memref as the
        /// dependency source. Clone the load inside the task region so the
        /// region has a local definition if it uses this value.
        auto &region = task.getRegion();
        {
          OpBuilder::InsertionGuard IG(rewriter);
          rewriter.setInsertionPointToStart(&region.front());
          auto newLoad = rewriter.create<memref::LoadOp>(
              loc, depMemrefLoad.getMemref(), depMemrefLoad.getIndices());
          replaceInRegion(region, depVar, newLoad.getResult());
        }

        {
          OpBuilder::InsertionGuard IG(rewriter);
          rewriter.setInsertionPointAfter(depMemrefLoad);
          SmallVector<Value> emptyIndices;
          Value dbControl = rewriter.create<DbControlOp>(
              depMemrefLoad.getLoc(), getDbMode(depClause.getValue()), depVar,
              emptyIndices);
          deps.push_back(dbControl);
        }
        continue;
      }

      ARTS_ERROR("Unsupported dependency variable producer. Expected "
                 "memref.alloca, memref.alloc, or memref.load of a memref");
      return failure();
    }
    return success();
  }

  LogicalResult matchAndRewrite(omp::TaskOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ARTS_INFO("Converting omp.task to arts.edt");
    /// Collect dependencies
    SmallVector<Value> deps;
    if (failed(collectTaskDependencies(deps, op, rewriter, loc))) {
      ARTS_ERROR("Failed to collect task dependencies");
      return failure();
    }

    /// Create a new `arts.edt` operation with intranode concurrency.
    auto edtOp = rewriter.create<EdtOp>(loc, EdtType::task,
                                        EdtConcurrency::intranode, deps);
    edtOp.setNoVerifyAttr(NoVerifyAttr::get(rewriter.getContext()));
    Block &blk = edtOp.getBody().front();

    /// Move the region's operations.
    Block &old = op.getRegion().front();
    blk.getOperations().splice(blk.end(), old.getOperations());

    /// Remove the original `omp.task`.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to replace `omp.wsloop` with `arts.for` loop
struct WsloopToARTSPattern : public OpRewritePattern<omp::WsLoopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::WsLoopOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ARTS_INFO("Converting omp.wsloop to arts.for");

    /// Get the loop bounds and step from the wsloop
    auto lowerBound = op.getLowerBound()[0];
    auto upperBound = op.getUpperBound()[0];
    auto step = op.getStep()[0];

    /// Map OpenMP schedule to Arts ForScheduleKindAttr if present
    arts::ForScheduleKindAttr schedAttr = nullptr;
    if (auto sched = op.getScheduleVal()) {
      switch (*sched) {
      case omp::ClauseScheduleKind::Static:
        schedAttr = arts::ForScheduleKindAttr::get(
            rewriter.getContext(), arts::ForScheduleKind::Static);
        break;
      case omp::ClauseScheduleKind::Dynamic:
        schedAttr = arts::ForScheduleKindAttr::get(
            rewriter.getContext(), arts::ForScheduleKind::Dynamic);
        break;
      case omp::ClauseScheduleKind::Guided:
        schedAttr = arts::ForScheduleKindAttr::get(
            rewriter.getContext(), arts::ForScheduleKind::Guided);
        break;
      case omp::ClauseScheduleKind::Auto:
        schedAttr = arts::ForScheduleKindAttr::get(rewriter.getContext(),
                                                   arts::ForScheduleKind::Auto);
        break;
      case omp::ClauseScheduleKind::Runtime:
        schedAttr = arts::ForScheduleKindAttr::get(
            rewriter.getContext(), arts::ForScheduleKind::Runtime);
        break;
      }
    }

    /// Capture OpenMP chunk size if it is a known constant so later passes can
    /// implement the desired scheduling policy. The loop step should remain the
    /// original iteration step irrespective of the chunk size.
    std::optional<int64_t> staticChunkSize;
    if (auto chunk = op.getScheduleChunkVar()) {
      int64_t chunkVal;
      if (getConstantIndex(chunk, chunkVal) && chunkVal > 0)
        staticChunkSize = chunkVal;
    }

    /// Collect reduction metadata if present on omp.wsloop
    SmallVector<Value> redAccs;
    DenseMap<Value, omp::ReductionDeclareOp> reductionDecls;
    collectReductionMetadata(op, rewriter, redAccs, reductionDecls);

    /// Create `arts.for` and move `omp.wsloop` body
    auto forOp = rewriter.create<arts::ForOp>(
        loc, ValueRange{lowerBound}, ValueRange{upperBound}, ValueRange{step},
        schedAttr, ValueRange{redAccs});

    /// Set chunk size attribute if present
    if (staticChunkSize)
      forOp->setAttr("chunk_size",
                     rewriter.getI64IntegerAttr(*staticChunkSize));

    /// Add region and argument to the forOp
    Region &dstRegion = forOp.getRegion();
    if (dstRegion.empty())
      dstRegion.push_back(new Block());
    Block &dst = dstRegion.front();
    if (dst.getNumArguments() == 0)
      dst.addArgument(rewriter.getIndexType(), loc);

    /// Move ops and inline reductions
    OpBuilder::InsertionGuard IG(rewriter);
    rewriter.setInsertionPointToStart(&dst);
    IRMapping mapper;
    Block &src = op.getRegion().front();
    if (!src.getArguments().empty())
      mapper.map(src.getArgument(0), dst.getArgument(0));
    moveOps(src, dst, rewriter, mapper, reductionDecls);
    rewriter.create<arts::YieldOp>(loc);

    /// Remove the original wsloop
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Collect reduction metadata from the OpenMP wsloop operation
  void collectReductionMetadata(
      omp::WsLoopOp op, PatternRewriter &rewriter, SmallVector<Value> &redAccs,
      DenseMap<Value, omp::ReductionDeclareOp> &reductionDecls) const {
    auto reds = op.getReductionsAttr();
    if (!reds)
      return;

    /// Get the reduction variables
    auto reductionVars = op.getReductionVars();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    assert(module && "Module is required");

    for (auto [attr, value] : llvm::zip(reds.getValue(), reductionVars)) {
      redAccs.push_back(arts::getUnderlyingValue(value));
      if (auto symRef = dyn_cast<SymbolRefAttr>(attr)) {
        auto decl = dyn_cast_or_null<omp::ReductionDeclareOp>(
            module.lookupSymbol(symRef.getLeafReference()));
        assert(decl && "Failed to resolve reduction declaration");
        reductionDecls.try_emplace(value, decl);
      }
    }
  }

  /// Inline reduction operations and move non-reduction operations
  void moveOps(Block &src, Block &dst, PatternRewriter &rewriter,
               IRMapping &mapper,
               DenseMap<Value, omp::ReductionDeclareOp> &reductionDecls) const {
    for (Operation &srcOp : src.without_terminator()) {
      if (auto redOp = dyn_cast<omp::ReductionOp>(&srcOp)) {
        if (inlineReduction(redOp, rewriter, mapper, reductionDecls))
          continue;
      }
      rewriter.clone(srcOp, mapper);
    }

    /// Analyzes the reductionDecls and removes the ones that are not used
    for (auto [value, decl] : reductionDecls) {
      if (decl.use_empty())
        rewriter.eraseOp(decl);
    }
  }

  /// Inline a reduction operation
  bool inlineReduction(
      omp::ReductionOp redOp, PatternRewriter &rewriter, IRMapping &mapper,
      const DenseMap<Value, omp::ReductionDeclareOp> &reductionDecls) const {
    auto it = reductionDecls.find(redOp.getAccumulator());
    if (it == reductionDecls.end())
      return false;

    Value operand = mapper.lookupOrDefault(redOp.getOperand());
    Value accumulator = mapper.lookupOrDefault(redOp.getAccumulator());
    auto memType = accumulator.getType().dyn_cast<MemRefType>();
    if (!memType)
      return false;

    Location loc = redOp.getLoc();
    unsigned rank = memType.getRank();
    SmallVector<Value> zeroIndices;
    zeroIndices.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      zeroIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));

    Value current =
        rewriter.create<memref::LoadOp>(loc, accumulator, zeroIndices);

    Value combined =
        cloneCombinerRegion(it->second, current, operand, rewriter, loc);
    if (!combined)
      return false;

    rewriter.create<memref::StoreOp>(loc, combined, accumulator, zeroIndices);
    return true;
  }

  /// Clone combiner region for reduction operation
  Value cloneCombinerRegion(omp::ReductionDeclareOp decl, Value lhs, Value rhs,
                            PatternRewriter &rewriter, Location loc) const {
    Block &combinerBlock = decl.getReductionRegion().front();
    IRMapping combinerMap;
    combinerMap.map(combinerBlock.getArgument(0), lhs);
    combinerMap.map(combinerBlock.getArgument(1), rhs);

    for (Operation &combOp : combinerBlock.without_terminator())
      rewriter.clone(combOp, combinerMap);

    auto yieldOp = dyn_cast<omp::YieldOp>(combinerBlock.getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
      return Value();
    return combinerMap.lookup(yieldOp.getOperand(0));
  }
};

/// Pattern to convert a subset of omp.atomic.update to arts.atomic_add
struct AtomicUpdateToArtsPattern
    : public OpRewritePattern<omp::AtomicUpdateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::AtomicUpdateOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Converting omp.atomic.update to arts.atomic_add");
    /// Only handle canonical form: one block, single arg, computes add of arg
    /// and a captured value
    auto &region = op.getRegion();
    if (!region.hasOneBlock())
      return failure();
    Block &blk = region.front();
    if (blk.getNumArguments() != 1)
      return failure();

    /// Heuristically detect add: last non-terminator must yield result of
    /// arith.add*
    Operation *yield = blk.getTerminator();
    if (!yield)
      return failure();
    Value yielded;
    if (auto y = dyn_cast<omp::YieldOp>(yield)) {
      if (y->getNumOperands() != 1)
        return failure();
      yielded = y->getOperand(0);
    } else {
      return failure();
    }

    Operation *def = yielded.getDefiningOp();
    if (!def)
      return failure();
    bool isAdd = isa<arith::AddIOp>(def) || isa<arith::AddFOp>(def);
    if (!isAdd)
      return failure();

    /// addr must be a memref (rank-0) for now
    Value addr = op.getX();

    /// Extract the non-block-arg operand as the increment value
    Value blockArg = blk.getArgument(0);
    Value inc;
    if (def->getOperand(0) == blockArg)
      inc = def->getOperand(1);
    else if (def->getOperand(1) == blockArg)
      inc = def->getOperand(0);
    else
      return failure();

    rewriter.replaceOpWithNewOp<arts::AtomicAddOp>(op, addr, inc);
    return success();
  }
};

/// Pattern to replace `omp.terminator` with `arts.yield`
struct TerminatorToARTSPattern : public OpRewritePattern<omp::TerminatorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::TerminatorOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<arts::YieldOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to replace `omp.barrier` with `arts.barrier`
struct BarrierToARTSPattern : public OpRewritePattern<omp::BarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::BarrierOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Converting omp.barrier to arts.barrier");
    auto loc = op.getLoc();
    rewriter.create<arts::BarrierOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to replace `omp.taskwait` with `arts.barrier`
struct TaskwaitToARTSPattern : public OpRewritePattern<omp::TaskwaitOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::TaskwaitOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Converting omp.taskwait to arts.barrier");
    auto loc = op.getLoc();
    rewriter.create<arts::BarrierOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to replace `omp.taskloop` with `arts.for` and EDT creation
struct TaskloopToARTSPattern : public OpRewritePattern<omp::TaskLoopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::TaskLoopOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ARTS_INFO("Converting omp.taskloop to arts.for with EDTs");

    /// Get the loop bounds and step from the taskloop
    auto lowerBound = op.getLowerBound()[0];
    auto upperBound = op.getUpperBound()[0];
    auto step = op.getStep()[0];

    /// Create arts.for and move taskloop body
    auto forOp = rewriter.create<arts::ForOp>(
        loc, ValueRange{lowerBound}, ValueRange{upperBound}, ValueRange{step},
        /*schedule*/ nullptr,
        /*reductionAccumulators*/ ValueRange{});
    Region &dstRegion = forOp.getRegion();
    if (dstRegion.empty())
      dstRegion.push_back(new Block());
    Block &dst = dstRegion.front();
    if (dst.getNumArguments() == 0)
      dst.addArgument(rewriter.getIndexType(), loc);

    OpBuilder::InsertionGuard IG(rewriter);
    rewriter.setInsertionPointToStart(&dst);
    IRMapping mapper;
    Block &src = op.getRegion().front();
    if (!src.getArguments().empty())
      mapper.map(src.getArgument(0), dst.getArgument(0));
    for (Operation &srcOp : src.without_terminator())
      rewriter.clone(srcOp, mapper);
    rewriter.create<arts::YieldOp>(loc);

    /// Remove the original taskloop
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to replace 'func.call' with an equivalent 'arts' call if exists.
struct CallToARTSPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (callee == "omp_get_thread_num") {
      rewriter.replaceOpWithNewOp<arts::GetCurrentNodeOp>(callOp);
      // rewriter.replaceOpWithNewOp<arts::GetCurrentWorkerOp>(callOp);
      return success();
    }
    if (callee == "omp_get_num_threads" || callee == "omp_get_max_threads") {
      rewriter.replaceOpWithNewOp<arts::GetTotalNodesOp>(callOp);
      // rewriter.replaceOpWithNewOp<arts::GetTotalWorkersOp>(callOp);
      return success();
    }
    /// nothing to do, leave the op as-is
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//
namespace {
struct ConvertOpenMPToArtsPass
    : public arts::ConvertOpenMPToArtsBase<ConvertOpenMPToArtsPass> {
  void runOnOperation() override;
};
} // end namespace

/// Pass to convert OpenMP operations to Arts operations
void ConvertOpenMPToArtsPass::runOnOperation() {
  ModuleOp module = getOperation();
  ARTS_INFO_HEADER(ConvertOpenMPToArtsPass);
  ARTS_DEBUG_REGION(module.dump(););
  MLIRContext *context = &getContext();

  /// Add patterns to convert OpenMP operations to Arts operations
  RewritePatternSet patterns(context);
  patterns.add<OMPParallelToArtsPattern, SCFParallelToArtsPattern,
               MasterToARTSPattern, TaskToARTSPattern, TaskloopToARTSPattern,
               WsloopToARTSPattern, AtomicUpdateToArtsPattern,
               TerminatorToARTSPattern, BarrierToARTSPattern,
               TaskwaitToARTSPattern, CallToARTSPattern>(context);
  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(module, std::move(patterns), config);

  removeUndefOps(module);
  ARTS_INFO_FOOTER(ConvertOpenMPToArtsPass);
  ARTS_DEBUG_REGION(module.dump(););
}

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//
namespace mlir {
namespace arts {
std::unique_ptr<Pass> createConvertOpenMPtoArtsPass() {
  return std::make_unique<ConvertOpenMPToArtsPass>();
}
} // namespace arts
} // namespace mlir
