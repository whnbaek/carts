///==========================================================================
/// File: ArtsOps.td
/// This file defines multiple operations for the ARTS Dialect.
///==========================================================================

#ifndef ARTS_OPS
#define ARTS_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/LLVMIR/LLVMInterfaces.td"

/// ARTS Dialect
include "arts/ArtsDialect.td"
include "arts/ArtsAttributes.td"
include "arts/ArtsTypes.td"

//===----------------------------------------------------------------------===//
// Custom Traits for ARTS
//===----------------------------------------------------------------------===//

class Arts_InferShapedTypeOpAdaptor<string mnemonic, list<Trait> traits = []>
    : Arts_Op<mnemonic, !listconcat(traits, [InferShapedTypeOpAdaptor, Pure])> {
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// arts.undef
//===----------------------------------------------------------------------===//
def UndefOp: Arts_Op<"undef", [Pure]> {
  let summary = "More flexible undef op";
  let results = (outs AnyType:$result);
  let hasCanonicalizer = true;
  let assemblyFormat = "attr-dict `:` type($result)";
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$type), [{
      $_state.addTypes(type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// arts.alloc
//===----------------------------------------------------------------------===//
def AllocOp : Arts_Op<"alloc", []> {
  let summary = "Allocate a memref in ARTS dialect";
  let description = [{
    The `arts.alloc` operation allocates a memref with a given shape.
    It mirrors `memref.alloc` but is customized for the ARTS dialect to enable
    specific behavior during code generation and lowering.
  }];

  let arguments = (ins Variadic<Index>:$dynamicSizes);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    `(` $dynamicSizes `)` attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Arts DB Operations
//===----------------------------------------------------------------------===//
def DbDimOp : Arts_Op<"db_dim", [Pure]> {
  let summary = "Query dimension of a datablock-backed memref";
  let description = [{
    The `arts.db_dim` op queries the size of the specified dimension of a
    datablock-backed memref. It is preferred over `memref.dim` so DB-oriented
    canonicalization can fold it to explicit values coming from `arts.db_alloc`
    or `arts.db_acquire` slice sizes.
  }];
  let arguments = (ins AnyMemRef:$source, Index:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = "$source `,` $dim attr-dict `:` type($source) `->` type($result)";
  let hasCanonicalizer = 1;
  let builders = [
    OpBuilder<(ins "Value":$source, "int64_t":$dim)>
  ];
}

def DbAllocOp : Arts_Op<"db_alloc", [
    AttrSizedOperandSegments, MemoryEffects<[MemAlloc]>
    ]> {
  let summary = "Allocate an ARTS DataBlock and get its pointer and GUID.";
  let description = [{
    This operation corresponds to the `artsDbCreate` runtime function.
    It allocates a DataBlock and returns its pointer and GUID.
    - mode: enum for "in", "out", "inout"
    - ptr: address that will be copied to the DataBlock
    - sizes: outer sizes (number of datablocks along each dimension)
    - elementSizes: logical dimensions of the datablock element payload
    - allocType: "stack", "heap", "global", "unknown" - tracks original allocation type
  }];
  let arguments = (ins
    ArtsModeAttr:$mode,
    I32:$route,
    DbAllocTypeAttr:$allocType,
    DbModeAttr:$dbMode,
    TypeAttr:$elementType,
    Optional<AnyMemRef>:$address,
    Variadic<Index>:$sizes,
    Variadic<Index>:$elementSizes
  );
  let results = (outs MemRefOf<[I64]>:$guid, AnyMemRef:$ptr);
  let assemblyFormat = [{
    `[` $mode `,` $allocType `,` $dbMode `]`
    `route` `(` $route `:` type($route) `)`
    ( `(` $address^ `:` type($address) `)` )?
    `sizes` `[` $sizes `]`
    `elementType` `(` $elementType `)`
    `elementSizes` `[` $elementSizes `]`
    attr-dict `:` `(` type($guid) `,` type($ptr) `)`
  }];
  let builders = [
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$route,
                   "DbAllocType":$allocType,
                   "DbMode":$dbMode,
                   "Type":$elementType,
                   CArg<"Value", "nullptr">:$address,
                   CArg<"SmallVector<Value>", "{}">:$sizes,
                   CArg<"SmallVector<Value>", "{}">:$elementSizes)>,
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$route,
                   "DbAllocType":$allocType,
                   "DbMode":$dbMode,
                   "Type":$elementType,
                   CArg<"SmallVector<Value>", "{}">:$sizes,
                   CArg<"SmallVector<Value>", "{}">:$elementSizes)>,
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$route,
                   "DbAllocType":$allocType,
                   "DbMode":$dbMode,
                   "Type":$elementType,
                   "Type":$pointerType,
                   CArg<"SmallVector<Value>", "{}">:$sizes,
                   CArg<"SmallVector<Value>", "{}">:$elementSizes)>
  ];
  let extraClassDeclaration = [{
    MemRefType getAllocatedElementType();
  }];
}

def DbAcquireOp : Arts_Op<"db_acquire",
  [AttrSizedOperandSegments]
  > {
  let summary = "Acquires access to an ARTS DataBlock.";
  let description = [{
    This operation acquires a local view of the DataBlock for safe access within EDTs.
    It subsumes dependency creation with mode and subregion support.
    - mode: enum for "in", "out", "inout"
    - source: DbAllocOp or previous DbAcquireOp
    - indices: indices for subregion
    - offsets: offsets for subregion
    - sizes: sizes for subregion
  }];
  let arguments = (ins
    ArtsModeAttr:$mode,
    AnyMemRef:$source_guid,
    AnyMemRef:$source_ptr,
    Variadic<Index>:$indices,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes
  );
  let results = (outs MemRefOf<[I64]>:$guid, AnyMemRef:$ptr);
  let assemblyFormat = [{
    `[` $mode `]`
    `(` $source_guid `,` $source_ptr `)`
    ( `indices` `[` $indices^ `]` )?
    ( `offsets` `[` $offsets^ `]` )?
    ( `sizes` `[` $sizes^ `]` )?
    attr-dict `:` `(` type($source_guid) `,` type($source_ptr) `)` `->` `(` type($guid) `,` type($ptr) `)`
  }];
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$source_guid,
                   "Value":$source_ptr,
                   CArg<"SmallVector<Value>", "{}">:$indices,
                   CArg<"SmallVector<Value>", "{}">:$offsets,
                   CArg<"SmallVector<Value>", "{}">:$sizes)>
  ];
}

def DbRefOp : Arts_Op<"db_ref", [Pure]> {
  let summary = "References an outer datablock element.";
  let description = [{
    References an element from a datablock handle using only the outer
    dimensions. The result type must match the element type of the source
    memref. 
  }];
  let arguments = (ins
    AnyMemRef:$source,
    Variadic<Index>:$indices
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $source `[` $indices `]` attr-dict `:` type($source) `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value":$source,
                   CArg<"ArrayRef<Value>", "{}">:$indices)>
  ];
  let hasVerifier = 1;
}

def DbReleaseOp : Arts_Op<"db_release", []> {
  let summary = "Releases access to an ARTS DataBlock source.";
  let description = [{
    This operation releases the acquired source, signaling completion to the runtime.
    - release_type: specifies whether to perform memory freeing for dynamic allocations
  }];
  let arguments = (ins
    AnyMemRef:$source,
    OptionalAttr<DbReleaseTypeAttr>:$release_type
  );
  let assemblyFormat = [{
    `(` $source `)` `:` type($source) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$source)>,
    OpBuilder<(ins "Value":$source, "DbReleaseType":$release_type)>
  ];
}

def DbFreeOp : Arts_Op<"db_free", []> {
  let summary = "Frees a DataBlock allocated by arts.db_alloc.";
  let description = [{
    This operation frees a DataBlock that was previously allocated with arts.db_alloc.
    It is used for memory management of dynamically allocated datablocks.
  }];
  let arguments = (ins
    AnyMemRef:$source
  );
  let assemblyFormat = [{
    `(` $source `)` `:` type($source) attr-dict
  }];
}

def DbControlOp : Arts_Op<"db_control", 
  [AttrSizedOperandSegments]> {
  let summary = "Creates a control dependency for arts.edt";
  let description = [{
    This operation creates a control dependency for the `arts.edt` operation.
    It is used to ensure that the DataBlock is ready before the EDT is executed.
    - mode: "in", "out", "inout"
    - ptr: base address of the DataBlock
    - elementType: type of the DataBlock
    - elementTypeSize: size of the element type
    - indices: indices for the subview
    - offsets: offsets for the subview
    - sizes: sizes for the subview
    Note: Currently ignored for execution, but retained for future correctness enforcement.
  }];
  let arguments = (ins
    ArtsModeAttr:$mode,
    AnyMemRef:$ptr,
    TypeAttr:$elementType,
    Index:$elementTypeSize,
    Variadic<Index>:$indices,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes
  );

  let results = (outs AnyMemRef:$subview);
  let assemblyFormat = [{
    `[` $mode `]` `(` $ptr `:` type($ptr) `)` 
    `elementType` `(` $elementType `)` 
    `elementTypeSize` `(` $elementTypeSize `:` type($elementTypeSize) `)` 
    `indices` `[` $indices `]` 
    `offsets` `[` $offsets `]` 
    `sizes` `[` $sizes `]` 
    attr-dict `:` type($subview)
  }];
  
  let builders = [
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$ptr,
                   CArg<"SmallVector<Value>", "{}">:$pinnedIndices)>
  ];
}

def DbNumElementsOp : Arts_Op<"db_num_elements", [Pure]> {
  let summary = "Compute the number of elements in a datablock.";
  let description = [{
    Computes the total number of elements in a datablock. This is used
    to calculate dependency counts for EDT creation.
  }];
  let arguments = (ins Variadic<Index>:$sizes);
  let results = (outs Index:$count);
  let assemblyFormat = "`sizes` `(` $sizes `)` attr-dict `:` `->` type($count)";

  let builders = [
    OpBuilder<(ins "DbAllocOp":$allocOp)>,
    OpBuilder<(ins "DbAcquireOp":$acquireOp)>
  ];
}

//===----------------------------------------------------------------------===//
// arts.edt
//===----------------------------------------------------------------------===//
def EdtOp : Arts_Op<"edt",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   SingleBlockImplicitTerminator<"arts::YieldOp">,
   RecursiveMemoryEffects]> {
  let summary = "Event-Driven Task in ARTS";
  let description = [{
    Defines an ARTS event-driven task (EDT) with:
      - `dependencies` that specify input/output memory dependencies (acquired views).

    The single-block region contains the task body.
    - if 'parallel' attribute is set, it can spawn multiple tasks.
    - if 'single' attribute is set, it is guaranteed to run in a single
      task, even if it is called from a parallel region.
    - 'concurrency' specifies the execution scope: intranode or internode.
    - 'workers' specifies the number of parallel workers to use (optional).
  }];
  let arguments = (ins
    EdtTypeAttr:$type,
    EdtConcurrencyAttr:$concurrency,
    I32:$route,
    OptionalAttr<WorkersAttr>:$workers,
    OptionalAttr<NoVerifyAttr>:$no_verify,
    Variadic<AnyMemRef>:$dependencies
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $type $concurrency `route` `(` $route `)` (`(` $dependencies^ `)` `:` type($dependencies))? attr-dict-with-keyword
    $body
  }];
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "EdtType":$type, "EdtConcurrency":$concurrency)>,
    OpBuilder<(ins "EdtType":$type, "EdtConcurrency":$concurrency, "ValueRange":$dependencies)>,
    OpBuilder<(ins "EdtType":$type, "EdtConcurrency":$concurrency, "Value":$route, "ValueRange":$dependencies)>
  ];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    SmallVector<Value> getDependenciesAsVector();
  }];
}

//===----------------------------------------------------------------------===//
// arts.epoch
//===----------------------------------------------------------------------===//
def EpochOp : Arts_Op<"epoch", 
  [SingleBlockImplicitTerminator<"arts::YieldOp">]> {

  let summary = "Epoch region in ARTS";
  let description = [{
    An ARTS epoch region. It can spawn multiple tasks and
    synchronize them at the end of the epoch. Returns the epoch GUID
    for use by contained EDTs.
  }];
  let regions = (region SizedRegion<1>:$body);
  let results = (outs I64:$epochGuid);

  let assemblyFormat = [{
    attr-dict-with-keyword $body `:` type($epochGuid)
  }];
}

//===----------------------------------------------------------------------===//
// arts.yield
//===----------------------------------------------------------------------===//
def YieldOp : Arts_Op<"yield", 
  [Pure, ReturnLike, Terminator, 
   ParentOneOf<["EdtOp", "EpochOp", "ForOp"]>]> {

  let summary = "ARTS dialect yield terminator";
  let description = [{
    The `arts.yield` operation serves as a terminator for various 
    ARTS operations, such as `arts.epoch`, `arts.parallel`, `arts.single`, 
    and `arts.edt`. It ensures proper control flow within the ARTS dialect 
    by marking the end of a region.
  }];
  let assemblyFormat = [{
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// arts.barrier
//===----------------------------------------------------------------------===//
def BarrierOp : Arts_Op<"barrier", [MemoryEffects<[MemWrite]>]> {
  let summary = "Barrier synchronization in ARTS";
  let description = [{
    Barrier synchronization operation in ARTS. It synchronizes all previous
    spawned EDTs of the parent region. This operation has memory write effects
    as it affects the synchronization state of the runtime system.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// arts.atomic_add
//===----------------------------------------------------------------------===//
def AtomicAddOp : Arts_Op<"atomic_add", [MemoryEffects<[MemWrite]>]> {
  let summary = "Atomic add to a scalar memref element";
  let description = [{
    Atomically adds a value to the element at the given address. This is the
    ARTS representation for reductions when converting from OpenMP.
    Only scalar memrefs (rank-0) are supported for now.
  }];

  let arguments = (ins AnyType:$addr, AnyType:$value);
  let assemblyFormat = "`(` $addr `,` $value `:` type($addr) `,` type($value) `)` attr-dict";
}

//===----------------------------------------------------------------------===//
// arts.reduction
//===----------------------------------------------------------------------===//
def ReductionOp : Arts_Op<"reduction", [MemoryEffects<[MemWrite]>]> {
  let summary = "Parallel reduction operation in ARTS";
  let description = [{
    Performs a parallel reduction operation across multiple workers.
    This operation encapsulates the complex reduction pattern that was previously
    handled manually in the OpenMP to ARTS conversion.
    
    The operation:
    1. Distributes work across available workers
    2. Each worker computes a partial reduction
    3. Synchronizes workers with a barrier
    4. Performs final reduction in a single worker
    
    - array: the input array to reduce over
    - result: the output location for the final reduction result (must contain initial value)
    - lower_bound: starting index for the reduction
    - upper_bound: ending index for the reduction  
    - step: step size for iteration
    - reduction_kind: the type of reduction (add, mul, min, max, etc.)
  }];

  let arguments = (ins 
    AnyMemRef:$array,
    AnyMemRef:$result,
    Index:$lower_bound,
    Index:$upper_bound,
    Index:$step,
    ReductionKindAttr:$reduction_kind
  );
  
  let results = (outs);
  let assemblyFormat = [{
    `(` $array `:` type($array) `,` $result `:` type($result) `,` 
        $lower_bound `,` $upper_bound `,` $step `,` 
        $reduction_kind `)` attr-dict
  }];
  
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$array, "Value":$result,
                   "Value":$lower_bound, "Value":$upper_bound, "Value":$step,
                   "ReductionKind":$reduction_kind)>
  ];
}

//===----------------------------------------------------------------------===//
// arts.get_total_workers
//===----------------------------------------------------------------------===//
def GetTotalWorkersOp : Arts_Op<"get_total_workers", 
  [MemoryEffects<[MemWrite]>, RecursiveMemoryEffects]> {
  let summary = "Get the total number of workers in ARTS";
  let description = [{
    The `arts.get_total_workers` operation retrieves the total number of 
    workers available in the ARTS runtime environment.
  }];
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes(IntegerType::get($_builder.getContext(), 32));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// arts.get_total_nodes
//===----------------------------------------------------------------------===//
def GetTotalNodesOp : Arts_Op<"get_total_nodes",
  [MemoryEffects<[MemWrite]>, RecursiveMemoryEffects]> {
  let summary = "Get the total number of nodes in ARTS";
  let description = [{
    The `arts.get_total_nodes` operation retrieves the total number of 
    nodes available in the ARTS runtime environment.
  }];
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes(IntegerType::get($_builder.getContext(), 32));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// arts.get_current_worker
//===----------------------------------------------------------------------===//
def GetCurrentWorkerOp : Arts_Op<"get_current_worker",
  [MemoryEffects<[MemWrite]>, RecursiveMemoryEffects]> {
  let summary = "Get the current worker ID in ARTS";
  let description = [{
    The `arts.get_current_worker` operation retrieves the ID of the 
    current worker executing the operation in the ARTS runtime environment.
  }];
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes(IntegerType::get($_builder.getContext(), 32));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// arts.get_current_node
//===----------------------------------------------------------------------===//
def GetCurrentNodeOp : Arts_Op<"get_current_node", 
  [MemoryEffects<[MemWrite]>, RecursiveMemoryEffects]> {
  let summary = "Get the current node ID in ARTS";
  let description = [{
    The `arts.get_current_node` operation retrieves the ID of the 
    current node executing the operation in the ARTS runtime environment.
  }];
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes(IntegerType::get($_builder.getContext(), 32));
    }]>
  ];
}


//===----------------------------------------------------------------------===//
// EDT Lowering Operations
//===----------------------------------------------------------------------===//
def EdtCreateOp : Arts_Op<"edt_create", [MemoryEffects<[MemWrite]>]> {
  let summary = "Outlined EDT call with parameter pack and dependency count.";
  let description = [{
    Intermediate op for calling outlined EDT function with param array and total dependency count.
    The depCount is the sum of number of elements of all dependencies.
    Optional epochGuid parameter for EDT creation with specific epoch.
    Returns EDT guid.
  }];
  let arguments = (ins AnyMemRef:$param_memref, I32:$depCount, I32:$route, Optional<I64>:$epochGuid);
  let results = (outs I64:$guid);
  
  let assemblyFormat = [{
    `(` $param_memref `:` type($param_memref) `,` $depCount `,` $route `)` 
    (`epoch` `(` $epochGuid^ `:` type($epochGuid) `)`)? attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$param_memref, "Value":$depCount)>,
    OpBuilder<(ins "Value":$param_memref, "Value":$depCount, "Value":$epochGuid)>
  ];
}

def EdtParamPackOp : Arts_Op<"edt_param_pack", [MemoryEffects<[MemWrite]>]> {
  let summary = "Pack parameters into memref<?xi64>.";
  let description = [{
    Packs multiple parameters (cast to i64) into a memref for ARTS paramv.
  }];
  let arguments = (ins Variadic<AnyType>:$params);
  let results = (outs AnyMemRef:$memref);
  let assemblyFormat = "`(` $params `)` attr-dict `:` type($params) `:` type($memref)";
}

def EdtParamUnpackOp : Arts_Op<"edt_param_unpack", [Pure]> {
  let summary = "Unpack parameters from memref<?xi64>.";
  let description = [{
    Unpacks memref to multiple original types (load i64, cast back).
  }];
  let arguments = (ins AnyMemRef:$memref);
  let results = (outs Variadic<AnyType>:$unpacked);
  let assemblyFormat = "$memref `:` type($memref) attr-dict `:` `(` type($unpacked) `)`";
}

def HandleDepOp : Arts_Op<"handle_dep", [MemoryEffects<[MemWrite]>]> {
  let summary = "Handle DB dependency.";
  let description = [{
    Handles DB dependencies for the EDT.
  }];
  let arguments = (ins I64:$edt_guid, AnyType:$db_guid, BoolAttr:$record, BoolAttr:$increment);
  let assemblyFormat = [{
  $edt_guid `(` 
    $db_guid `:` type($db_guid) `,` 
    `record` $record `,` 
    `increment` $increment 
  `)` attr-dict `:` type($edt_guid)
  }];
}

def RecordDepOp : Arts_Op<"record_dep", [MemoryEffects<[MemWrite]>]> {
  let summary = "Record in-dependencies for EDT.";
  let description = [{
    Records all in-dependencies (datablocks) for the EDT in order.
    Each datablock will be recorded as a dependency for the EDT.
  }];
  let arguments = (ins I64:$edt_guid, Variadic<AnyType>:$datablocks);
  let assemblyFormat = "$edt_guid `(` $datablocks `:` type($datablocks) `)` attr-dict";
}

def IncrementDepOp : Arts_Op<"increment_dep", [MemoryEffects<[MemWrite]>]> {
  let summary = "Increment out-latches for EDT dependencies.";
  let description = [{
    Increments latch count for all out-dependencies (datablocks) of the EDT.
    Each datablock will have its output latch incremented.
  }];
  let arguments = (ins I64:$edt_guid, Variadic<AnyType>:$datablocks);
  let assemblyFormat = "$edt_guid `(` $datablocks `:` type($datablocks) `)` attr-dict";
}

//===----------------------------------------------------------------------===//
// Dependency Structure Access Operations
//===----------------------------------------------------------------------===//

def DepGepOp : Arts_Op<"dep_gep", [Pure, AttrSizedOperandSegments]> {
  let summary = "Extract pointer field from an EDT dependency structure";
  let description = [{
    Extracts the pointer field from an EDT dependency structure.
    The input can be a pointer to a dependency structure or a memref of
    dependency structures. Optional indices specify which dependency entry
    to access when the input is an array of dependencies (e.g., when a
    DbAlloc/Acquire returned multiple datablocks). 
  }];
  let arguments = (ins AnyType:$dep_struct, Index:$offset,
                       Variadic<AnyType>:$indices, Variadic<Index>:$strides);
  let results = (outs AnyType:$ptr);
  let assemblyFormat = [{
    $dep_struct 
     `offset` `[` $offset `:` type($offset) `]`
    ( `indices` `[` $indices^ `:` type($indices) `]` )?
    ( `strides` `[` $strides^ `]` )?
    attr-dict `:` type($dep_struct) `:` type($ptr)
  }];
  
  let builders = [
    OpBuilder<(ins "Value":$dep_struct, "Value":$offset)>,
    OpBuilder<(ins "Value":$dep_struct, "Value":$offset, "SmallVector<Value>":$indices)>,
    OpBuilder<(ins "Value":$dep_struct, "Value":$offset, "SmallVector<Value>":$indices, "SmallVector<Value>":$strides)>
  ];
}


def DbGepOp : Arts_Op<"db_gep", [Pure, AttrSizedOperandSegments]> {
  let summary = "Compute element pointer (LLVM ptr) using byte strides.";
  let description = [{
    Computes a raw LLVM pointer to an element from a base pointer using indices and strides.
    Always produces an LLVM pointer result regardless of the input type.
  }];
  let arguments = (ins AnyType:$base_ptr, Variadic<Index>:$indices, Variadic<Index>:$strides);
  let results = (outs AnyType:$ptr);
  let assemblyFormat = [{
    `(` $base_ptr `:` type($base_ptr) `)`
    `indices` `[` $indices `]` `strides` `[` $strides `]`
    attr-dict `:` type($ptr)
  }];
  let builders = [
    OpBuilder<(ins "Type":$ptr, "Value":$base_ptr, "SmallVector<Value>":$indices, "SmallVector<Value>":$strides)>
  ];
}

def ForOp : Arts_Op<"for", [AttrSizedOperandSegments, SingleBlockImplicitTerminator<"arts::YieldOp">]> {
  let summary = "ARTS parallel for loop operation";
  let description = [{
    The `arts.for` operation represents a parallel for loop in the ARTS dialect.
    It preserves the parallel semantics from OpenMP worksharing loops while
    providing a native ARTS construct for parallel iteration.
    
    This operation takes lower bound, upper bound, and step values, and executes
    the loop body in parallel across available workers.
  }];
  
  let arguments = (ins Variadic<Index>:$lowerBound, 
                      Variadic<Index>:$upperBound, 
                      Variadic<Index>:$step,
                      OptionalAttr<ForScheduleKindAttr>:$schedule,
                      Variadic<AnyType>:$reductionAccumulators);
  let results = (outs);
  let regions = (region AnyRegion:$region);
  
  let assemblyFormat = [{
    `(` $lowerBound `)` `to` `(` $upperBound `)` `step` `(` $step `)`
    ( `schedule` `(` $schedule^ `)` )?
    ( `reduction` `(` $reductionAccumulators^ `:` type($reductionAccumulators) `)` )?
    `{` $region `}` attr-dict
  }];
  
  let hasCanonicalizer = true;
}

#endif // ARTS_OPS
