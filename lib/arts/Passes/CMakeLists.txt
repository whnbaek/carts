# Print it here
message(STATUS "POLYGEIST_INCLUDE_DIR: ${POLYGEIST_BUILD_DIR}/include")
message(STATUS "POLYGEIST_LIBRARY_DIR: ${POLYGEIST_BUILD_DIR}/lib")

add_mlir_dialect_library(MLIRArtsTransforms
  ArtsInliner.cpp
  ConvertOpenMPToArts.cpp
  Edt.cpp
  CreateDbs.cpp
  CanonicalizeMemrefs.cpp
  Db.cpp
  CreateEpochs.cpp
  ConvertArtsToLLVM.cpp
  EpochLowering.cpp
  EdtLowering.cpp
  EdtPtrRematerialization.cpp
  EdtInvariantCodeMotion.cpp
  EdtPtrRematerialization.cpp
  CanonicalizeDbs.cpp
  DbLowering.cpp
  ParallelEdtLowering.cpp
  EdtLowering.cpp
  Concurrency.cpp
  ${CARTS_TRANSFORMS_SOURCE_FILES}
  ${CARTS_UTILS_SOURCE_FILES}
  ${CARTS_CODEGEN_SOURCE_FILES}
  ${CARTS_ANALYSIS_SOURCE_FILES}

DEPENDS
  MLIRArtsOpsIncGen
  MLIRArtsPassIncGen

LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRArts
  MLIRSideEffectInterfaces
  MLIRSCFToControlFlow
  MLIRTargetLLVMIRImport
  MLIRTransformUtils
  MLIRControlFlowToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRArithToLLVM
  MLIROpenMPToLLVM
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(MLIRArtsTransforms PUBLIC stdc++fs)
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
    target_link_libraries(MLIRArtsTransforms PUBLIC c++experimental)
  elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(MLIRArtsTransforms PUBLIC c++fs)
  endif()
endif()

# Find Polygeist lib
target_link_libraries(MLIRArtsTransforms PRIVATE ${MLIR_POLYGEIST_LIB})

target_compile_definitions(obj.MLIRArtsTransforms
  PRIVATE
  CARTS_PGO_DEFAULT_DATA_DIR="${CARTS_PGO_DEFAULT_DATA_DIR}"
  CARTS_PGO_ALTERNATIVE_ENV_VAR="${CARTS_PGO_ALTERNATIVE_ENV_VAR}"
  CARTS_PGO_DATA_DIR_ENV_VAR="${CARTS_PGO_DATA_DIR_ENV_VAR}"
)
