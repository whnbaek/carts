set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  AggressiveInstCombine
  InstCombine
  Instrumentation
  MC
  IPO
  MCParser
  Option
  ScalarOpts
  Support
  TransformUtils
  Vectorize
)

add_clang_executable(carts-run
  carts-run.cpp
)

install(
  TARGETS carts-run
  EXPORT CartsTargets
  RUNTIME DESTINATION ${CARTS_TOOLS_DIR}
  COMPONENT carts-run
)

target_include_directories(carts-run PRIVATE
  "${LLVM_SOURCE_DIR}/../clang/include"
  "${CMAKE_BINARY_DIR}/tools/clang/include"
)

target_compile_definitions(carts-run PUBLIC -DLLVM_OBJ_ROOT="${LLVM_BINARY_DIR}")

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRArts
  MLIRArtsTransforms
  MLIRFuncAllExtensions
  MLIRPolygeist
  MLIRPolygeistTransforms
  MLIRSupport
  MLIRIR
  MLIRAnalysis
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIROpenMPDialect
  MLIRGPUTransforms
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRFuncToLLVM
  MLIRFunctionInterfaces
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRMathToLLVM
  MLIRTargetLLVMIRImport
  MLIRSCFTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToOpenMP
  MLIROpenMPToLLVM
  MLIROpenMPToLLVMIRTranslation
  MLIRGPUToNVVMTransforms
  MLIRGPUToGPURuntimeTransforms
  MLIRFuncAllExtensions
  MLIRLLVMIRToLLVMTranslation
  MLIRFromLLVMIRTranslationRegistration
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRCAPIRegisterEverything
  MLIRBuiltinToLLVMIRTranslation
)

target_link_libraries(carts-run PRIVATE ${LIBS})
add_dependencies(carts-run MLIRArtsOpsIncGen MLIRArtsPassIncGen)