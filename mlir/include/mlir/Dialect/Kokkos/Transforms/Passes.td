//===-- Passes.td - Kokkos pass definition file -----*- tablegen -*-===//

#ifndef MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES
#define MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ParallelUnitStep : Pass<"parallel-unit-step"> {
  let summary = "Rewrites scf.parallel ops to have lower bound 0 and step size 1 in all dimensions";
  let description = [{
  }];
  let constructor = "mlir::createParallelUnitStepPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect"
  ];
}

def KokkosLoopMapping : Pass<"kokkos-loop-mapping"> {
  let summary = "Maps nested SCF parallel loops to Kokkos parallel loops";
  let description = [{
  }];
  let constructor = "mlir::createKokkosLoopMappingPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "kokkos::KokkosDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def KokkosMemorySpaceAssignment : Pass<"kokkos-assign-memory-spaces"> {
  let summary = "Determines the MemorySpace attributes for each memref value";
  let description = [{
  }];
  let constructor = "mlir::createKokkosMemorySpaceAssignmentPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "kokkos::KokkosDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def KokkosDualViewManagement : Pass<"kokkos-dualview-management", "ModuleOp"> {
  let summary = "Insert sync and modify operations for DualView-typed memrefs";
  let description = [{
  }];
  let constructor = "mlir::createKokkosDualViewManagementPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "kokkos::KokkosDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

// Old Kokkos GPU codegen (not using Kokkos dialect ops)
def SparseKokkosCodegen : Pass<"sparse-kokkos-codegen", "ModuleOp"> {
  let summary = "Generates Kokkos code during sparsification";
  let description = [{
    Enables sparse compiler to generate Kokkos code, which can use non-host backends.
  }]; 
  let constructor = "mlir::createSparseKokkosCodegenPass()";
  let dependentDialects = [ 
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "emitc::EmitCDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparse_tensor::SparseTensorDialect",
  ];  
}

#endif // MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES
