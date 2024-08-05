//===- KokkosOps.td - Kokkos dialect (base, enum attributes, ops) ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KOKKOS_DIALECT_TD_H
#define KOKKOS_DIALECT_TD_H

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ParallelCombiningOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def Kokkos_Dialect : Dialect {
  let name = "kokkos";
  let cppNamespace = "::mlir::kokkos";
  let description = [{
     The `kokkos` dialect provides MLIR representations for Kokkos
     constructs like hierarchical parallelism, parallel_scan, Kokkos
     memory spaces and execution space instances.
  }];
}

// The predicate indicates the type of the comparison to perform:
// (un)orderedness, (in)equality and less/greater than (or equal to) as
// well as predicates that are always true or false.
def Kokkos_MemorySpace : I64EnumAttr<
    "MemorySpace", "",
    [
      I64EnumAttrCase<"Host", 0, "host">,
      I64EnumAttrCase<"Device", 1, "device">,
      I64EnumAttrCase<"DualView", 2, "dualview">
    ]> {
  let cppNamespace = "::mlir::kokkos";
}

def Kokkos_ExecutionSpace : I64EnumAttr<
    "ExecutionSpace", "",
    [
      I64EnumAttrCase<"Host", 0, "host">,
      I64EnumAttrCase<"Device", 1, "device">,
      I64EnumAttrCase<"TeamHandle", 2, "team_handle">
    ]> {
  let cppNamespace = "::mlir::kokkos";
}

def Kokkos_ParallelLevel : I64EnumAttr<
    "ParallelLevel", "",
    [
      I64EnumAttrCase<"RangePolicy", 0, "range">,
      I64EnumAttrCase<"TeamVector", 1, "team_vector">,
      I64EnumAttrCase<"TeamThread", 2, "team_thread">,
      I64EnumAttrCase<"ThreadVector", 3, "thread_vector">
    ]> {
  let cppNamespace = "::mlir::kokkos";
}

def Kokkos_SingleLevel : I64EnumAttr<
    "SingleLevel", "",
    [
      I64EnumAttrCase<"PerTeam", 0, "per_team">,
      I64EnumAttrCase<"PerThread", 1, "per_thread">
    ]> {
  let cppNamespace = "::mlir::kokkos";
}

// Base op class.
class Kokkos_Op<string mnemonic, list<Trait> traits = []>
    : Op<Kokkos_Dialect, mnemonic, traits>;


def RangeParallelOp : Kokkos_Op<"range_parallel",
    [AutomaticAllocationScope,
     AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     SingleBlockImplicitTerminator<"scf::YieldOp">]> {
  let summary = "Kokkos base parallel for operation";
  let description = [{
  }];

  let arguments = (ins Kokkos_ExecutionSpace:$executionSpace,
                       Kokkos_ParallelLevel:$parallelLevel,
                       Variadic<Index>:$upperBound,
                       Variadic<AnyType>:$initVals);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::kokkos::ExecutionSpace":$executionSpace, "::mlir::kokkos::ParallelLevel":$parallelLevel, "ValueRange":$upperBounds,
      "ValueRange":$initVals,
      CArg<"function_ref<void (OpBuilder &, Location, ValueRange)>",
           "nullptr">:$bodyBuilderFn)>,
    OpBuilder<(ins "::mlir::kokkos::ExecutionSpace":$executionSpace, "::mlir::kokkos::ParallelLevel":$parallelLevel, "ValueRange":$upperBounds,
      CArg<"function_ref<void (OpBuilder &, Location, ValueRange)>",
           "nullptr">:$bodyBuilderFn)>,
  ];

  let extraClassDeclaration = [{
    ValueRange getInductionVars() {
      return getBody()->getArguments();
    }
    unsigned getNumLoops() { return getUpperBound().size(); }
    unsigned getNumReductions() { return getInitVals().size(); }
  }];

  let hasCanonicalizer = 0;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def TeamParallelOp : Kokkos_Op<"team_parallel",
    [AutomaticAllocationScope,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     SingleBlockImplicitTerminator<"scf::YieldOp">]> {
  let summary = "Kokkos TeamPolicy parallel for/reduce";
  let description = [{
  }];

  let arguments = (ins Index:$leagueSize,
                       Index:$teamSizeHint,
                       Index:$vectorLengthHint,
                       Variadic<AnyType>:$initVals);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;

  // ValueRange (body block arguments) should contain exactly 5 values:
  // league size, team size, vector length, league rank, team rank

  // Note that because we don't know the backend/architecture ahead of time,
  // the team size and vector length passed into the body
  // will be computed at runtime based on TeamPolicy::team_size_
  // match the TeamParallel's input values
  let builders = [
    OpBuilder<(ins "Value":$leagueSize, "Value":$teamSizeHint, "Value":$vectorLengthHint,
      "ValueRange":$initVals,
      CArg<"function_ref<void (OpBuilder &, Location, ValueRange)>",
           "nullptr">:$bodyBuilderFn)>,
    OpBuilder<(ins "Value":$leagueSize, "Value":$teamSizeHint, "Value":$vectorLengthHint,
      CArg<"function_ref<void (OpBuilder &, Location, ValueRange)>",
           "nullptr">:$bodyBuilderFn)>,
  ];

  let extraClassDeclaration = [{
    unsigned getNumReductions() { return getInitVals().size(); }
  }];

  let hasCanonicalizer = 0;
  let hasCustomAssemblyFormat = 0;
  let hasVerifier = 0;
}

def TeamBarrierOp : Kokkos_Op<"barrier", [HasParent<"TeamParallelOp">]> {
  let summary = "Synchronize all threads in a team. In C++, t.team_barrier()";
  let description = [{
    Kokkos team barrier, to be called within a TeamPolicy context. All threads in
    the team must reach the barrier together to avoid deadlock, but this is enforced by the
    fact that TeamParallelOp must be the parent.
    ```
    kokkos.barrier
    ```
  }];

  let assemblyFormat = "attr-dict";
}

def SyncOp : Kokkos_Op<"sync"> {

  let description = [{
    For a memref with DualView memory space, sync its contents to the
    provided memory space if it has been marked modified in the opposite space.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the memref to sync">:$view,
                       Kokkos_MemorySpace:$memorySpace);

  let assemblyFormat = [{
    $view attr-dict `:` type($view)
  }];

  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

def ModifyOp : Kokkos_Op<"modify"> {
  let description = [{
    For a memref with DualView memory space, mark its contents as modified
    in the given space. This operation is idempotent.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the memref to mark modified">:$view,
                       Kokkos_MemorySpace:$memorySpace);

  let assemblyFormat = [{
    $view attr-dict `:` type($view)
  }];

  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

def SingleOp : Kokkos_Op<"single", [SingleBlockImplicitTerminator<"kokkos::YieldOp">, RecursiveMemoryEffects, NoRegionArguments, InferTypeOpAdaptor]> {
  let summary = "Restricts a block to execute exactly once per team or per thread";
  let description = [{
    "Restricts a block to execute exactly once per team or per thread"
  }];

  let arguments = (ins Kokkos_SingleLevel:$level);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
  let hasCustomAssemblyFormat = 0;
  let hasVerifier = 0;
}

def ReduceOp : Kokkos_Op<"reduce",
[ParentOneOf<["RangeParallelOp", "TeamParallelOp"]>]>
{
  let summary = "reduce operation for parallel for";
  let description = [{
    The result of the reduce operation's body must have the same type as the
    operands and associated result value of the parallel loop operation.
    Example:

    ```mlir
    %operand = arith.constant 1.0 : f32
    kokkos.reduce(%operand) : f32 {
      ^bb0(%lhs : f32, %rhs: f32):
        %res = arith.addf %lhs, %rhs : f32
        kokkos.yield %res : f32
    }
    ```
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$operand,
      CArg<"function_ref<void (OpBuilder &, Location, Value, Value)>",
           "nullptr">:$bodyBuilderFn)>
  ];

  let arguments = (ins AnyType:$operand);
  let hasCustomAssemblyFormat = 0;
  let regions = (region SizedRegion<1>:$reductionOperator);
  let hasRegionVerifier = 0;
}

def YieldOp : Kokkos_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["RangeParallelOp", "TeamParallelOp", "SingleOp", "ReduceOp"]>]> {
  let summary = "termination operation";
  let description = [{ 
    "kokkos.yield" terminates the region of a parallel, single or reduction op.
    If enclosed in a parallel op, it should have zero operands.
    If enclosed in a single, the operand types must match the results
    of the single, and these are the values that will be broadcast by the single.
    If enclosed in the i'th region of a reduce, the operand is the i'th updated
    partial reduction.
    If the enclosing operation has no results and a yield is not added explicitly,
    then the custom IR builders will insert one automatically.
  }];
    
  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  
  let assemblyFormat = 
      [{  attr-dict ($results^ `:` type($results))? }];
}

#endif
