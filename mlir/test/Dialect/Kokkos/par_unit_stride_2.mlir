// RUN: mlir-opt %s --parallel-unit-step
module {
// CHECK-LABEL:   func.func @myfunc(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<?x?x?xf32>,
// CHECK-SAME:                      %[[VAL_1:.*]]: memref<?x?xf32>,
// CHECK-SAME:                      %[[VAL_2:.*]]: index,
// CHECK-SAME:                      %[[VAL_3:.*]]: index,
// CHECK-SAME:                      %[[VAL_4:.*]]: index) -> memref<?x?x?xf32> {
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 3.141590e+00 : f32
// CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_4]], %[[VAL_6]] : index
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_4]], %[[VAL_3]] : index
// CHECK:           %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_6]] : index
// CHECK:           %[[VAL_11:.*]] = arith.divui %[[VAL_10]], %[[VAL_3]] : index
// CHECK:           %[[VAL_12:.*]] = arith.subi %[[VAL_4]], %[[VAL_6]] : index
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_3]] : index
// CHECK:           %[[VAL_14:.*]] = arith.subi %[[VAL_13]], %[[VAL_6]] : index
// CHECK:           %[[VAL_15:.*]] = arith.divui %[[VAL_14]], %[[VAL_3]] : index
// CHECK:           scf.parallel (%[[VAL_16:.*]], %[[VAL_17:.*]], %[[VAL_18:.*]]) = (%[[VAL_5]], %[[VAL_5]], %[[VAL_5]]) to (%[[VAL_8]], %[[VAL_11]], %[[VAL_15]]) step (%[[VAL_6]], %[[VAL_6]], %[[VAL_6]]) {
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_16]], %[[VAL_6]] : index
// CHECK:             %[[VAL_20:.*]] = arith.muli %[[VAL_17]], %[[VAL_3]] : index
// CHECK:             %[[VAL_21:.*]] = arith.muli %[[VAL_18]], %[[VAL_3]] : index
// CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_6]] : index
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_19]], %[[VAL_20]], %[[VAL_22]]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_24:.*]] = arith.addf %[[VAL_23]], %[[VAL_7]] : f32
// CHECK:             memref.store %[[VAL_24]], %[[VAL_0]]{{\[}}%[[VAL_19]], %[[VAL_20]], %[[VAL_22]]] : memref<?x?x?xf32>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_0]] : memref<?x?x?xf32>
// CHECK:         }
  func.func @myfunc(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?xf32>, %arg2: index, %arg3: index, %arg4: index) -> memref<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cPI = arith.constant 3.14159 : f32
    scf.parallel (%i, %j, %k) = (%c1, %c0, %c1) to (%arg4, %arg4, %arg4) step (%c1, %arg3, %arg3) {
      %0 = memref.load %arg0[%i, %j, %k] : memref<?x?x?xf32>
      %2 = arith.addf %0, %cPI : f32
      memref.store %2, %arg0[%i, %j, %k] : memref<?x?x?xf32>
      scf.yield
    }
    return %arg0 : memref<?x?x?xf32>
  }
}
