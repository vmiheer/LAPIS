// RUN: mlir-opt %s --parallel-unit-step
module {
// CHECK-LABEL:   func.func @myfunc(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<?x?xf32>,
// CHECK-SAME:                      %[[VAL_1:.*]]: memref<?x?xf32>,
// CHECK-SAME:                      %[[VAL_2:.*]]: index) -> memref<?x?xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = memref.alloc(%[[VAL_2]], %[[VAL_2]]) {alignment = 64 : i64} : memref<?x?xf32>
// CHECK:           scf.parallel (%[[VAL_7:.*]], %[[VAL_8:.*]]) = (%[[VAL_3]], %[[VAL_3]]) to (%[[VAL_4]], %[[VAL_2]]) step (%[[VAL_5]], %[[VAL_5]]) {
// CHECK:             %[[VAL_9:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_8]]] : memref<?x?xf32>
// CHECK:             %[[VAL_10:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_8]], %[[VAL_8]]] : memref<?x?xf32>
// CHECK:             %[[VAL_11:.*]] = arith.addf %[[VAL_9]], %[[VAL_10]] : f32
// CHECK:             memref.store %[[VAL_11]], %[[VAL_6]]{{\[}}%[[VAL_8]], %[[VAL_7]]] : memref<?x?xf32>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_6]] : memref<?x?xf32>
// CHECK:         }
  func.func @myfunc(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: index) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc(%arg2, %arg2) {alignment = 64 : i64} : memref<?x?xf32>
    scf.parallel (%i, %j) = (%c0, %c0) to (%c5, %arg2) step (%c1, %c1) {
      %0 = memref.load %arg0[%i, %j] : memref<?x?xf32>
      %1 = memref.load %arg1[%j, %j] : memref<?x?xf32>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %alloc[%j, %i] : memref<?x?xf32>
      scf.yield
    }
    return %alloc : memref<?x?xf32>
  }
}
