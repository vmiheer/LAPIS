// RUN: mlir-opt %s --parallel-unit-step
module {
// CHECK-LABEL:   func.func @myfunc(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<?x?x?xf64>,
// CHECK-SAME:                      %[[VAL_1:.*]]: memref<?xindex>,
// CHECK-SAME:                      %[[VAL_2:.*]]: memref<?xf64>,
// CHECK-SAME:                      %[[VAL_3:.*]]: memref<?xf64>) {
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 5 : index
// CHECK:           scf.parallel (%[[VAL_7:.*]]) = (%[[VAL_5]]) to (%[[VAL_6]]) step (%[[VAL_4]]) {
// CHECK:             %[[VAL_8:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_5]], %[[VAL_4]]] : memref<?x?x?xf64>
// CHECK:             %[[VAL_9:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_7]]] : memref<?xindex>
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_4]] : index
// CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK:             %[[VAL_12:.*]] = arith.subi %[[VAL_11]], %[[VAL_9]] : index
// CHECK:             %[[VAL_13:.*]] = scf.parallel (%[[VAL_14:.*]]) = (%[[VAL_5]]) to (%[[VAL_12]]) step (%[[VAL_4]]) init (%[[VAL_8]]) -> f64 {
// CHECK:               %[[VAL_15:.*]] = arith.addi %[[VAL_9]], %[[VAL_14]] : index
// CHECK:               %[[VAL_16:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK:               %[[VAL_17:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_15]]] : memref<?xf64>
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_16]]] : memref<?xf64>
// CHECK:               %[[VAL_19:.*]] = arith.mulf %[[VAL_17]], %[[VAL_18]] : f64
// CHECK:               scf.reduce(%[[VAL_19]])  : f64 {
// CHECK:               ^bb0(%[[VAL_20:.*]]: f64, %[[VAL_21:.*]]: f64):
// CHECK:                 %[[VAL_22:.*]] = arith.addf %[[VAL_20]], %[[VAL_21]] : f64
// CHECK:                 scf.reduce.return %[[VAL_22]] : f64
// CHECK:               }
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             memref.store %[[VAL_13]], %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]]] : memref<?x?x?xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
  func.func @myfunc(%arg0: memref<?x?x?xf64>, %arg1: memref<?xindex>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    scf.parallel (%arg4) = (%c0) to (%c5) step (%c1) {
      %0 = memref.load %arg0[%arg4, %c0, %c1] : memref<?x?x?xf64>
      %1 = memref.load %arg1[%arg4] : memref<?xindex>
      %2 = arith.addi %arg4, %c1 : index
      %3 = memref.load %arg1[%2] : memref<?xindex>
      %4 = scf.parallel (%arg5) = (%1) to (%3) step (%c1) init (%0) -> f64 {
        %5 = memref.load %arg1[%arg5] : memref<?xindex>
        %6 = memref.load %arg2[%arg5] : memref<?xf64>
        %7 = memref.load %arg3[%5] : memref<?xf64>
        %8 = arith.mulf %6, %7 : f64
        scf.reduce(%8)  : f64 {
        ^bb0(%arg7: f64, %arg8: f64):
          %9 = arith.addf %arg7, %arg8 : f64
          scf.reduce.return %9 : f64
        }
        scf.yield
      }
      memref.store %4, %arg0[%arg4, %arg4, %arg4] : memref<?x?x?xf64>
      scf.yield
    }
    return
  }
}
