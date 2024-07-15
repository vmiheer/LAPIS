//===- InitAllTranslations.h - MLIR Translations Registration ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all translations
// in and out of MLIR to the system.
//
//===----------------------------------------------------------------------===//

#ifndef LAPIS_INITALLTRANSLATIONS_H
#define LAPIS_INITALLTRANSLATIONS_H

#include "mlir/InitAllTranslations.h"

namespace mlir {

void registerToKokkosTranslation();

// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerAllKokkosTranslations() {
  registerAllTranslations();
  static bool initOnce = []() {
    registerToKokkosTranslation();
    return true;
  }();
  (void)initOnce;
}
} // namespace mlir

#endif // LAPIS_INITALLTRANSLATIONS_H
