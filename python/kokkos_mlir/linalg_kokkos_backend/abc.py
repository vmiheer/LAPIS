# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import abc
from typing import TypeVar

# A type shared between the result of `LinalgOnTensorsBackend.compile` and the
# input to `LinalgOnTensorsBackend.load`. Each backend will likely have a
# different definition of this type.
CompiledArtifact = TypeVar('CompiledArtifact')

# A wrapper around a backend-specific loaded program representation
# that uniformly translates the `x.method(...)` interface expected of
# Torch modules into appropriate lower-level operations.
Invoker = TypeVar('Invoker')


class LinalgKokkosBackend(abc.ABC):
    """The interface to an linalg-on-tensors backend.

    Backends are recommended to raise meaningful exceptions in case of error,
    ideally with easy reproduction instructions.
    """
    @abc.abstractmethod
    def compile(self, module) -> CompiledArtifact:
        """Compile the provided MLIR module into a compiled artifact.

        The module adheres to the linalg-on-tensors backend contract
        (see the VerifyLinalgOnTensorsBackendContract pass).

        The compiled artifact can be any type, but must be correctly
        interpreted by the `load` method.
        """

