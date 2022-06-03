"""This module provides for type hinting."""

import typing
from typing import Generic, TypeVar, Union

if typing.TYPE_CHECKING:
    from blueqat import Circuit

T = TypeVar('T')
C = TypeVar('C')
Targets = Union[int, slice, tuple]

class GeneralCircuitOperation(Generic[C, T]):
    """Type definition of dynamic method."""
    def __call__(self, *args) -> T:
        ...

    def __getitem__(self, targets: Targets) -> C:
        ...

CircuitOperation = GeneralCircuitOperation['Circuit', T]
