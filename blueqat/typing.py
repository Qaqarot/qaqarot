"""This module provides for type hinting."""

import typing
from typing import Generic, TypeVar, Union

if typing.TYPE_CHECKING:
    from blueqat import Circuit

T = TypeVar('T')
Targets = Union[int, slice, tuple]

class CircuitOperation(Generic[T]):
    """Type definition of dynamic method."""
    def __call__(self, *args) -> T:
        ...

    def __getitem__(self, targets: Targets) -> 'Circuit':
        ...
