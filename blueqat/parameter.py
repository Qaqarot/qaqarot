from enum import Enum
from collections.abc import Sequence, Mapping
from typing import List, NamedTuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from sympy import Basic as SympyBasic

# ParamAssign = Union[Mapping[str, float], Sequence[float]]
# For python<=3.8
ParamAssign = Union[Mapping, Sequence]

class ParamOp(Enum):
    NEG = 1

class Parameter(NamedTuple):
    name: str
    idx: int
    ops: List[ParamOp] = []

    def subs(self, params: ParamAssign) -> float:
        if isinstance(params, Mapping):
            val = params[self.name]
        else:
            val = params[self.idx]
        for op in self.ops:
            if op == ParamOp.NEG:
                val = -val
            else:
                raise ValueError(f'Unknown ParamOp {op}')
        return val

    def __neg__(self) -> 'Parameter':
        return Parameter(self.name, self.idx, self.ops + [ParamOp.NEG])

FParam = Union[float, Parameter, "SympyBasic"]