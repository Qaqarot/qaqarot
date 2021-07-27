import typing
from typing import Any
from blueqat import Circuit

SCHEMA_NAME = 'blueqat-circuit'
SCHEMA_VERSION = "1"

if typing.TYPE_CHECKING:
    from blueqat.gate import Operation
    try:
        from typing import List, TypedDict
    except ImportError:
        CircuitJsonDict = Any
        OpJsonDict = Any
    else:
        class SchemaJsonDict(TypedDict):
            name: str
            version: str

        class OpJsonDict(TypedDict):
            name: str
            params: List[float]
            targets: List[int]

        class CircuitJsonDict(TypedDict):
            schema: SchemaJsonDict
            n_qubits: int
            ops: List[OpJsonDict]


def serialize(c: Circuit) -> 'CircuitJsonDict':
    def serialize_op(op: 'Operation') -> 'OpJsonDict':
        # TODO: When targets is slice, what to do?
        if isinstance(op.targets, int):
            targets = [op.targets]
        else:
            targets = [int(t) for t in op.targets]
        return {
            'name': str(op.lowername),
            'params': [float(p) for p in op.params],
            'targets': targets
        }

    return {
        'schema': {
            'name': SCHEMA_NAME,
            'version': SCHEMA_VERSION
        },
        'n_qubits': c.n_qubits,
        'ops': [serialize_op(op) for op in c.ops]
    }


def deserialize(data: 'CircuitJsonDict') -> Circuit:
    def make_op(opdata: 'OpJsonDict') -> 'Operation':
        from blueqat.circuit import GATE_SET
    schema = data.get('schema', {})
    if schema.get('name', '') != SCHEMA_NAME:
        raise ValueError('Invalid schema')
    if schema.get('version', '') != SCHEMA_VERSION:
        raise ValueError('Unknown schema version')
    try:
        n_qubits = data['n_qubits']
    except KeyError:
        raise ValueError('Invalid data')
    if not isinstance(n_qubits, int):
        raise TypeError('Invalid n_qubits type')
    try:
        ops = data['ops']
    except KeyError:
        raise ValueError('Invalid data')
    return Circuit(n_qubits, [
        # TODO: Impl.
    ])
