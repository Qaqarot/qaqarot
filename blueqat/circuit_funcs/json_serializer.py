"""Defines JSON serializer and deserializer."""
import typing
from typing import Any
from blueqat import Circuit

from ..gateset import create
from .flatten import flatten

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
    """Serialize Circuit into JSON type dict"""
    def serialize_op(op: 'Operation') -> 'OpJsonDict':
        target = op.targets
        if not isinstance(target, int):
            raise TypeError('Not flatten circuit.')
        return {
            'name': str(op.lowername),
            'params': [float(p) for p in op.params],
            'targets': [target]
        }

    c = flatten(c)
    return {
        'schema': {
            'name': SCHEMA_NAME,
            'version': SCHEMA_VERSION
        },
        'n_qubits': c.n_qubits,
        'ops': [serialize_op(op) for op in c.ops]
    }


def deserialize(data: 'CircuitJsonDict') -> Circuit:
    """Deserialize JSON type dict into Circuit"""
    def make_op(opdata: 'OpJsonDict') -> 'Operation':
        return create(opdata['name'],
                      tuple(opdata['targets']),
                      tuple(float(p) for p in opdata['params']))
    schema = data.get('schema', {})
    if schema.get('name', '') != SCHEMA_NAME:
        raise ValueError('Invalid schema')
    if schema.get('version', '') != SCHEMA_VERSION:
        raise ValueError('Unknown schema version')
    n_qubits = data['n_qubits']
    ops = data['ops']
    return Circuit(n_qubits, [
        make_op(opdata) for opdata in ops
    ])
