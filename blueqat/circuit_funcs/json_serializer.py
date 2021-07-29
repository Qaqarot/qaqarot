"""Defines JSON serializer and deserializer."""
import typing
from blueqat import Circuit

from ..gateset import create
from .flatten import flatten

SCHEMA_NAME = 'blueqat-circuit'
SCHEMA_VERSION = "1"

if typing.TYPE_CHECKING:
    from blueqat.gate import Operation
    try:
        from typing import Any, Dict, List, TypedDict
    except ImportError:
        CircuitJsonDict = Dict[str, Any]
        OpJsonDict = Dict[str, Any]
    else:
        class SchemaJsonDict(TypedDict):
            """Schema header for detect data type"""
            name: str
            version: str

        class OpJsonDict(TypedDict):
            """Data type of Operation"""
            name: str
            params: List[float]
            targets: List[int]

        class CircuitJsonDict(TypedDict):
            """Data type of Circuit"""
            schema: SchemaJsonDict
            n_qubits: int
            ops: List[OpJsonDict]


def serialize(c: Circuit) -> 'CircuitJsonDict':
    """Serialize Circuit into JSON type dict.

    In this implementation, serialized circuit is flattened.
    However, it's not specifications of JSON schema.
    """
    def serialize_op(op: 'Operation') -> 'OpJsonDict':
        targets = op.targets
        if isinstance(targets, slice):
            raise TypeError('Not flatten circuit.')
        if isinstance(targets, int):
            targets = [targets]
        if isinstance(targets, tuple):
            targets = list(targets)
        return {
            'name': str(op.lowername),
            'params': [float(p) for p in op.params],
            'targets': targets
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
