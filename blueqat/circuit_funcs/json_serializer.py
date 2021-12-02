"""Defines JSON serializer and deserializer."""
import typing
from blueqat import Circuit

from ..gateset import create
from .flatten import flatten

from blueqat.gate import Measurement, Operation

SCHEMA_NAME = 'blueqat-circuit'
AVAILABLE_SCHEMA_VERSIONS = ["1", "2"]
LATEST_SCHEMA_VERSION = "2"

if typing.TYPE_CHECKING:
    from typing import Any, Dict, List, Union
    try:
        from typing import TypedDict
    except ImportError:
        CircuitJsonDictV2 = Dict[str, Any]
        CircuitJsonDictV1 = Dict[str, Any]
        OpJsonDictV2 = Dict[str, Any]
        OpJsonDictV1 = Dict[str, Any]
    else:
        class SchemaJsonDict(TypedDict):
            """Schema header for detect data type"""
            name: str
            version: str

        class OpJsonDictV1(TypedDict):
            """Data type of Operation"""
            name: str
            params: List[float]
            targets: List[int]

        class OpJsonDictV2(TypedDict):
            """Data type of Operation"""
            name: str
            params: List[float]
            options: Dict[str, Any]
            targets: List[int]

        class CircuitJsonDictV1(TypedDict):
            """Data type of Circuit"""
            schema: SchemaJsonDict
            n_qubits: int
            ops: List[OpJsonDictV1]

        class CircuitJsonDictV2(TypedDict):
            """Data type of Circuit"""
            schema: SchemaJsonDict
            n_qubits: int
            ops: List[OpJsonDictV2]

    CircuitJsonDict = Union[CircuitJsonDictV1, CircuitJsonDictV2]


def serialize(c: Circuit) -> 'CircuitJsonDictV2':
    """Serialize Circuit into JSON type dict.

    In this implementation, serialized circuit is flattened.
    However, it's not specifications of JSON schema.
    """
    def serialize_op(op: 'Operation') -> 'OpJsonDictV2':
        targets = op.targets
        if isinstance(targets, slice):
            raise TypeError('Not flatten circuit.')
        if isinstance(targets, int):
            targets = [targets]
        if isinstance(targets, tuple):
            targets = list(targets)
        options = {}
        if isinstance(op, Measurement):
            if op.key is not None:
                options['key'] = op.key
            if op.duplicated is not None:
                options['duplicated'] = op.duplicated
        return {
            'name': str(op.lowername),
            'params': [float(p) for p in op.params],
            'options': options,
            'targets': targets
        }

    c = flatten(c)
    return {
        'schema': {
            'name': SCHEMA_NAME,
            'version': LATEST_SCHEMA_VERSION
        },
        'n_qubits': c.n_qubits,
        'ops': [serialize_op(op) for op in c.ops]
    }


def deserialize(data: 'CircuitJsonDict') -> Circuit:
    """Deserialize JSON type dict into Circuit"""
    def make_op(opdata: 'Union[OpJsonDictV1, OpJsonDictV2]') -> 'Operation':
        return create(opdata['name'],
                      tuple(opdata['targets']),
                      tuple(float(p) for p in opdata['params']),
                      opdata.get('options'))
    schema = data.get('schema', {})
    if schema.get('name', '') != SCHEMA_NAME:
        raise ValueError('Invalid schema')
    if schema.get('version', '') not in AVAILABLE_SCHEMA_VERSIONS:
        raise ValueError('Unknown schema version')
    n_qubits = data['n_qubits']
    ops = data['ops']
    return Circuit(n_qubits, [
        make_op(opdata) for opdata in ops
    ])
