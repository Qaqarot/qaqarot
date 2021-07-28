"""This module manages the set of operations, and provides a factory method of operations."""

from typing import Dict, Optional, Type

from . import gate
from .typing import Targets

GATE_SET: Dict[str, Type[gate.Operation]] = {
    # 1 qubit gates (alphabetical)
    "h": gate.HGate,
    "i": gate.IGate,
    "mat1": gate.Mat1Gate,
    "p": gate.PhaseGate,
    "phase": gate.PhaseGate,
    "r": gate.PhaseGate,
    "rx": gate.RXGate,
    "ry": gate.RYGate,
    "rz": gate.RZGate,
    "s": gate.SGate,
    "sdg": gate.SDagGate,
    "sx": gate.SXGate,
    "sxdg": gate.SXDagGate,
    "t": gate.TGate,
    "tdg": gate.TDagGate,
    "u": gate.UGate,
    "x": gate.XGate,
    "y": gate.YGate,
    "z": gate.ZGate,
    # Controlled gates (alphabetical)
    "ccx": gate.ToffoliGate,
    "ccz": gate.CCZGate,
    "cnot": gate.CXGate,
    "ch": gate.CHGate,
    "cp": gate.CPhaseGate,
    "cphase": gate.CPhaseGate,
    "cr": gate.CPhaseGate,
    "crx": gate.CRXGate,
    "cry": gate.CRYGate,
    "crz": gate.CRZGate,
    "cswap": gate.CSwapGate,
    "cu": gate.CUGate,
    "cx": gate.CXGate,
    "cy": gate.CYGate,
    "cz": gate.CZGate,
    "toffoli": gate.ToffoliGate,
    # Other multi qubit gates (alphabetical)
    "rxx": gate.RXXGate,
    "ryy": gate.RYYGate,
    "rzz": gate.RZZGate,
    "swap": gate.SwapGate,
    "zz": gate.ZZGate,
    # Measure and reset (alphabetical)
    "m": gate.Measurement,
    "measure": gate.Measurement,
    "reset": gate.Reset,
}

def get_op_type(name: str) -> Optional[Type[gate.Operation]]:
    """Get a class of operation from operation name."""
    return GATE_SET.get(name)

def create(name: str, targets: Targets, params: tuple) -> gate.Operation:
    """Create an operation from name, targets and params."""
    op_type = get_op_type(name)
    if op_type is None:
        raise ValueError(f"Unknown operation `{name}'.")
    return op_type.create(targets, params)

def register_operation(name: str, op_type: Type[gate.Operation]) -> None:
    """Register an operation. If operation is already exists, overwrite it."""
    GATE_SET[name] = op_type

def unregister_operation(name: str) -> None:
    """Unregister an operation. If operation is not exists, do nothing."""
    try:
        del GATE_SET[name]
    except KeyError:
        pass
