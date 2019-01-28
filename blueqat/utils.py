"""Utilities for convenient."""
from collections import Counter

def to_inttuple(bitstr):
    """Convert from bit string likes '01011' to int tuple likes (0, 1, 0, 1, 1)

    Args:
        bitstr (str, Counter, dict): String which is written in "0" or "1".
            If all keys are bitstr, Counter or dict are also can be converted by this function.

    Returns:
        tuple of int, Counter, dict: Converted bits.
            If bitstr is Counter or dict, returns the Counter or dict
            which contains {converted key: original value}.

    Raises:
        ValueError: If bitstr type is unexpected or bitstr contains illegal character.
    """
    if isinstance(bitstr, str):
        return tuple(int(b) for b in bitstr)
    if isinstance(bitstr, Counter):
        return Counter({tuple(int(b) for b in k): v for k, v in bitstr.items()})
    if isinstance(bitstr, dict):
        return {tuple(int(b) for b in k): v for k, v in bitstr.items()}
    raise ValueError("bitstr type shall be `str`, `Counter` or `dict`")
