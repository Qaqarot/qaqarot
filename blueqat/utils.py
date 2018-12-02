from collections import Counter

def to_inttuple(bitstr):
    """Convert from bit string likes '01011' to int tuple likes (0, 1, 0, 1, 1)"""
    if isinstance(bitstr, str):
        return tuple(int(b) for b in bitstr)
    if isinstance(bitstr, Counter):
        return Counter((tuple(int(b) for b in k), v) for k, v in bitstr.items())
    if isinstance(bitstr, dict):
        return {tuple(int(b) for b in k): v for k, v in bitstr.items()}
    raise ValueError("bitstr type shall be `str`, `Counter` or `dict`")
