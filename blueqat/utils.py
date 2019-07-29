# Copyright 2019 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

def ignore_global_phase(statevec):
    """Multiple e^-iθ to `statevec` where θ is a phase of first non-zero element.

    Args:
        statevec np.array: Statevector.

    Returns:
        np.array: `statevec` is returned.
    """
    for q in statevec:
        if abs(q) > 0.0000001:
            ang = abs(q) / q
            statevec *= ang
            break
    return statevec
