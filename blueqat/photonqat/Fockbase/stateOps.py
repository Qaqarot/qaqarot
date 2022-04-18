
import numpy as np
from scipy.special import factorial as fact

def coherent(initState, mode, alpha, cutoff):
    n = np.arange(cutoff + 1)
    state = np.exp(- 0.5 * np.abs(alpha) ** 2) / np.sqrt(fact([n])) * alpha ** n
    initState[mode, :] = state
    return initState

def vacuum(initState, mode):
    initState[mode, :] = 0
    initState[mode, 0] = 1
    return initState

# arXiv:quant-ph/0509137
def cat(initState, mode, alpha, parity, cutoff):
    n = np.arange(cutoff + 1)
    if parity == 'e':
        N = 1 / np.sqrt(2 * (1 + np.exp(-2 * np.abs(alpha) ** 2)))
        coeff = 2 * N * np.exp(-(np.abs(alpha) ** 2) / 2)
        initState[mode, :] = coeff * alpha ** (n) / np.sqrt(fact(n)) * np.mod(n + 1, 2)
    elif parity == 'o':
        N = 1 / np.sqrt(2 * (1 - np.exp(-2 * np.abs(alpha)**2)))
        coeff = 2 * N * np.exp(-(np.abs(alpha) ** 2) / 2)
        initState[mode, :] = coeff * alpha ** (n) / np.sqrt(fact(n)) * np.mod(n, 2)
    return initState

def photonNumState(initState, mode, n, cutoff):
    photonNumState = np.zeros(cutoff + 1)
    photonNumState[n] = 1
    initState[mode, :] = photonNumState
    return initState