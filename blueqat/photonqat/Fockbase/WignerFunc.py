

"""
`WignerFunc` module implements calculation of Wigner function.
This module is internally used.
"""

import numpy as np
from scipy.special import factorial as fact

def FockWigner(xmat, pmat, fockState, mode, method = 'clenshaw', tol=1e-10):
    if fockState.ndim < mode + 1:
        raise  ValueError("The mode is not exist.")
    if fockState.ndim > 1:
        rho = reduceState(fockState, mode)
    else:
        rho = np.outer(np.conj(fockState), fockState)
    if method == 'moyal':
        W = _Wigner_Moyal(rho, xmat, pmat, tol)
    elif method == 'clenshaw':
        W = _Wigner_clenshaw(rho, xmat, pmat, tol)
    else:
        raise ValueError("method is invalid.")
    return W

def reduceState(fockState, mode):
    modeNum = fockState.ndim
    cutoff = fockState.shape[-1] - 1
    fockState = np.swapaxes(fockState, mode, -1)
    fockState = fockState.flatten()
    rho = np.outer(np.conj(fockState), fockState)
    for i in range(modeNum - 1):
        rho = partialTrace(rho, cutoff)
    return rho

def partialTrace(rho, cutoff):
    dim1 = np.int(cutoff + 1)
    dim2 = np.int(rho.shape[0] / dim1)
    rho_ = np.zeros([dim2, dim2]) + 0j
    for j in range(dim1):
        rho_ += rho[(j * dim2):(j * dim2 + dim2), (j * dim2):(j * dim2 + dim2)]
    return rho_

def _Wigner_Moyal(rho, xmat, pmat, tol):
    dim = rho.shape[0]
    [l, m] = np.indices([dim, dim])
    A = np.max(np.dstack([l, m]), axis=2)
    B = np.abs(l - m)
    C = np.min(np.dstack([l, m]), axis=2)
    R0 = xmat**2 + pmat**2
    xmat = xmat[:, :, np.newaxis, np.newaxis]
    pmat = pmat[:, :, np.newaxis, np.newaxis]
    R = xmat**2 + pmat**2
    X = xmat - np.sign(l-m) * 1j * pmat
    W = 2 * (-1)**C * np.sqrt(2**(B) * fact(C) / fact(A))
    W = W * np.exp(-R) * X**(B)
    S = _Sonin(C, B, 2 * R0)
    W = W * S
    W = rho * W
    W = np.sum(np.sum(W, axis = -1), axis = -1)
    if np.max(np.imag(W)) < tol:
        W = np.real(W)
    else:
        raise ValueError("Wigner plot has imaginary value.")
    return W

# Based on QuTiP
def _Wigner_clenshaw(rho, xmat, pmat, tol, hbar = 1):
    g = np.sqrt(2 / hbar)
    M = rho.shape[0]
    A2 = g * (xmat + 1.0j * pmat)    
    B = np.abs(A2)
    B *= B
    w0 = (2*rho[0, -1])*np.ones_like(A2)
    L = M-1
    rho = rho * (2*np.ones((M,M)) - np.diag(np.ones(M)))
    while L > 0:
        L -= 1
        w0 = _Wigner_laguerre(L, B, np.diag(rho, L)) + w0 * A2 * (L+1)**-0.5
    W = w0 * np.exp(-B * 0.5) * (g ** 2 * 0.5 / np.pi)
    # if np.max(np.imag(W)) < tol:
    #     W = np.real(W)
    # else:
    #     raise ValueError("Wigner plot has imaginary value.")
    W = np.real(W)
    return W

def _Wigner_laguerre(L, x, c):
    
    if len(c) == 1:
        y0 = c[0]
        y1 = 0
    elif len(c) == 2:
        y0 = c[0]
        y1 = c[1]
    else:
        k = len(c)
        y0 = c[-2]
        y1 = c[-1]
        for i in range(3, len(c) + 1):
            k -= 1
            y0,    y1 = c[-i] - y1 * (float((k - 1)*(L + k - 1))/((L+k)*k))**0.5, \
            y0 - y1 * ((L + 2*k -1) - x) * ((L+k)*k)**-0.5
            
    return y0 - y1 * ((L + 1) - x) * (L + 1)**-0.5

def _to_2d_ndarray(a):
    if isinstance(a,(np.ndarray)):
        return a
    else:
        return np.array([[a]])

# slow!
def _Sonin(n, alpha, x):
    n = _to_2d_ndarray(n)
    alpha = _to_2d_ndarray(alpha)
    x = _to_2d_ndarray(x)
    a = fact(n + alpha)
    k0 = np.arange(np.max(n) + 1)
    k0 = k0[:, np.newaxis, np.newaxis]
    k = k0 * np.ones([np.max(n) + 1, n.shape[0], n.shape[0]], dtype = np.int)
    mask = np.ones(k.shape, dtype = np.int)
    for i in range(k.shape[0]):
        ind = (np.ones(n.shape) * i) > n
        mask[i, ind] = 0
    k *= mask
    S = mask * (-1)**k * a / fact(n - k) / fact(k + alpha) / fact(k)
    X = x ** k0
    S = S[:, np.newaxis, np.newaxis, :, :] * X[:, :, :, np.newaxis, np.newaxis]
    return np.sum(S, axis = 0)