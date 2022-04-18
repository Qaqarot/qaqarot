'''
References
arxiv: quant-ph/0503237
Physical Review A 49, 1567 (1994)

Conversion between vectors of deffrent ordered operators
R = (q_1, p_1, ..., q_n, p_n)
S = (q_1, ..., q_n, p_1, ..., p_n)
T = (a_1, ..., a^*_1, ..., a^*n)
'''

import numpy as np

def StoTmat(cov):
    N2 = cov.shape[0]
    N = np.int(N2/2)
    Omega = np.zeros([N2, N2]) + 0j
    for i in range(N):
        Omega[i, i] = 1 / np.sqrt(2)
        Omega[i, i + N] = 1j / np.sqrt(2)
        Omega[i + N, i] = 1 / np.sqrt(2)
        Omega[i + N, i + N] = -1j / np.sqrt(2)

    cov_T = np.dot(Omega, np.dot(cov, np.transpose(np.conj(Omega))))
    return cov_T

def RtoSmat(cov):
    N2 = cov.shape[0]
    N = np.int64(N2/2)
    P = np.zeros([N2, N2]) + 0j
    for i in range(N):
        P[i, i * 2] = 1
    for i in range(N, N2):
        P[i, (i-N) * 2 + 1] = 1
    cov_T = np.dot(P, np.dot(cov, np.transpose(P)))
    return cov_T

def RtoTmat(cov):
    cov_T = RtoSmat(cov)
    return StoTmat(cov_T)

def StoTvec(vec):
    N2 = len(vec)
    N = np.int(N2/2)
    T = np.zeros(N2) + 0j
    T[:N] = (vec[:N] + 1j * vec[N:N2]) / np.sqrt(2)
    T[N:N2] = (vec[:N] - 1j * vec[N:N2]) / np.sqrt(2)
    return T

def RtoSvec(vec):
    N2 = len(vec)
    N = np.int(N2/2)
    S = np.zeros(N2) + 0j
    S[:N] = vec[0:N2:2]
    S[N:N2] = vec[1:N2:2]
    return S

def RtoTvec(vec):
    S = RtoSvec(vec)
    return StoTvec(S)

def RtoQmat(cov):
    N = np.int(cov.shape[0] / 2)
    V = RtoSmat(cov)
    Vxx = V[:N, :N]
    Vxp = V[:N, N:]
    Vpp = V[N:, N:]
    A = Vxx - 1j * (Vxp - Vxp.T) + Vpp
    B = Vxx + 1j * (Vxp - Vxp.T) + Vpp
    C = Vxx - 1j * (Vxp + Vxp.T) - Vpp
    sigma_Q = np.block([[A, C], [np.conj(C), B]]) * 0.5 + np.eye(2 * N) * 0.5
    return sigma_Q