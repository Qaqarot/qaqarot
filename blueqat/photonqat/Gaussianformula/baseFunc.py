import numpy as np
from .ordering import *
from scipy import special
from thewalrus import hafnian

def GaussianWigner(xi, V, mu):
    xi = xi - mu
    xi_tmp = np.ravel(xi)
    N = np.int(len(xi_tmp) / 2)
    det_V = np.linalg.det(V)
    V_inv = np.linalg.inv(V)
    W = (2 * np.pi)**(-N) / np.sqrt(det_V) * np.exp(-1/2 * np.dot(xi_tmp, np.dot(V_inv, xi_tmp.T)))
    return W

def StateAfterMeasurement(mu, V, idx, res, Pi):
    N = np.int(V.shape[0] / 2)
    subSysA = np.delete(np.delete(V, [2 * idx, 2 * idx + 1], 0), [2 * idx, 2 * idx + 1], 1)
    subSysB = V[(2 * idx):(2 * idx + 2), (2 * idx):(2 * idx + 2)]
    arrayList = []
    for j in range(N):
        if j != idx:
            arrayList.append(V[(2 * j):(2 * j + 2), (2 * idx):(2 * idx + 2)])
    C = np.concatenate(arrayList)
    post_V = subSysA - np.dot(C, np.dot(1 / np.sum(subSysB * Pi) * Pi, C.T))
    post_V = np.insert(post_V, 2 * idx, [[0], [0]], axis = 0)
    post_V = np.insert(post_V, 2 * idx, [[0], [0]], axis = 1)
    post_V[2 * idx, 2 * idx] = 1
    post_V[2 * idx + 1, 2 * idx + 1] = 1
    
    post_mu = np.delete(mu, [2 * idx, 2 * idx + 1]) + \
    np.dot(np.dot(C, 1 / np.sum(subSysB * Pi) * Pi), res * np.diag(Pi) - mu[(2 * idx):(2 * idx + 2)])
    post_mu = np.insert(post_mu, 2 * idx, [0, 0])
    
    return post_mu, post_V

def GaussianQfunc(alpha, V, mu):
    mu_Q = RtoTvec(mu)
    V_Q = RtoTmat(V)
    alpha_Q = RtoTvec(alpha)
    alpha_Q = alpha_Q - mu_Q
    V_Q = V_Q + (np.eye(V_Q.shape[0]) * 0.5)

    det_V_Q = np.linalg.det(V_Q)
    V_Qinv = np.linalg.inv(V_Q)

    Q = 1 / np.sqrt(det_V_Q * np.pi) * np.exp(-1/2 * np.dot(np.conj(alpha_Q), np.dot(V_Qinv, alpha_Q)))

def FockDensityMatrix(cov, mu, m, n, tol = 1e-10):
    if np.max(cov - cov.T) > tol:
        raise ValueError("Covariance matrix must be symmetric.")
    else:
        cov = (cov + cov.T) / 2

    N = np.int(cov.shape[0] / 2)
    cov_Q = RtoQmat(cov)
    cov_Q_inv = np.linalg.inv(cov_Q)
    cov_Q_det = np.linalg.det(cov_Q)
    mu_Q = RtoTvec(mu)

    T1 = np.exp(-0.5 * np.dot(np.dot(np.conj(mu_Q), cov_Q_inv), mu_Q))
    T2 = np.sqrt(cov_Q_det * np.prod(special.factorial(m)) * np.prod(special.factorial(n)))
    T = T1 / T2

    X = np.block([[np.zeros([N, N]), np.eye(N)], [np.eye(N), np.zeros([N, N])]])
    A = np.dot(X, (np.eye(2 * N) - cov_Q_inv))
    A = (A + A.T) / 2 # cancel the numeric error of symmetray
    A_rp = np.repeat(A, np.hstack([n, m]), axis = 0)
    A_rp = np.repeat(A_rp, np.hstack([n, m]), axis = 1)
    gamma = np.dot(np.conj(mu_Q), cov_Q_inv)
    gamma_rp = np.repeat(gamma, np.hstack([n, m]))
    np.fill_diagonal(A_rp, gamma_rp)
    prob = T * hafnian(A_rp, loop = True)
    return prob
