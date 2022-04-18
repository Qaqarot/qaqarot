
import numpy as np
from .baseFunc import StateAfterMeasurement

def Xsqueeze(state, N, idx, r):
    mu = state[0]
    V = state[1]
    idx = 2 * idx
    S = np.eye(2 * N)
    S[idx:idx+2, idx:idx+2] = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
    V_ = np.dot(S, np.dot(V, S.T))
    mu_ = np.dot(S, mu)
    return [mu_, V_]

def Rotation(state, N, idx, theta):
    mu = state[0]
    V = state[1]
    idx = 2 * idx
    S = np.eye(2 * N)
    S[idx:idx+2, idx:idx+2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    V_ = np.dot(S, np.dot(V, S.T))
    mu_ = np.dot(S, mu)
    return [mu_, V_]

# 10.1103/RevModPhys.77.513
def Beamsplitter(state, N, idx1, idx2, theta):
    mu = state[0]
    V = state[1]
    idx1 = 2 * idx1
    idx2 = 2 * idx2
    S = np.eye(2 * N)
    S[idx1:idx1+2, idx1:idx1+2] = np.array([[np.cos(theta), 0], [0, np.cos(theta)]])
    S[idx1:idx1+2, idx2:idx2+2] = np.array([[np.sin(theta), 0], [0, np.sin(theta)]])
    S[idx2:idx2+2, idx1:idx1+2] = np.array([[-np.sin(theta), 0], [0, -np.sin(theta)]])
    S[idx2:idx2+2, idx2:idx2+2] = np.array([[np.cos(theta), 0], [0, np.cos(theta)]])
    V_ = np.dot(S, np.dot(V, S.T))
    mu_ = np.dot(S, mu)
    return [mu_, V_]

def Displace(state, idx, alpha):
    mu = state[0]
    V = state[1]
    dx = np.real(alpha) * np.sqrt(2) # np.sqrt(2 * hbar)
    dp = np.imag(alpha) * np.sqrt(2) # np.sqrt(2 * hbar)
    mu[2 * idx:2 * idx + 2] = mu[2 * idx:2 * idx + 2] + np.array([dx, dp])
    mu_ = mu
    return [mu_, V]

def Xtrans(state, idx, dx):
    mu = state[0]
    V = state[1]
    mu[2 * idx] += dx
    mu_ = mu
    return [mu_, V]

def Ztrans(state, idx, dp):
    mu = state[0]
    V = state[1]
    mu[2 * idx + 1] += dp
    mu_ = mu
    return [mu_, V]

def twoModeSqueezing(state, N, idx1, idx2, r):
    mu = state[0]
    V = state[1]
    idx1 = 2 * idx1
    idx2 = 2 * idx2
    S = np.eye(2 * N)
    S[idx1:idx1+2, idx1:idx1+2] = np.array([[np.cosh(r), 0], [0, np.cosh(r)]])
    S[idx1:idx1+2, idx2:idx2+2] = np.array([[np.sinh(r), 0], [0, -np.sinh(r)]])
    S[idx2:idx2+2, idx1:idx1+2] = np.array([[np.sinh(r), 0], [0, -np.sinh(r)]])
    S[idx2:idx2+2, idx2:idx2+2] = np.array([[np.cosh(r), 0], [0, np.cosh(r)]])
    V_ = np.dot(S, np.dot(V, S.T))
    mu_ = np.dot(S, mu)
    return [mu_, V_]

def HomodyneX(state, idx):
    mu = state[0]
    V = state[1]
    res = np.random.normal(mu[2 * idx], np.sqrt(V[2 * idx, 2 * idx]))
    mu_, V_ = StateAfterMeasurement(mu, V, idx, res, np.diag([1, 0]))
    return res, [mu_, V_]

def HomodyneP(state, idx):
    mu = state[0]
    V = state[1]
    res = np.random.normal(mu[2 * idx + 1], np.sqrt(V[2 * idx + 1, 2 * idx + 1]))
    mu_, V_ = StateAfterMeasurement(mu, V, idx, res, np.diag([0, -1]))
    return res, [mu_, V_]


