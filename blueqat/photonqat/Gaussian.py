import numpy as np
from .Gaussianformula.baseFunc import *
from .Gaussianformula.ordering import *
from .Gaussianformula.gates import *
import matplotlib.pyplot as plt

GATE_SET = {
    "D": Dgate,
    "BS": BSgate,
    "S": Sgate,
    "R": Rgate,
    "XS": Sgate,
    "PS": PSgate,
    "X": Xgate,
    "Z": Zgate,
    "TMS": TMSgate,
    "MeasX": MeasX,
    "MeasP": MeasP
}

class Gaussian():
    """
    Class for continuous variable quantum compting in Gaussian formula.
    This class can only deal with gaussian states and gaussian gate.
    """    
    def __init__(self, N):
        self.N = N
        self.V = (np.eye(2 * N)) * 0.5
        self.mu = np.zeros(2 * N)
        self.ops = []
        self.creg = [[None, None] for i in range(self.N)] # [x, p]
    
    def  __getattr__(self, name):
        if name in GATE_SET:
            self.ops.append(GATE_SET[name])
            return self._setGateParam
        else:
            raise AttributeError('The state method does not exist')

    def _setGateParam(self, *args, **kwargs):
        self.ops[-1] = self.ops[-1](self, *args, **kwargs)
        return self

    def Creg(self, idx, var, scale = 1):
        return CregReader(self.creg, idx, var, scale)
    
    def run(self):
        """
        Run the circuit.
        """
        for gate in self.ops:
            [self.mu, self.V] = gate.run(state = [self.mu, self.V])
        return self

    def mean(self, idx):
        res = np.copy(self.mu[2 * idx:2 * idx + 2])
        return res

    def cov(self, idx):
        res = np.copy(self.V[(2 * idx):(2 * idx + 2), (2 * idx):(2 * idx + 2)])
        return res

    def Wigner(self, idx, plot = 'y', xrange = 5.0, prange = 5.0):
        """
        Calculate the Wigner function of a selected mode.
        
        Args:
            mode (int): Selecting a optical mode
            plot: If 'y', the plot of wigner function is output using matplotlib. If 'n', only the meshed values are returned
            x(p)range: The range in phase space for calculateing Wigner function
            
        """
        idx = idx * 2
        x = np.arange(-xrange, xrange, 0.1)
        p = np.arange(-prange, prange, 0.1)
        m = len(x)
        xx, pp = np.meshgrid(x, p)
        xi_array = np.dstack((pp, xx))
        W = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                W[i][j] = GaussianWigner(xi_array[j][i], self.V[idx:idx+2, idx:idx+2], self.mu[idx:idx+2])
        if plot == 'y':
            h = plt.contourf(x, p, W)
            plt.show()
        return (x, p, W)

    def PhotonDetectionProb(self, m, n):
        """
        Calculate Fock density matrix element rho_{mn}.
        m and n should be numpy array which length is same as mode number.
        When m = n, the returned value is probability for photon number m is measured.
        For example, if m = n = np.array([0, 0]), returned value is probability \
            for detecting 0 photon for both two modes.
        """
        if len(m) != self.N or len(n) != self.N:
            raise ValueError("Input array dimension must be same as mode Number.")
        return np.real(FockDensityMatrix(self.V, self.mu, m, n))

    def GaussianToFock(self, cutoffDim = 10):
        photonNumList = []
        cutoffDim += 1
        rho = np.empty([cutoffDim ** self.N, cutoffDim ** self.N])
        for i in range(cutoffDim ** self.N):
            photonNum = []
            for j in range(self.N):
                photonNum.insert(0, np.int(np.floor(i / (cutoffDim ** j))) % cutoffDim)
            photonNumList.append(photonNum)

        for i in range(cutoffDim ** self.N):
            for j in range(cutoffDim ** self.N):
                m = np.array(photonNumList[i])
                n = np.array(photonNumList[j])
                row = [m[i] ** (self.N - i - 1) for i in range(self.N)]
                col = [n[i] ** (self.N - i - 1) for i in range(self.N)]
                rho[row, col] = FockDensityMatrix(self.V, self.mu, m, n)

        return rho

    def Interferometer(self, U):
        num = self.N
        n = U.shape[0]
        BSang = [0]*(int(n*(n-1)/2))
        rot = [0]*(int(n*(n-1)/2))
        for i in range (1, n):
            if ((i+2)%2==1):
                for j in range (i): 
                    T = np.identity(n, dtype = complex)
                    if U[n-j-1][i-j-1]==0:
                        theta = 0
                        alpha = 0
                    elif U[n-j-1][i-j]==0:
                        theta = 0
                        alpha = np.pi/2
                    else:
                        theta = np.angle(U[n-j-1][i-1-j])-np.angle(U[n-j-1][i-j])
                        alpha = np.arctan(np.absolute(U[n-j-1][i-1-j])/np.absolute(U[n-j-1][i-j]))
                    BSang[int(i/2)+j] = alpha
                    rot[int(i/2)+j] = theta
                    e = np.cos(-theta) + np.sin(-theta)*1j
                    T[i-1-j][i-1-j] = e*(np.cos(alpha)+0*1j)
                    T[i-1-j][i-j] = -np.sin(alpha)+0*1j
                    T[i-j][i-1-j] = e*(np.sin(alpha)+0*1j)
                    T[i-j][i-j] = np.cos(alpha)+0*1j
                    U = U @ np.transpose(T)
                else:
                    for j in range (i):
                        T = np.identity(n, dtype = complex)
                        if U[n-i+j][j]==0:
                            theta = 0
                            alpha = 0
                        elif U[n-i+j-1][j]==0:
                            theta = 0
                            alpha = np.pi/2
                        else:
                            theta = -np.angle(U[n-i+j-1][j])+np.angle(U[n-i+j][j])
                            alpha = np.arctan(-(np.absolute(U[n-i+j][j])/np.absolute(U[n-i+j-1][j])))
                        BSang[-(i-1+j)] = alpha
                        rot[-(i-1+j)] = theta
                        e = np.cos(theta) + np.sin(theta)*1j
                        T[n-i+j-1][n-i+j-1] = e*(np.cos(alpha)+0*1j)
                        T[n-i+j-1][n-i+j] = -np.sin(alpha)+0*1j
                        T[n-i+j][n-i+j-1] = e*(np.sin(alpha)+0*1j)
                        T[n-i+j][n-i+j] = np.cos(alpha)+0*1j
                        U = T @ U

        counter = 0
        for i in range(1, n, 2):
            for k in range(i):
                self.ops.append(GATE_SET['R'])
                self.ops[-1] = self.ops[-1](self, i-1-k, rot[counter])
                self.ops.append(GATE_SET['BS'])
                self.ops[-1] = self.ops[-1](self, i-1-k, i-1-k+1, BSang[counter])
                counter += 1
        for i in reversed(range(2, n-2 if (n-1)%2==0 else n-1, 2)):
            for k in range(i):
                self.ops.append(GATE_SET['R'])
                self.ops[-1] = self.ops[-1](self, n-2-k, rot[counter])
                self.ops.append(GATE_SET['BS'])
                self.ops[-1] = self.ops[-1](self, n-2-k, n-1-k, BSang[counter])
                counter += 1

