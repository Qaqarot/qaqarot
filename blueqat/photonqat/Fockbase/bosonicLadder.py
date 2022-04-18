
import numpy as np
from scipy.special import factorial as fact
from scipy.linalg import expm
from sympy.physics.quantum import Operator
from sympy.parsing.sympy_parser import parse_expr
from sympy import Matrix, matrix2numpy, MatrixSymbol
from sympy import I as symI

def _downMat(dim, order):    
    if order == 0:
        A = np.eye(dim)
        return A
    else:
        A = np.zeros([dim, dim])
        for i in np.arange(order, dim):
            A[i, i - order] = np.prod(np.sqrt(np.arange(i, i - order, -1)))
        return A

def _upMat(dim, order):        
    if order == 0:
        A = np.eye(dim)
        return A
    else:
        A = np.zeros([dim, dim])
        for i in np.arange(0, dim - order):
            A[i, i + order] = np.prod(np.sqrt(np.arange(i + 1, i + 1 + order)))
        return A

def _downMatLeft(dim, order):    
    if order == 0:
        A = np.eye(dim)
        return A
    else:
        A = np.zeros([dim, dim])
        for i in np.arange(order, dim):
            A[i - order, i] = np.prod(np.sqrt(np.arange(i, i - order, -1)))
        return A

def _upMatLeft(dim, order):        
    if order == 0:
        A = np.eye(dim)
        return A
    else:
        A = np.zeros([dim, dim])
        for i in np.arange(0, dim - order):
            A[i - order, i] = np.prod(np.sqrt(np.arange(i + 1, i + 1 + order)))
        return A

def _nMat(dim, order):
    if order == 0:
        A = np.eye(dim)
        return A
    else:
        A = np.diag(np.arange(dim) ** order)
        return A

def exp_annihiration(fockState, alpha, order = 1, cutoff = 10):
    row = fockState.shape[0]
    mat = _downMat(fockState.shape[-1], order)
    mat_ = np.empty(mat.shape, dtype=np.complex)
    mat_ = expm(alpha * mat)
    res = np.dot(fockState, mat_)
    return res

def exp_creation(fockState, alpha, order = 1, cutoff = 10):
    row = fockState.shape[0]
    mat = _upMat(fockState.shape[-1], order)
    mat_ = np.empty(mat.shape, dtype=np.complex)
    mat_ = expm(alpha * mat)
    res = np.dot(fockState, mat_)
    return res

def exp_photonNum(fockState, alpha, order = 1, cutoff = 10):
    row = fockState.shape[0]
    mat = _nMat(fockState.shape[-1], order)
    mat_ = np.empty(mat.shape, dtype=np.complex)
    mat_ = expm(alpha * mat)
    res = np.dot(fockState, mat_)
    return res

def _mat_for_mode2(mat):
    l = mat.shape[0]
    mat_ = np.zeros(np.array(mat.shape)**2)
    for i in range(mat.shape[0]):
        mat_[i*l:i*l+l, i*l:i*l+l] = mat
    return mat_

def _mat_for_mode1(mat):
    l = mat.shape[0]
    mat_ = np.zeros(np.array(mat.shape)**2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            mat_[i*l:i*l+l, j*l:j*l+l] = np.eye(l) * mat[i, j]
    return mat_

def exp_BS(fockState, alpha, cutoff):
    state = np.zeros(fockState.shape) + 0j
    down = _downMat(cutoff + 1, 1)
    up = _upMat(cutoff + 1, 1)
    mat1_ = np.dot(_mat_for_mode1(up), _mat_for_mode2(down))
    mat2_ = np.dot(_mat_for_mode1(down), _mat_for_mode2(up))
    mat_ = mat1_ - mat2_
    emat_ = expm(alpha * mat_)
    res = np.dot(fockState, emat_)
    return res

def exp_AAaa(fockState, alpha, cutoff):
    mat = _downMat(fockState.shape[-1], 2)
    #mat = np.dot(_upMat(fockState.shape[-1], 2), mat)
    mat = np.dot(mat, _upMat(fockState.shape[-1], 2))
    mat_ = np.empty(mat.shape, dtype=np.complex)
    mat_ = expm(alpha * mat)
    res = np.dot(fockState, mat_)
    return res

def exp_aa_minus_AA(fockState, alpha, cutoff):
    mat_a = _downMat(fockState.shape[-1], 2) # aa
    mat_A = _upMat(fockState.shape[-1], 2) # AA
    mat_ = np.empty(mat_a.shape, dtype=np.complex)
    mat_ = expm(np.conj(alpha) * mat_a - alpha * mat_A)
    res = np.dot(fockState, mat_)
    return res

def expand_xp_to_aA(expr_string, dim, evaluate = True, hbar = 1):
    x = Operator('x')
    p = Operator('p')
    expr = parse_expr(expr_string, local_dict = {'x':x, 'p':p})
    a_ = Operator('a')
    A_ = Operator('A')
    expr_ = expr.subs([(x, (a_ + A_) * sym.sqrt(hbar / 2)), (p, (a_ - A_) * sym.sqrt(hbar / 2) / 1j)])
    expr_ = expr_.expand()
    aA = str(expr_)
    if evaluate == False:
        return aA

    a = sym.MatrixSymbol('a', dim, dim)
    A = sym.MatrixSymbol('A', dim, dim)
    down = Matrix(downMatLeft(dim, 1))
    up = Matrix(upMatLeft(dim, 1))
    expr__ = parse_expr(aA, local_dict = {'a':a, 'A':A, 'I':sym.I})
    expr__ = expr__.subs([(a, down), (A, up)])
    res = matrix2numpy(expr__, dtype = np.complex)
    return res

def exp_str_aA(expr_string, alpha, N = 5):
    a = sym.Symbol('a', commutative = False)
    A = sym.Symbol('A', commutative = False)
    E = sym.Symbol('E', commutative = False)
    expr = parse_expr(expr_string, local_dict = {'a':a, 'A':A})
    exp_expr = E
    for i in range(1, N + 1):
        exp_expr += alpha**i * (expr)**i / np.math.factorial(i)
    exp_aA = exp_expr.expand()
    return str(exp_aA)
    
def str_to_aA_mat(expr_string, dim, hbar = 1):
    a = sym.MatrixSymbol('a', dim, dim)
    A = sym.MatrixSymbol('A', dim, dim)
    down = Matrix(_downMatLeft(dim, 1))
    up = Matrix(_upMatLeft(dim, 1))
    expr_ = parse_expr(expr_string, local_dict = {'a':a, 'A':A, 'I':sym.I, 'E':sym.eye(dim)})
    expr_ = expr_.subs([(a, down), (A, up)])
    res = matrix2numpy(expr_, dtype = np.complex)
    return res