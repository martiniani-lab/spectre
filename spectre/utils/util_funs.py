import torch
import scipy
import numpy as np
from scipy.linalg import ldl


def dynm_fun(f):
    """A wrapper for the dynamical function"""

    def wrapper(self, t, x):
        new_fun = lambda t, x: f(self, t, x)
        return new_fun(t, x)

    return wrapper


def generate_stable_system(n):
    """
    This function returns a randomly generated stable linear dynamical system (real part of the eigenvalues are negative)
    :param n: dimension of the system
    :return:
    """
    A = torch.randn(n, n)
    L, V = torch.linalg.eig(A)
    L = - torch.abs(torch.real(L)) + 1j*torch.imag(L)
    return torch.real(V @ torch.diag(L) @ torch.linalg.inv(V))


def cholesky_decomposition(A):
    """
    This function returns the cholesky decomposition of a matrix A.
    :param A: The matrix to be decomposed.
    :return: The cholesky decomposition of A.
    """
    L = torch.linalg.cholesky(A)
    return L

def make_square(L, S):
    """
    This function returns a square system from a rectanguar system.
    :param L: The noise coefficient matrix (a torch tensor).
    :param S: The square-root diagonal matrix (a torch tensor).
    :return: square_L, square_S.
    """
    C = L @ S @ S.T @ L.T
    n = L.shape[0]
    m = L.shape[1]
    if n > m:
        square_L = torch.cat((L, torch.zeros(n, n-m)), dim=1)
        square_S = torch.cat((S, torch.zeros(m, n-m)), dim=1)
        square_S = torch.cat((square_S, torch.zeros(n-m, n)), dim=0)
    elif n < m:
        # Calculate LDL decomposition
        lu, d, perm = ldl(C.numpy(), lower=0)
        square_L = torch.tensor(lu)
        square_S = torch.sqrt(torch.tensor(d))
    else:
        square_L = L
        square_S = S
    return square_L, square_S