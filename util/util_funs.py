import torch


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