"""This function calculates the power spectral density using the matrix solution"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import mpmath
from torch.func import jacrev
import sympy as sp
import os
import timeit
from functools import lru_cache

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_default_dtype(torch.float64)


class matrix_solution:
    def __init__(self, J=None, L=None, S=None):
        """
        This function initializes the matrices required for the matrix solution.
        :param J: Jacobian matrix
        :param L: Noise coefficient matrix
        :param S: Diagonal matrix containing the standard deviation of the Wiener increments.
        """
        self.N = None
        self.J = J
        self.L = L
        self.S = S
        self.D = S**2
        self.noise_mat = (self.L @ self.D @ self.L.T).type(torch.cdouble)

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, J):
        if torch.any(torch.real(torch.linalg.eigvals(J)) > 0):
            raise ValueError("The eigenvalues of the Jacobian matrix are not less than 0")
        self._J = J
        self.N = J.shape[0]

    def spectral_matrix(self, freq=None, J=None):
        """
        This function calculates the power spectral density matrix of the system
        for different frequencies.
        :param freq: frequency tensor
        :return: the PSD matrix
        """
        freq = torch.logspace(np.log10(1), np.log10(1000), 100) if freq is None else freq
        if J is None:
            J = self.J
        if J is None:
            raise ValueError("Jacobian matrix is not defined")

        om = (2 * np.pi * freq).to(device)
        m = freq.size(0)

        n = self.N
        S = torch.zeros((m, n, n), dtype=torch.cdouble, device=device)

        with torch.no_grad():
            for i in range(m):
                S[i] = torch.inverse(
                    J + 1j * om[i] * torch.eye(n, device=device)) @ self.noise_mat @ torch.inverse(
                    torch.transpose(J - 1j * om[i] * torch.eye(n, device=device), 0, 1))
        return S

    def auto_spectrum(self, i=None, freq=None, J=None):
        """
        This function calculates the power spectral density of a given variable (index i)
        using the matrix solution.
        :param freq: frequency over which the spectrum is desired.
        :param i: The index of the variable for which the spectrum is desired.
        :return: the Power Spectral Density.
        """
        i = self.N // 2 if i is None else i
        freq = torch.logspace(np.log10(0.001), np.log10(1000), 100) if freq is None else freq

        S = self.spectral_matrix(freq, J)
        m = freq.size(0)

        Sxx = torch.zeros(m, device=device)
        for j in range(m):
            Sxx[j] = torch.squeeze(torch.real(S[j, i, i]))
        return Sxx.cpu(), freq

    def cross_spectrum(self, i=None, j=None, freq=None, J=None):
        """
        This function calculates the cross power spectral density between variables
        with the indices i and j using the matrix solution.
        :param freq: frequency over which the spectrum is desired.
        :param i: The index of the variable between which the spectrum is desired.
        :param j: The index of the variable between which the spectrum is desired.
        :return: the cross Power Spectral Density.
        """
        i = self.N // 2 if i is None else i
        j = self.N // 2 + 1 if j is None else j
        freq = torch.logspace(np.log10(0.001), np.log10(1000), 100) if freq is None else freq

        S = self.spectral_matrix(freq, J)
        m = freq.size(0)

        Sxy = torch.zeros(m, device=device, dtype=torch.cdouble)
        for k in range(m):
            Sxy[k] = torch.squeeze(S[k, i, j])
        return Sxy.cpu(), freq

    def coherence(self, i=None, j=None, freq=None, J=None):
        """
        This function calculates the squared-coherence between variables
        with the indices i and j using the matrix solution.
        :param freq: frequency over which the coherence is desired.
        :param i: The index of the variable between which the coherence is desired.
        :param j: The index of the variable between which the coherence is desired.
        :return: coherence: |Sxy|^2 / (Sxx * Syy).
        """
        i = self.N // 2 if i is None else i
        j = self.N // 2 + 1 if j is None else j
        freq = torch.logspace(np.log10(0.001), np.log10(1000), 100) if freq is None else freq

        S = self.spectral_matrix(freq, J)
        m = freq.size(0)

        coh = torch.zeros(m, device=device, dtype=torch.cdouble)
        for k in range(m):
            coh[k] = torch.squeeze(torch.abs(S[k, i, j])**2 / (torch.real(S[k, i, i]) * torch.real(S[k, j, j])))
        return coh.cpu(), freq


if __name__ == '__main__':
    pass
