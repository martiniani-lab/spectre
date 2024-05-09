"""This function calculates the power spectral density using the matrix solution"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.linalg
import control
import math
import mpmath
import sympy as sp
import os
import timeit
from functools import lru_cache


class matrix_solution:
    def __init__(
        self, J=None, L=None, S=None, device=None, is_spd=False, is_spd_diagonal=False
    ):
        """
        This function initializes the matrices required for the matrix solution.
        :param J: Jacobian matrix
        :param L: Noise coefficient matrix
        :param S: Diagonal matrix containing the standard deviation of the Wiener increments.
        :param is_spd: If True, the noise matrix is positive definite (positive semi-definite otherwise).
        :param is_spd_diagonal: If True, the noise matrix is diagonal and positive definite (inversion straightforward).
        """

        """Device to run the simulation on"""
        self.device = device if device is not None else torch.device("cpu")

        self.N = None
        self.J = J.to(self.device)
        self.is_spd = is_spd
        self.is_spd_diagonal = is_spd_diagonal

        self.spectral_mat = {}

        """Inverse of the noise covariance matrix"""
        if self.is_spd_diagonal:
            ls = torch.diag(L) * torch.diag(S)
            self.noise_mat = torch.diag(ls**2).to(self.device)
            self.A2 = torch.diag(1 / ls**2).to(self.device)
        elif self.is_spd:
            LS = L @ S
            self.noise_mat = (LS @ LS.T).to(self.device)
            self.A2 = torch.cholesky_inverse(torch.linalg.cholesky(self.noise_mat)).to(
                self.device
            )
        else:
            LS = L @ S
            self.noise_mat = (LS @ LS.T).to(self.device)
            self.A2 = None

    @staticmethod
    def cholesky_inverse(J, A2, omega):
        """
        This function returns the inverse of a complex matrix mat by using the Cholesky decomposition.
        :param J: The Jacobian matrix.
        :param A2: The inverse of the noise covariance matrix.
        :param omega: The frequency.
        :return: The spectral density matrix.
        """
        A1 = J.T @ A2 @ J
        A3 = J.T @ A2 - A2 @ J
        mat = A1 + (omega**2) * A2 + 1j * omega * A3
        return torch.cholesky_inverse(torch.linalg.cholesky(mat))

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, J):
        if torch.any(torch.real(torch.linalg.eigvals(J)) > 0):
            raise ValueError(
                "The eigenvalues of the Jacobian matrix are not less than 0"
            )
        self._J = J
        self.N = J.shape[0]

    def spectral_matrix(self, freq=None, J=None):
        """
        This function calculates the power spectral density matrix of the system
        for different frequencies.
        :param freq: frequency tensor
        :param J: Jacobian matrix
        :return: a list of S(omega) matrices.
        """
        freq = (
            torch.logspace(np.log10(1), np.log10(1000), 100) if freq is None else freq
        )
        if J is None:
            J = self.J
        if J is None:
            raise ValueError("Jacobian matrix is not defined")

        om = (2 * np.pi * freq).to(self.device)
        m = freq.size(0)

        n = self.N

        S = []

        with torch.no_grad():
            for i in range(m):
                if self.is_spd or self.is_spd_diagonal:
                    S.append(self.cholesky_inverse(J, self.A2, om[i]))
                else:
                    Left_inv = torch.linalg.inv(
                        J + 1j * om[i] * torch.eye(n, device=self.device)
                    )
                    S.append(
                        Left_inv
                        @ self.noise_mat.to(Left_inv.dtype)
                        @ torch.conj(Left_inv).T
                    )
        self.spectral_mat[tuple(freq.tolist())] = S
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
        freq = (
            torch.logspace(np.log10(0.001), np.log10(1000), 100)
            if freq is None
            else freq
        )
        J = self.J if J is None else J

        if tuple(freq.tolist()) in self.spectral_mat:
            S = self.spectral_mat[tuple(freq.tolist())]
        else:
            S = self.spectral_matrix(freq, J)

        m = freq.size(0)

        Sxx = torch.zeros(m, device=self.device)
        for j in range(m):
            Sxx[j] = torch.squeeze(torch.real(S[j][i, i]))
        return Sxx, freq

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
        freq = (
            torch.logspace(np.log10(0.001), np.log10(1000), 100)
            if freq is None
            else freq
        )
        J = self.J if J is None else J

        if tuple(freq.tolist()) in self.spectral_mat:
            S = self.spectral_mat[tuple(freq.tolist())]
        else:
            S = self.spectral_matrix(freq, J)

        m = freq.size(0)

        Sxy = torch.zeros(m, device=self.device, dtype=torch.cdouble)
        for k in range(m):
            Sxy[k] = torch.squeeze(S[k][i, j])
        return Sxy, freq

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
        freq = (
            torch.logspace(np.log10(0.001), np.log10(1000), 100)
            if freq is None
            else freq
        )

        if tuple(freq.tolist()) in self.spectral_mat:
            S = self.spectral_mat[tuple(freq.tolist())]
        else:
            S = self.spectral_matrix(freq, J)

        m = freq.size(0)

        coh = torch.zeros(m, device=self.device)
        for k in range(m):
            coh[k] = torch.squeeze(
                torch.abs(S[k][i, j]) ** 2
                / (torch.real(S[k][i, i]) * torch.real(S[k][j, j]))
            )
        return coh, freq

    def S0(self, J=None):
        """
        This function returns the spectral density matrix at zero-frequency.
        This value is supposedly related to the Fano-factor of the variables.
        """
        if J is None:
            J = self.J
        with torch.no_grad():
            S = (
                torch.inverse(J)
                @ self.noise_mat
                @ torch.inverse(torch.transpose(J, 0, 1))
            )
        return S

    def lyap_solution(self, J=None, verify=True, tol=1e-8):
        """
        This function finds the steady-state Covariance matrix associated with the
        system by solving the lyapunov equation (J @ P + P @ J.T + L @ D @ L.T = 0).
        Note that just using the scipy function isn't good because it doesn't give the
        right solution (numerically unstable).
        :return: the solution of the Lyapunov equation.
        """
        if J is None:
            J = self.J
        P = control.lyap(J.numpy(), self.noise_mat.numpy())
        P = torch.from_numpy(P)
        if verify:
            assert self.lyap_verify(P=P) < tol
        return P

    def lyap_verify(self, P=None, J=None):
        """
        This function verifies the solution of the Lyapunov equation.
        J @ P + P @ J.T + L @ D @ L.T = 0
        :return: the residual of the Lyapunov equation.
        """
        if J is None:
            J = self.J
        if P is None:
            P = self.lyap_solution(J, verify=False)
        return torch.norm(J @ P + P @ J.T + self.noise_mat)


if __name__ == "__main__":
    pass
