"""This function calculates the power spectral density using the recursive solution"""

import numpy as np
import torch
from torch.func import jacrev
import sympy as sp
import os
import timeit
from functools import lru_cache


class recursive_solution_g:
    def __init__(self, G=None, L=None, Y=None):
        """
        In this constructor function, we define and assign the different matrices "O"
        and list "l", upon which our spectrum solution depends.
        :param J: the Jacobian matrix
        :param L: the matrix containing noise coefficients
        :param S: the rate matrix
        """
        self.G = sp.Matrix([[sp.Rational(str(j)) for j in i] for i in G.tolist()])
        self.L = sp.Matrix([[sp.Rational(str(j)) for j in i] for i in L.tolist()])
        self.S = sp.Matrix([[sp.Rational(str(j)) for j in i] for i in Y.tolist()])
        self.Y = self.L * self.S * self.L.T
        self.Y_inv = self.Y.inv()
        self.n = G.shape[0]

        """Define the solution lists"""
        self.P = [sp.zeros(self.n, self.n) for _ in range(2 * self.n + 1)]
        self.q = [sp.Rational("1") for _ in range(2 * self.n + 1)]

        """Find the solution matrices"""
        self.find_solution_recursive()

    def find_solution_recursive(self):
        """
        This function finds the solution of the coefficient matrices of the numerator
        and the coefficients of the denominator recursively.
        :return: None
        """
        for alpha in range(2 * self.n - 1, 0, -1):
            self.P[alpha - 1] = recursive_solution_g.P(
                self.q[alpha + 1], self.Y, self.G, self.P[alpha], self.P[alpha+1]
            )
            self.q[alpha] = recursive_solution_g.q(
                self.G, self.n, alpha, self.Y_inv, self.P[alpha], self.P[alpha-1]
            )
        self.q[0] = recursive_solution_g.q(self.G, self.n, 0, self.Y_inv, self.P[0], self.P[-1])
        return None

    @staticmethod
    def P(q, Y, G, P, P1):
        """
        This function returns P_{alpha-1} using the recursive solution.
        P_{alpha-1} = q_{alpha+1} * Y + G * P_alpha + P_alpha * G^T - G * P_{alpha+1} * G^T
        """
        return q * Y + G * P + P * G.T - G * P1 * G.T

    @staticmethod
    def q(G, n, alpha, Y_inv, P, P1):
        """
        This function returns q_{alpha} using the recursive solution.
        q_{alpha} = (Tr(Y^-1 * G * P_alpha * G^T) - Tr(Y^-1 * G * P_{alpha - 1} )) / (n-alpha/2)
        """
        return ((Y_inv * G * P * G.T).trace() - (Y_inv * G * P1).trace()) / (n - alpha/2)

    def q_all_coeffs(self):
        """
        Returns the list of all the coefficients of the denominator.
        """
        return self.q

    def p_auto_all_coeffs(self, idx):
        """
        This function returns all the coefficients of the numerator of the auto-spectrum.
        :param idx: the index of the variable for which the auto-spectrum is desired.
        :return: the list of coefficients of the numerator of the auto-spectrum.
        """
        return [self.P[i][idx, idx] for i in range(self.n)]

    def p_cross_all_coeffs(self, idx1, idx2):
        """
        This function returns all the coefficients of the real part of the numerator
        of cross-spectrum.
        :param idx1: the index of the 1st variable for which the auto-spectrum is desired.
        :param idx2: the index of the 2nd variable for which the auto-spectrum is desired.
        :return: the coefficients of the real part of the numerator of the cross-spectrum.
        """
        return [self.P[i][idx1, idx2] for i in range(self.n)]

    def auto_spectrum(self, idx, frequency=None):
        """
        This function returns the auto-spectrum (diagonal elements) of the spectral
        density matrix.
        :param frequency: Torch tensor containing the frequencies for which the spectra
         are desired.
        :param idx: the index of the variable for which the auto-spectra is desired.
         Note, here we follow 0 indexing.
        :return: the auto-PSD for the given variable.
        """
        frequency = (
            torch.logspace(np.log10(1), np.log10(1000), 100)
            if frequency is None
            else frequency
        )

        # Convert the frequency from Hz to omega
        freq = sp.Matrix([sp.Rational(str(i)) for i in frequency.tolist()])
        om = 2 * sp.pi * freq
        n = self.n

        # Denominator
        q_all = self.q_all_coeffs()
        powers = torch.arange(2 * n + 1)
        denm = torch.zeros(om.shape[0], dtype=torch.float64)
        for j in range(om.shape[0]):
            for m in range(2 * n + 1):
                denm[j] += float(q_all[m] * om[j] ** powers[m])

        # Numerator
        p_all = self.p_auto_all_coeffs(idx)
        powers = torch.arange(2 * n - 1)
        num = torch.zeros(om.shape[0], dtype=torch.float64)
        for j in range(om.shape[0]):
            for m in range(2 * n - 1):
                num[j] += float(p_all[m] * om[j] ** powers[m])

        Sxx = num / denm
        return Sxx, frequency

    def cross_spectrum(self, idx1, idx2, frequency=None):
        """
        This function returns the cross-spectrum (off-diagonal elements) of the spectral
        density matrix.
        :param frequency: Torch tensor containing the frequencies for which the spectra
         are desired.
        :param idx1, idx2: the indices of the variables between which the cross-spectra
        is desired. Note, here we follow 0 indexing.
        :return: the cross-PSD for the given variables.
        """
        frequency = (
            torch.logspace(np.log10(1), np.log10(1000), 100)
            if frequency is None
            else frequency
        )
        # Convert the frequency from Hz to omega
        freq = sp.Matrix(
            [sp.Rational(str(i)) for i in frequency.round(decimals=5).tolist()]
        )
        om = 2 * sp.pi * freq
        n = self.n

        # Denominator
        q_all = self.q_all_coeffs()
        powers = torch.arange(2 * n + 1)
        denm = torch.zeros(om.shape[0], dtype=torch.float64)
        for j in range(om.shape[0]):
            for m in range(2 * n + 1):
                denm[j] += float(q_all[m] * om[j] ** powers[m])

        # Numerator
        p_all = self.p_cross_all_coeffs(idx1, idx2)
        powers = torch.arange(2 * n - 1)
        num = torch.zeros(om.shape[0], dtype=torch.float64)
        for k in range(om.shape[0]):
            for m in range(2 * n - 1):
                num[k] += float(p_all[m] * om[k] ** powers[m])

        Sxy = num / denm
        return Sxy, frequency


if __name__ == "__main__":
    pass
