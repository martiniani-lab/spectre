import numpy as np
import torch
from torch.func import jacrev
import sympy as sp
import os
import timeit
from functools import lru_cache

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_default_dtype(torch.float64)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# n_cores = 1


class recursive_solution:
    def __init__(self, J=None, L=None, S=None):
        """
        In this constructor function, we define and assign the different matrices "O"
        and list "l", upon which our spectrum solution depends.
        :param J: the Jacobian matrix
        :param L: the matrix containing noise coefficients
        :param S: the matrix containing the variance of the noise terms added
        """
        self.J = sp.Matrix([[sp.Rational(str(j)) for j in i] for i in J.tolist()])
        self.L = sp.Matrix([[sp.Rational(str(j)) for j in i] for i in L.tolist()])
        self.S = sp.Matrix([[sp.Rational(str(j)) for j in i] for i in S.tolist()])
        self.D = self.S ** 2
        self.C = self.L * self.D * self.L.T
        self.n = J.shape[0]

        """Define the solution lists"""
        self.P = [sp.zeros(self.n, self.n) for _ in range(self.n + 1)]
        self.P_prime = [sp.zeros(self.n, self.n) for _ in range(self.n)]
        self.Q = [sp.zeros(self.n, self.n) for _ in range(self.n + 1)]
        self.Q_prime = [sp.zeros(self.n, self.n) for _ in range(self.n)]
        self.q = [sp.Rational('1') for _ in range(self.n + 1)]

        """Find the solution matrices"""
        self.find_solution_recursive()

    def find_solution_recursive(self):
        """
        This function finds the solution of the coefficient matrices of the numerator
        and the coefficients of the denominator recursively.
        :return: None
        """
        self.P[self.n-1] = self.q[self.n] * self.C
        self.Q[self.n-1] = sp.eye(self.n)
        for alpha in range(self.n - 1, 0, -1):
            self.P_prime[alpha - 1] = recursive_solution.P_prime(self.J, self.P[alpha], self.P_prime[alpha])
            self.Q_prime[alpha - 1] = recursive_solution.P_prime(self.J, self.Q[alpha], self.Q_prime[alpha])
            self.q[alpha] = recursive_solution.q(self.J, self.Q_prime[alpha - 1], self.Q[alpha], self.n, alpha)
            self.P[alpha - 1] = recursive_solution.P(self.q[alpha], self.C, self.P_prime[alpha - 1], self.J, self.P[alpha])
            self.Q[alpha - 1] = recursive_solution.P(self.q[alpha], sp.eye(self.n), self.Q_prime[alpha - 1], self.J, self.Q[alpha])
        self.q[0] = recursive_solution.q(self.J, self.Q_prime[-1], self.Q[0], self.n, 0)
        return None

    @staticmethod
    def P(q, C, P_prime, J, P):
        """
        This function returns P_{alpha-1} using the recursive solution.
        P_{alpha-1} = q_alpha * C + P_prime_{alpha-1} * J^T - J P_prime_{alpha-1} - J * P_alpha * J^T
        """
        return q * C + P_prime * J.T - J * P_prime - J * P * J.T

    @staticmethod
    def P_prime(J, P, P_prime):
        """
        This function returns P_prime_{alpha-1} using the recursive solution.
        P_prime_{alpha-1} = J * P_alpha - P_alpha * J^T - J * P_prime_{alpha} * J^T
        """
        return J * P - P * J.T - J * P_prime * J.T

    @staticmethod
    def q(J, Q_prime, Q, n, alpha):
        """
        This function returns q_{alpha} using the recursive solution.
        q_{alpha} = (Tr(J * Q_prime_{alpha-1}) + Tr(J * Q_alpha * J^T)) / (n-alpha)
        """
        return ((J * Q_prime).trace() + (J * Q * J.T).trace()) / (n - alpha)

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

    def p_cross_r_all_coeffs(self, idx1, idx2):
        """
        This function returns all the coefficients of the real part of the numerator
        of cross-spectrum.
        :param idx1: the index of the 1st variable for which the auto-spectrum is desired.
        :param idx2: the index of the 2nd variable for which the auto-spectrum is desired.
        :return: the coefficients of the real part of the numerator of the cross-spectrum.
        """
        return [self.P[i][idx1, idx2] for i in range(self.n)]

    def p_cross_i_all_coeffs(self, idx1, idx2):
        """
        This function returns all the coefficients of the imag part of the numerator
        of cross-spectrum.
        :param idx1: the index of the 1st variable for which the auto-spectrum is desired.
        :param idx2: the index of the 2nd variable for which the auto-spectrum is desired.
        :return: the coefficients of the imag part of the numerator of the cross-spectrum.
        """
        return [self.P_prime[i][idx1, idx2] for i in range(self.n-1)]

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
        frequency = torch.logspace(np.log10(1), np.log10(1000), 100) if frequency is None else frequency

        # Convert the frequency from Hz to omega
        freq = sp.Matrix([sp.Rational(str(i)) for i in frequency.tolist()])
        om = 2 * sp.pi * freq
        n = self.n

        # Denominator
        q_all = self.q_all_coeffs()
        powers = 2 * torch.arange(n+1)
        denm = torch.zeros(om.shape[0])
        for j in range(om.shape[0]):
            for m in range(n+1):
                denm[j] += float(q_all[m] * om[j] ** powers[m])

        # Numerator
        p_all = self.p_auto_all_coeffs(idx)
        powers = 2 * torch.arange(n)
        num = torch.zeros(om.shape[0])
        for j in range(om.shape[0]):
            for m in range(n):
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
        frequency = torch.logspace(np.log10(1), np.log10(1000), 100) if frequency is None else frequency
        # Convert the frequency from Hz to omega
        freq = sp.Matrix([sp.Rational(str(i)) for i in frequency.round(decimals=5).tolist()])
        om = 2 * sp.pi * freq
        n = self.n

        # Denominator
        q_all = self.q_all_coeffs()
        powers = 2 * torch.arange(n+1)
        denm = torch.zeros(om.shape[0])
        for j in range(om.shape[0]):
            for m in range(n+1):
                denm[j] += float(q_all[m] * om[j] ** powers[m])

        # Numerator (real part)
        p_r_all = self.p_cross_r_all_coeffs(idx1, idx2)
        powers = 2 * torch.arange(n)
        num_r = torch.zeros(om.shape[0])
        for k in range(om.shape[0]):
            for m in range(n):
                num_r[k] += float(p_r_all[m] * om[k] ** powers[m])

        # Numerator (Imaginary part)
        p_i_all = self.p_cross_i_all_coeffs(idx1, idx2)
        powers = 2 * torch.arange(n-1) + 1
        num_i = torch.zeros(om.shape[0])
        for k in range(om.shape[0]):
            for m in range(n-1):
                num_i[k] += float(p_i_all[m] * om[k] ** powers[m])

        num = num_r + 1j * num_i

        Sxy = num / denm
        return Sxy, frequency

    def coherence(self, idx1, idx2, frequency=None):
        """
        This function returns the coherence-squared (off-diagonal elements) of the
        spectral density matrix.
        :param frequency: Torch tensor containing the frequencies for which the spectra
         are desired.
        :param idx1, idx2: the index of the variables between which the coherence is
        desired. Note, here we follow 0 indexing.
        :return: the coherence: |Sxy|^2 / (Sxx * Syy), between the given variables.
        """
        frequency = torch.logspace(np.log10(1), np.log10(1000), 100) if frequency is None else frequency
        # Convert the frequency from Hz to omega
        freq = sp.Matrix([sp.Rational(str(i)) for i in frequency.round(decimals=5).tolist()])
        om = 2 * sp.pi * freq
        n = self.n

        # Cross-spectrum (numerator of coherence)
        p_r_all = self.p_cross_r_all_coeffs(idx1, idx2)
        powers = 2 * torch.arange(n)
        num_r = torch.zeros(om.shape[0])
        for k in range(om.shape[0]):
            for m in range(n):
                num_r[k] += float(p_r_all[m] * om[k] ** powers[m])

        p_i_all = self.p_cross_i_all_coeffs(idx1, idx2)
        powers = 2 * torch.arange(n-1) + 1
        num_i = torch.zeros(om.shape[0])
        for k in range(om.shape[0]):
            for m in range(n-1):
                num_i[k] += float(p_i_all[m] * om[k] ** powers[m])

        num = num_r**2 + num_i**2

        # Auto-spectrum (denominator of coherence)
        p_all1 = self.p_auto_all_coeffs(idx1)
        powers = 2 * torch.arange(n)
        denm1 = torch.zeros(om.shape[0])
        for j in range(om.shape[0]):
            for m in range(n):
                denm1[j] += float(p_all1[m] * om[j] ** powers[m])

        p_all2 = self.p_auto_all_coeffs(idx2)
        powers = 2 * torch.arange(n)
        denm2 = torch.zeros(om.shape[0])
        for j in range(om.shape[0]):
            for m in range(n):
                denm2[j] += float(p_all2[m] * om[j] ** powers[m])

        denm = denm1 * denm2

        coh = num / denm
        return coh, frequency


if __name__ == '__main__':
    pass
