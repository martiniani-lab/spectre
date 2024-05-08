"""This function calculates the power spectral density coefficients using the recursive solution in numpy"""

import numpy as np
import torch
from torch.func import jacrev
import os
import timeit
from functools import lru_cache


class recursive_solution_np:
    def __init__(self, J=None, L=None, S=None):
        """
        In this constructor function, we define and assign the different matrices "O"
        and list "l", upon which our spectrum solution depends.
        :param J: the Jacobian matrix
        :param L: the matrix containing noise coefficients
        :param S: the matrix containing the variance of the noise terms added
        """
        self.J = J.numpy()
        self.L = L.numpy()
        self.S = S.numpy()
        self.D = self.S @ self.S
        self.C = self.L @ self.D @ self.L.T
        self.n = J.shape[0]

        """Define the solution lists"""
        self.P = [np.zeros((self.n, self.n)) for _ in range(self.n + 1)]
        self.P_prime = [np.zeros((self.n, self.n)) for _ in range(self.n)]
        self.Q = [np.zeros((self.n, self.n)) for _ in range(self.n + 1)]
        self.Q_prime = [np.zeros((self.n, self.n)) for _ in range(self.n)]
        self.q = [np.array([1]) for _ in range(self.n + 1)]

        """Find the solution matrices"""
        self.find_solution_recursive()

    def find_solution_recursive(self):
        """
        This function finds the solution of the coefficient matrices of the numerator
        and the coefficients of the denominator recursively.
        :return: None
        """
        self.P[self.n - 1] = self.q[self.n] * self.C
        self.Q[self.n - 1] = np.eye(self.n)
        for alpha in range(self.n - 1, 0, -1):
            self.P_prime[alpha - 1] = recursive_solution_np.P_prime(
                self.J, self.P[alpha], self.P_prime[alpha]
            )
            self.Q_prime[alpha - 1] = recursive_solution_np.P_prime(
                self.J, self.Q[alpha], self.Q_prime[alpha]
            )
            self.q[alpha] = recursive_solution_np.q(
                self.J, self.Q_prime[alpha - 1], self.Q[alpha], self.n, alpha
            )
            self.P[alpha - 1] = recursive_solution_np.P(
                self.q[alpha], self.C, self.P_prime[alpha - 1], self.J, self.P[alpha]
            )
            self.Q[alpha - 1] = recursive_solution_np.P(
                self.q[alpha],
                np.eye(self.n),
                self.Q_prime[alpha - 1],
                self.J,
                self.Q[alpha],
            )
        self.q[0] = recursive_solution_np.q(
            self.J, self.Q_prime[-1], self.Q[0], self.n, 0
        )
        return None

    @staticmethod
    def P(q, C, P_prime, J, P):
        """
        This function returns P_{alpha-1} using the recursive solution.
        P_{alpha-1} = q_alpha * C + P_prime_{alpha-1} * J^T - J P_prime_{alpha-1} - J * P_alpha * J^T
        """
        return q * C + P_prime @ J.T - J @ P_prime - J @ P @ J.T

    @staticmethod
    def P_prime(J, P, P_prime):
        """
        This function returns P_prime_{alpha-1} using the recursive solution.
        P_prime_{alpha-1} = J * P_alpha - P_alpha * J^T - J * P_prime_{alpha} * J^T
        """
        return J @ P - P @ J.T - J @ P_prime @ J.T

    @staticmethod
    def q(J, Q_prime, Q, n, alpha):
        """
        This function returns q_{alpha} using the recursive solution.
        q_{alpha} = (Tr(J * Q_prime_{alpha-1}) + Tr(J * Q_alpha * J^T)) / (n-alpha)
        """
        return (np.trace((J @ Q_prime)) + np.trace((J @ Q @ J.T))) / (n - alpha)

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
        return [self.P_prime[i][idx1, idx2] for i in range(self.n - 1)]


if __name__ == "__main__":
    pass
