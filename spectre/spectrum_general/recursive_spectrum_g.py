import numpy as np
import torch
import sympy as sp
from sympy import Rational


class recursive_solution_g:
    def __init__(self, G=None, Y=None):
        """
        In this constructor function, we define and assign the different matrices "O"
        and list "l", upon which our spectrum solution depends.
        :param J: the Jacobian matrix
        :param L: the matrix containing noise coefficients
        :param S: the rate matrix
        """
        self.G = sp.Matrix([[sp.Rational(str(j)) for j in i] for i in G.tolist()])
        self.Y = sp.Matrix([[sp.Rational(str(j)) for j in i] for i in Y.tolist()])
        self.Y_inv = self.Y.inv()
        self.n = G.shape[0]

        """Define the solution lists"""
        self.P = [sp.zeros(self.n, self.n) for _ in range(2 * self.n + 1)]
        self.S = [sp.zeros(self.n, self.n) for _ in range(2 * self.n + 1)]
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
            self.P[alpha - 1] = recursive_solution_g.P_func(
                self.q[alpha + 1], self.Y, self.G, self.P[alpha], self.P[alpha + 1]
            )
            self.S[2 * self.n - 1 - alpha] = self.S_func(alpha, self.P[alpha - 1])
            self.q[alpha] = recursive_solution_g.q_func(
                self.G, self.n, alpha, self.Y_inv, self.P[alpha], self.P[alpha - 1]
            )
        self.S[2 * self.n - 1] = self.S_func(0, self.P[-1])
        self.q[0] = recursive_solution_g.q_func(
            self.G, self.n, 0, self.Y_inv, self.P[0], self.P[-1]
        )
        self.S[2 * self.n] = self.S_func(-1, self.P[-1])
        return None

    @staticmethod
    def P_func(q, Y, G, P, P1):
        """
        This function returns P_{alpha-1} using the recursive solution.
        P_{alpha-1} = q_{alpha+1} * Y + G * P_alpha + P_alpha * G^T - G * P_{alpha+1} * G^T
        """
        return q * Y + G * P + P * G.T - G * P1 * G.T

    @staticmethod
    def q_func(G, n, alpha, Y_inv, P, P1):
        """
        This function returns q_{alpha} using the recursive solution.
        q_{alpha} = (Tr(Y^-1 * G * P_alpha * G^T) - Tr(Y^-1 * G * P_{alpha - 1} )) / (n-alpha/2)
        """
        return ((Y_inv * G * P * G.T).trace() - (Y_inv * G * P1).trace()) / (
            n - Rational(alpha, 2)
        )

    def S_func(self, alpha, P):
        """
        This function returns S_{2n-1-alpha} using the recursive solution.
        S_{2n-1-alpha} = P_{alpha-1} - \sum_{k=0}^{2n-2-alpha} q_{alpha+k+1} S_{k}
        """
        # write the function
        S = P
        for k in range(2 * self.n - 1 - alpha):
            S -= self.q[alpha + k + 1] * self.S[k]
        return S


if __name__ == "__main__":
    pass
