import numpy as np
import torch
import math
import sympy as sp
import os
import timeit
from functools import lru_cache


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.set_default_dtype(torch.float64)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class symbolic:
    def __init__(self, n, L, S):
        """
        In this constructor function, we define and assign the different matrices "O"
        and list "l", upon which our spectrum solution depends.
        :param J: the Jacobian matrix
        :param L: the matrix containing noise coefficients
        :param D: the matrix containing the variance of the noise terms added
        """
        super().__init__()
        # self.J = sp.Matrix([[sp.symbols('a%d%d' % (k, j)) for j in range(n)] for k in range(n)])
        self.J = sp.symbols('J')
        self.L = L
        self.S = S
        self.n = n

        self.O = [[sp.Symbol('O_{}{}'.format(i, j)) for j in range(self.n)] for i in range(self.n)]
        self.O_prime = [[sp.Symbol('O_{}{}\''.format(i, j)) for j in range(n)] for i in range(n)]

        # We define the coefficients of the denominator. Note: all are the same.
        self.q_all = None

    @staticmethod
    def hessenberg_det(A):
        """
        This function returns the determinant of the upperHessenberg matrix A.
        This returns the answer in O(n^2). This code is adapted from the julia
        implementation found at:
        'https://github.com/JuliaLang/julia/blob/master/stdlib/LinearAlgebra/src/hessenberg.jl'
        """
        m = A.shape[0]
        det = A[0, 0]
        pdet = sp.Rational("1")

        if m == 1:
            return det

        prods = [0] * (m-1)

        for j in range(1, m):
            prods[j - 1] = pdet
            pdet = det
            det *= A[j, j]
            a = A[j, j - 1]
            for k in range(j - 1, 0, -2):
                prods[k] *= a
                prods[k - 1] *= a
                det -= A[k, j] * prods[k] - A[k - 1, j] * prods[k - 1]
            if (j - 1) % 2 == 0:
                prods[0] *= a
                det -= A[0, j] * prods[0]

        return det

    def p_auto_all_coeffs(self, i):
        """
        This function returns all the coefficients of the numerator of the auto-spectrum.
        :param i: the index of the neuron for which the auto-spectrum is desired.
        :return: the coefficients of the numerator of the auto-spectrum.
        """
        alphas = np.arange(self.n)
        O = [self.O[j][i] for j in range(self.n)]
        coeffs = [sp.together(sp.simplify(self.p_auto(i, j.item(), O))) for j in alphas]
        return coeffs

    def p_auto(self, i, alpha, O):
        """
        This function returns the coefficient of the numerator of the auto-spectrum
        of the i-th variable.
        :param i: the index of the variable for which the auto-spectra is desired.
        :param alpha: power of omega
        :return: the value of the coefficient of omega^2alpha.
        """
        temp = sp.Rational("0")
        n = self.n
        for m in range(n):
            coeff0 = self.S[m, m] ** 2
            temp += coeff0 * self.L[i, m] ** 2 * self.d(O[i], alpha)
            for j in range(n):
                if not (j == i):
                    temp += coeff0 * self.L[j, m] ** 2 * self.f(O[j], None, alpha)
                    temp -= coeff0 * self.L[i, m] * self.L[j, m] * self.s_1(O[i], O[j], None, alpha)
            for j in range(n):
                if not (j == i):
                    for k in range(j):
                        if not (k == i):
                            temp += coeff0 * self.L[j, m] * self.L[k, m] * self.t_1(O[j], O[k], None, None, alpha)
        return temp

    def p_cross_r_all_coeffs(self, i, j):
        alphas = np.arange(self.n)
        Oi = [self.O[k][i] for k in range(self.n)]
        Oj = [self.O[k][j] for k in range(self.n)]
        coeffs = [sp.together(sp.simplify(self.p_cross_r(i, j, k.item(), Oi, Oj))) for k in alphas]
        return coeffs

    def p_cross_r(self, i, j, alpha, Oi, Oj):
        """
        This function returns the coefficient of the real part of the numerator
        for the cross-spectrum between i-j th variable.
        :param alpha: power of omega
        :return: the value of the coefficient of omega^2alpha.
        """
        temp = sp.Rational("0")
        n = self.n
        for m in range(n):
            coeff0 = self.S[m, m] ** 2 / 2
            temp += coeff0 * self.L[i, m] * self.L[j, m] * self.h_1(Oi[i], Oj[j], alpha)
            for k in range(n):
                if not (k == j):
                    temp -= coeff0 * self.L[i, m] * self.L[k, m] * self.s_1(Oi[i], Oj[k], None, alpha)
            for k in range(n):
                if not (k == i):
                    temp -= coeff0 * self.L[k, m] * self.L[j, m] * self.s_1(Oj[j], Oi[k], None, alpha)
            for k in range(n):
                if not (k == i):
                    for q in range(n):
                        if not (q == j):
                            temp += coeff0 * self.L[k, m] * self.L[q, m] * self.t_1(Oi[k], Oj[q], None, None, alpha)
        return temp

    def p_cross_i_all_coeffs(self, i, j):
        alphas = np.arange(self.n-1)
        Oi = [self.O[k][i] for k in range(self.n)]
        Oj = [self.O[k][j] for k in range(self.n)]
        coeffs = [sp.together(sp.simplify(self.p_cross_i(i, j, k.item(), Oi, Oj))) for k in alphas]
        return coeffs

    def p_cross_i(self, i, j, alpha, Oi, Oj):
        """
        This function returns the coefficient of the imaginary part of the numerator
        for the cross-spectrum between i-j th variable.
        :param alpha: power of omega
        :return: the value of the coefficient of omega^(2alpha+1).
        """
        temp = sp.Rational("0")
        n = self.n
        for m in range(n):
            coeff0 = self.S[m, m] ** 2 / 2
            temp -= coeff0 * self.L[i, m] * self.L[j, m] * self.h_2(Oi[i], Oj[j], alpha)
            for k in range(n):
                if not (k == j):
                    temp -= coeff0 * self.L[i, m] * self.L[k, m] * self.s_2(Oi[i], Oj[k], None, alpha)
            for k in range(n):
                if not (k == i):
                    temp += coeff0 * self.L[k, m] * self.L[j, m] * self.s_2(Oj[j], Oi[k], None, alpha)
            for k in range(n):
                if not (k == i):
                    for q in range(n):
                        if not (q == j):
                            temp += coeff0 * self.L[k, m] * self.L[q, m] * self.t_2(Oi[k], Oj[q], None, None, alpha)
        return temp

    def q_all_coeffs(self):
        alphas = np.arange(self.n+1)
        coeffs = [sp.together(sp.simplify(self.q(i.item()))) for i in alphas]
        return coeffs

    def q(self, alpha):
        """
        This function returns the coefficient of the denominator of the spectrum.
        Note that the denominator is the same for all the variables.
        :param alpha: power of omega
        :return: the value of the coefficient of omega^2alpha.
        """
        return self.d(self.J, alpha)

    def d(self, A, alpha):
        """
        d(w; A) = ||A+iwI||^2
        Returns the coefficient of w^{2*alpha}
        """
        n = self.shape(A)
        return (-1) ** abs(n - alpha) * self.comp_bell(self.bell_inp(A, 2, n - alpha)) / sp.factorial(n - alpha)

    def g_1(self, A, B, alpha):
        """
        g1 = Real{2w Conjugate{|A+iwI|}|B+iwI|}
        Returns the coefficient of the power of 2*alpha + 1. See SI for details.
        """
        n = self.shape(A)
        temp = sp.Rational("0")
        for j in range(n + 1):
            k = 2 * alpha - j
            if k <= n - 1 and k >= 0:
                coeff = 2 * (-1) ** abs(alpha - j - 1) / (sp.factorial(n - j) * sp.factorial(n - k - 1))
                temp += coeff * self.comp_bell(
                    self.bell_inp(A, 1, n - j)) * self.comp_bell(
                    self.bell_inp(B, 1, n - k - 1))
        return temp

    def g_2(self, A, B, alpha):
        """
        g2 = Imaginary{2w Conjugate{|A+iwI|}|B+iwI|}
        Returns the coefficient of the power of 2*alpha. See SI for details.
        """
        n = self.shape(A)
        temp = sp.Rational("0")
        for j in range(n + 1):
            k = 2 * alpha - 1 - j
            if k <= n - 1 and k >= 0:
                coeff = 2 * (-1) ** abs(alpha - j - 1) / (sp.factorial(n - j) * sp.factorial(n - k - 1))
                temp += coeff * self.comp_bell(
                    self.bell_inp(A, 1, n - j)) * self.comp_bell(
                    self.bell_inp(B, 1, n - k - 1))
        return temp

    def h_1(self, A, B, alpha):
        """
        h1 = Real{2 Conjugate{|A+iwI|}|B+iwI|}
        Returns the coefficient of the power of 2*alpha. See SI for details.
        """
        n = self.shape(A)  # A and C have same dimensions
        temp = sp.Rational("0")
        for j in range(n + 1):
            k = 2 * alpha - j
            if k <= n and k >= 0:
                coeff = 2 * (-1) ** abs(alpha - k) / (sp.factorial(n - j) * sp.factorial(n - k))
                temp += coeff * self.comp_bell(
                    self.bell_inp(A, 1, n - j)) * self.comp_bell(
                    self.bell_inp(B, 1, n - k))
        return temp

    def h_2(self, A, B, alpha):
        """
        h2 = Imaginary{2 Conjugate{|A+iwI|}|B+iwI|}
        Returns the coefficient of the power of 2*alpha+1. See SI for details.
        """
        n = self.shape(A)  # A and C have same dimensions
        temp = sp.Rational("0")
        for j in range(n + 1):
            k = 2 * alpha + 1 - j
            if k <= n and k >= 0:
                coeff = 2 * (-1) ** abs(alpha - k) / (sp.factorial(n - j) * sp.factorial(n - k))
                temp += coeff * self.comp_bell(
                    self.bell_inp(A, 1, n - j)) * self.comp_bell(
                    self.bell_inp(B, 1, n - k))
        return temp

    def f(self, A, l, alpha):
        """
        See SI for definition.
        """
        # Calculate minor about l,l element
        B = self.add_string(A, '\'')
        return self.d(A, alpha) + self.d(B, alpha - 1) + self.g_2(A, B, alpha)

    def s_1(self, A, B, l, alpha):
        """
        See SI for definition.
        """
        C = self.add_string(B, '\'')
        return self.h_1(A, B, alpha) + self.g_2(A, C, alpha)

    def s_2(self, A, B, l, alpha):
        """
        See SI for definition.
        """
        C = self.add_string(B, '\'')
        return - self.h_2(A, B, alpha) + self.g_1(A, C, alpha)

    def t_1(self, A, B, l1, l2, alpha):
        """
        See SI for definition.
        """
        C = self.add_string(A, '\'')
        D = self.add_string(B, '\'')
        return self.h_1(A, B, alpha) + self.h_1(C, D, alpha - 1) + self.g_2(B, C, alpha) + self.g_2(A, D, alpha)

    def t_2(self, A, B, l1, l2, alpha):
        """
        See SI for definition.
        """
        C = self.add_string(A, '\'')
        D = self.add_string(B, '\'')
        return - self.h_2(A, B, alpha) - self.h_2(C, D, alpha - 1) - self.g_1(B, C, alpha) + self.g_1(A, D, alpha)

    def bell_inp(self, mat, kappa, k):
        """
        Constructs the input to the Bell polynomial. See, SI.
        """
        if k != 0:
            x = sp.zeros(k, 1)
            for i in range(0, k):
                x[i] = - self.r_k(mat, kappa, i + 1)
            return sp.ImmutableMatrix(x)
        else:
            return sp.ImmutableMatrix(sp.ones(1))

    def comp_bell(self, x):
        k = x.shape[0]
        B_k = sp.zeros(k, k)
        for i in range(0, k):
            for j in range(0, k):
                if j - i < -1:
                    B_k[i, j] = 0
                elif j - i == -1:
                    B_k[i, j] = -i
                else:
                    B_k[i, j] = x[j - i]
        # return sp.det(B_k)
        return self.hessenberg_det(B_k)

    def r_k(self, mat, kappa, k):
        """
        This function calculates the k-th power sum.
        """
        return self.Tr(self.mat_power(mat, kappa * k))
    
    @staticmethod
    def Tr(x):
        return sp.Function('Tr')(x)

    @staticmethod
    def mat_power(mat, p):
        return sp.sympify(mat**p)
    
    @staticmethod
    def add_string(a, string):
        return sp.Symbol(string.join([str(a)[:1], str(a)[1:]]))

    def shape(self, mat):
        if mat == self.J:
            return self.n
        elif str(mat)[1] == '\'':
            return self.n - 2
        else:
            return self.n - 1

        

if __name__ == '__main__':
    pass
