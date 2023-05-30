import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import functorch as ftorch
import os
# import symengine as sp
import timeit
from functools import lru_cache

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


torch.set_default_dtype(torch.float64)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# n_cores = 1


class PSD:
    def __init__(self, J=-torch.eye(3), L=torch.eye(3), S=torch.eye(3)):
        """
        In this constructor function, we define and assign the different matrices "O"
        and list "l", upon which our spectrum solution depends.
        :param J: the Jacobian matrix
        :param L: the matrix containing noise coefficients
        :param S: the matrix containing the S.D. of the noise terms added
        """
        self.J = J.numpy()
        self.L = L.numpy()
        self.S = S.numpy()
        self.n = J.shape[0]

        # we define the index matrix and the l matrix
        self.l = torch.zeros(self.n, self.n, dtype=torch.int64)
        idx = torch.ones((self.n, self.n, self.n, self.n), dtype=torch.bool)
        index_row = np.tile(np.arange(self.n - 1).astype(np.ushort),
                            (self.n, self.n, 1))
        index_col = np.tile(np.arange(self.n - 1).astype(np.ushort),
                            (self.n, self.n, 1))
        self.l = sp.Matrix([[sp.Rational(str(j)) for j in i] for i in self.l.tolist()])

        for i in range(self.n):
            for j in range(self.n):
                idx[i, j, i] = False
                idx[i, j, :, j] = False
                if not i == j:
                    if i > j:
                        self.l[i, j] = j
                        index_col[i, j] = np.insert(np.delete(index_col[i, j], i - 1), j, i - 1)
                    else:
                        self.l[i, j] = i
                        index_row[i, j] = np.insert(np.delete(index_row[i, j], j - 1), i, j - 1)

        index_row = torch.from_numpy(index_row.astype(np.int32)).long()
        index_col = torch.from_numpy(index_col.astype(np.int32)).long()

        """
        Now we define the O tensor.
        First we assign the corresponding excluded matrices to O tensor. After that, we
        apply the row/column change operation using vmap
        """
        # Define the tensor storing the O matrices required for the solution, we first
        self.O = J.repeat(self.n, self.n, 1, 1)[idx].reshape(self.n, self.n, self.n - 1, self.n - 1)
        self.O = ftorch.vmap(PSD.make_O, in_dims=(0, 0, 0))(
            self.O.reshape(-1, self.n - 1, self.n - 1),
            index_row.reshape(-1, self.n - 1), index_col.reshape(-1, self.n - 1))
        self.O = self.O.reshape(self.n, self.n, self.n - 1, self.n - 1).numpy()

        # We define the coefficients of the denominator. Note: all are the same.
        self.q_all = None
        # This dictionary stores the O matrices in rational form
        self.O_dict = {}
        # Convert the Jacobian matrix into a hashable object
        self.J = sp.matrices.ImmutableMatrix(self.J)

        # Define all the cache containers
        self.q = lru_cache(maxsize=None)(self._q)
        self.d = lru_cache(maxsize=None)(self._d)
        self.g_1 = lru_cache(maxsize=None)(self._g_1)
        self.g_2 = lru_cache(maxsize=None)(self._g_2)
        self.h_1 = lru_cache(maxsize=None)(self._h_1)
        self.h_2 = lru_cache(maxsize=None)(self._h_2)
        self.f = lru_cache(maxsize=None)(self._f)
        self.s_1 = lru_cache(maxsize=None)(self._s_1)
        self.s_2 = lru_cache(maxsize=None)(self._s_2)
        self.t_1 = lru_cache(maxsize=None)(self._t_1)
        self.t_2 = lru_cache(maxsize=None)(self._t_2)
        self.bell_inp = lru_cache(maxsize=None)(self._bell_inp)
        self.comp_bell = lru_cache(maxsize=None)(self._comp_bell)
        self.r_k = lru_cache(maxsize=None)(self._r_k)

    @staticmethod
    def make_O(O, index_row, index_col):
        return torch.index_select(torch.index_select(O, 0, index_row), 1, index_col)

    @staticmethod
    @lru_cache(maxsize=None)
    def excluded_mat(A, l):
        """
        This function returns the excluded matrix of A about (l, l).
        """
        B = A.as_mutable()
        B.row_del(l)
        B.col_del(l)
        return sp.ImmutableMatrix(B)

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

    def O_dict_return(self, i):
        """
        This function returns the set of O matrices [:][i] in rational form in sympy.
        """
        if i not in self.O_dict.keys():
            l = [[[sp.Rational(str(k)) for k in j] for j in i] for i in self.O[:, i].tolist()]
            self.O_dict[i] = list(map(sp.matrices.ImmutableMatrix, l))
        return self.O_dict[i]

    def auto_spectrum(self, idx, frequency=None):
        """
        This function returns the auto-spectrum (diagonal elements) of the spectral
        density matrix.
        :param frequency: Torch tensor containing the frequencies for which the spectra
         are desired.
        :param idx: the index of the variable for which the auto-spectra is desired.
        Note, here we follow indexing starting with 0.
        :return: the auto-PSD for the given variable.
        """
        frequency = torch.logspace(np.log10(1), np.log10(1000), 100) if frequency is None else frequency
        # Convert the frequency from Hz to omega
        freq = sp.Matrix([sp.Rational(str(i)) for i in frequency.tolist()])
        om = 2 * sp.pi * freq
        n = self.n

        # Denominator
        if self.q_all is None:
            self.q_all = self.q_all_coeffs()
            q_all = self.q_all
        else:
            q_all = self.q_all

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
        if self.q_all is None:
            self.q_all = self.q_all_coeffs()
            q_all = self.q_all
        else:
            q_all = self.q_all

        # Denominator
        powers = 2 * torch.arange(n + 1)
        denm = torch.zeros(om.shape[0])
        for k in range(om.shape[0]):
            for m in range(n + 1):
                denm[k] += float(q_all[m] * om[k] ** powers[m])

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

    def p_auto_all_coeffs(self, i):
        """
        This function returns all the coefficients of the numerator of the auto-spectrum.
        :param i: the index of the variable for which the auto-spectrum is desired.
        :return: the coefficients of the numerator of the auto-spectrum.
        """
        alphas = np.arange(self.n)
        O = self.O_dict_return(i)
        coeffs = [self.p_auto(i, j.item(), O) for j in alphas]
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
                    temp += coeff0 * self.L[j, m] ** 2 * self.f(O[j], self.l[j, i], alpha)
                    temp -= coeff0 * self.L[i, m] * self.L[j, m] * self.s_1(O[i], O[j], self.l[j, i], alpha)
            for j in range(n):
                if not (j == i):
                    for k in range(j):
                        if not (k == i):
                            temp += coeff0 * self.L[j, m] * self.L[k, m] * self.t_1(O[j], O[k], self.l[j, i], self.l[k, i], alpha)
        return temp

    def p_cross_r_all_coeffs(self, i, j):
        """
        This function returns all the coefficients of the real part of the numerator
        of cross-spectrum.
        :param i: the index of the 1st variable for which the auto-spectrum is desired.
        :param j: the index of the 2nd variable for which the auto-spectrum is desired.
        :return: the coefficients of the real part of the numerator of the cross-spectrum.
        """
        alphas = np.arange(self.n)
        Oi = self.O_dict_return(i)
        Oj = self.O_dict_return(j)
        coeffs = [self.p_cross_r(i, j, k.item(), Oi, Oj) for k in alphas]
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
                    temp -= coeff0 * self.L[i, m] * self.L[k, m] * self.s_1(Oi[i], Oj[k], self.l[k, j], alpha)
            for k in range(n):
                if not (k == i):
                    temp -= coeff0 * self.L[k, m] * self.L[j, m] * self.s_1(Oj[j], Oi[k], self.l[k, i], alpha)
            for k in range(n):
                if not (k == i):
                    for q in range(n):
                        if not (q == j):
                            temp += coeff0 * self.L[k, m] * self.L[q, m] * self.t_1(Oi[k], Oj[q], self.l[i, k], self.l[j, q], alpha)
        return temp

    def p_cross_i_all_coeffs(self, i, j):
        """
        This function returns all the coefficients of the imag part of the numerator
        of cross-spectrum.
        :param i: the index of the 1st variable for which the auto-spectrum is desired.
        :param j: the index of the 2nd variable for which the auto-spectrum is desired.
        :return: the coefficients of the imag part of the numerator of the cross-spectrum.
        """
        alphas = np.arange(self.n-1)
        Oi = self.O_dict_return(i)
        Oj = self.O_dict_return(j)
        coeffs = [self.p_cross_i(i, j, k.item(), Oi, Oj) for k in alphas]
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
                    temp -= coeff0 * self.L[i, m] * self.L[k, m] * self.s_2(Oi[i], Oj[k], self.l[k, j], alpha)
            for k in range(n):
                if not (k == i):
                    temp += coeff0 * self.L[k, m] * self.L[j, m] * self.s_2(Oj[j], Oi[k], self.l[k, i], alpha)
            for k in range(n):
                if not (k == i):
                    for q in range(n):
                        if not (q == j):
                            temp += coeff0 * self.L[k, m] * self.L[q, m] * self.t_2(Oi[k], Oj[q], self.l[i, k], self.l[j, q], alpha)
        return temp

    def q_all_coeffs(self):
        alphas = np.arange(self.n+1)
        coeffs = [self.q(i.item()) for i in alphas]
        return coeffs

    def _q(self, alpha):
        """
        This function returns the coefficient of the denominator of the spectrum.
        Note that the denominator is the same for all the variables.
        :param alpha: power of omega
        :return: the value of the coefficient of omega^2alpha.
        """
        return self.d(self.J, alpha)

    def _d(self, A, alpha):
        """
        d(w; A) = ||A+iwI||^2
        Returns the coefficient of w^{2*alpha}
        """
        n = A.shape[0]
        return (-1) ** abs(n - alpha) * self.comp_bell(self.bell_inp(A, 2, n - alpha)) / sp.factorial(n - alpha)

    def _g_1(self, A, B, alpha):
        """
        g1 = Real{2w Conjugate{|A+iwI|}|B+iwI|}
        Returns the coefficient of the power of 2*alpha + 1. See SI for details.
        """
        n = A.shape[0]
        temp = sp.Rational("0")
        for j in range(n + 1):
            k = 2 * alpha - j
            if k <= n - 1 and k >= 0:
                coeff = 2 * (-1) ** abs(alpha - j - 1) / (sp.factorial(n - j) * sp.factorial(n - k - 1))
                temp += coeff * self.comp_bell(
                    self.bell_inp(A, 1, n - j)) * self.comp_bell(
                    self.bell_inp(B, 1, n - k - 1))
        return temp

    def _g_2(self, A, B, alpha):
        """
        g2 = Imaginary{2w Conjugate{|A+iwI|}|B+iwI|}
        Returns the coefficient of the power of 2*alpha. See SI for details.
        """
        n = A.shape[0]
        temp = sp.Rational("0")
        for j in range(n + 1):
            k = 2 * alpha - 1 - j
            if k <= n - 1 and k >= 0:
                coeff = 2 * (-1) ** abs(alpha - j - 1) / (sp.factorial(n - j) * sp.factorial(n - k - 1))
                temp += coeff * self.comp_bell(
                    self.bell_inp(A, 1, n - j)) * self.comp_bell(
                    self.bell_inp(B, 1, n - k - 1))
        return temp

    def _h_1(self, A, B, alpha):
        """
        h1 = Real{2 Conjugate{|A+iwI|}|B+iwI|}
        Returns the coefficient of the power of 2*alpha. See SI for details.
        """
        n = A.shape[0]  # A and C have same dimensions
        temp = sp.Rational("0")
        for j in range(n + 1):
            k = 2 * alpha - j
            if k <= n and k >= 0:
                coeff = 2 * (-1) ** abs(alpha - k) / (sp.factorial(n - j) * sp.factorial(n - k))
                temp += coeff * self.comp_bell(
                    self.bell_inp(A, 1, n - j)) * self.comp_bell(
                    self.bell_inp(B, 1, n - k))
        return temp

    def _h_2(self, A, B, alpha):
        """
        h2 = Imaginary{2 Conjugate{|A+iwI|}|B+iwI|}
        Returns the coefficient of the power of 2*alpha+1. See SI for details.
        """
        n = A.shape[0]  # A and C have same dimensions
        temp = sp.Rational("0")
        for j in range(n + 1):
            k = 2 * alpha + 1 - j
            if k <= n and k >= 0:
                coeff = 2 * (-1) ** abs(alpha - k) / (sp.factorial(n - j) * sp.factorial(n - k))
                temp += coeff * self.comp_bell(
                    self.bell_inp(A, 1, n - j)) * self.comp_bell(
                    self.bell_inp(B, 1, n - k))
        return temp

    def _f(self, A, l, alpha):
        """
        See SI for definition.
        """
        # Calculate minor about l,l element
        B = PSD.excluded_mat(A, l)
        return self.d(A, alpha) + self.d(B, alpha - 1) + self.g_2(A, B, alpha)

    def _s_1(self, A, B, l, alpha):
        """
        See SI for definition.
        """
        C = PSD.excluded_mat(B, l)
        return self.h_1(A, B, alpha) + self.g_2(A, C, alpha)

    def _s_2(self, A, B, l, alpha):
        """
        See SI for definition.
        """
        C = PSD.excluded_mat(B, l)
        return - self.h_2(A, B, alpha) + self.g_1(A, C, alpha)

    def _t_1(self, A, B, l1, l2, alpha):
        """
        See SI for definition.
        """
        C = PSD.excluded_mat(A, l1)
        D = PSD.excluded_mat(B, l2)
        return self.h_1(A, B, alpha) + self.h_1(C, D, alpha - 1) + self.g_2(B, C, alpha) + self.g_2(A, D, alpha)

    def _t_2(self, A, B, l1, l2, alpha):
        """
        See SI for definition.
        """
        C = PSD.excluded_mat(A, l1)
        D = PSD.excluded_mat(B, l2)
        return - self.h_2(A, B, alpha) - self.h_2(C, D, alpha - 1) - self.g_1(B, C, alpha) + self.g_1(A, D, alpha)

    def _bell_inp(self, mat, kappa, k):
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

    def _comp_bell(self, x):
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
        return PSD.hessenberg_det(B_k)

    def _r_k(self, mat, kappa, k):
        """
        This function calculates the k-th power sum.
        """
        return sp.trace(self.mat_power(mat, kappa * k))

    @staticmethod
    @lru_cache(maxsize=None)
    def mat_power(mat, p):
        """
        Calculates the matrix power of a matrix. We use recursion to make the code
        faster.
        """
        if p == 0:
            return sp.ImmutableMatrix(sp.eye(mat.shape[0]))
        elif p == 1:
            return mat
        elif p % 2 == 0:
            return PSD.mat_power(mat * mat, p / 2)
        else:
            return mat * PSD.mat_power(mat * mat, (p - 1) / 2)

        # return mat**p


if __name__ == '__main__':
    n = 7
    mat = torch.randn(n, n)
    mat = mat.round(decimals=3)
    L = torch.eye(n)
    L[1, 1] = 2/10
    L[2, 3] = 5/6

    S = torch.eye(n)
    # S[0, 0] = math.sqrt(0.1)
    # S[3, 3] = math.sqrt(1/200)
    # S[1,1] = torch.sqrt(2.3)

    # mat = mat - torch.diag(mat)

    model = PSD(mat, L, S)
    # a_coeffs = model.a_auto_all_coeffs(0)
    # val = model.a_auto(0, 0)
    start_time = timeit.default_timer()
    print(list(model.p_auto_all_coeffs(0)))
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    # print(model.a_cross_i_all_coeffs(0, 1))
    # print(model.a_cross_r_all_coeffs(0, 1))
    # freq = np.linspace(0.01, 100, 100)
    # cross_pow = model.cross_spectrum(freq, 0, 1)
    # power = model.auto_spectrum(freq, 2)
    # print(power)
    # psd = model.cross_spectrum(2 / (2 * torch.pi), 0, 0)
    # print(psd)







