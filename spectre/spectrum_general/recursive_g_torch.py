import torch
import numpy as np

class recursive_g_torch:
    def __init__(self, G=None, Y=None):
        """
        In this constructor function, we define and assign the different matrices "O"
        and list "l", upon which our spectrum solution depends.
        :param G: the matrix G
        :param Y: the matrix Y
        """
        self.G = G
        self.Y = Y
        self.Y_inv = torch.inverse(self.Y)
        self.n = G.shape[0]

        """Define the solution lists"""
        self.t = [torch.tensor(1.0, dtype=torch.float64) for _ in range(2 * self.n + 1)]
        self.R = [torch.zeros(self.n, self.n, dtype=torch.float64) for _ in range(2 * self.n + 1)]
        self.S = [torch.zeros(self.n, self.n, dtype=torch.float64) for _ in range(2 * self.n + 1)]

        """Find the solution matrices"""
        self.find_solution_recursive()

    def find_solution_recursive(self):
        """
        This function finds the recursive solution.
        """
        for j in range(2 * self.n + 1):
            self.t[j] = self.t_func(j)
            self.R[j] = self.R_func(j)
            self.S[j] = self.S_func(j)
            # print(self.t[j])
            if j==1:
                print(self.R[j])
            # print(self.S[j])
        return None
    
    def t_func(self, j):
        """
        This function returns q_{alpha} using the recursive solution.
        t_{j} = (Tr(Y^-1 * G * R_{j-2} * G^T) - Tr(Y^-1 * G * R_{j-1} )) / (2/j)
        """
        if j == 0:
            return torch.tensor(1.0, dtype=torch.float64)
        elif j == 1:
            return - torch.trace(self.Y_inv @ self.G @ self.R[j-1]) * (2.0 / j)
        else:
            term1 = torch.trace(self.Y_inv @ self.G @ self.R[j-2] @ self.G.T)
            term2 = torch.trace(self.Y_inv @ self.G @ self.R[j-1])
            return (term1 - term2) * (2.0 / j)

    def R_func(self, j):
        """
        This function returns R_{j} using the recursive solution.
        R_{j} = t_{j} * Y + G * R_{j-1} + R_{j-1} * G^T - G * R_{j-2} * G^T
        """
        if j == 0:
            return self.t[j] * self.Y
        elif j == 1:
            print(self.t[j] * self.Y + self.G @ self.R[j-1] + self.R[j-1] @ self.G.T)
            return self.t[j] * self.Y + self.G @ self.R[j-1] + self.R[j-1] @ self.G.T
        else:
            return self.t[j] * self.Y + self.G @ self.R[j-1] + self.R[j-1] @ self.G.T - self.G @ self.R[j-2] @ self.G.T

    def S_func(self, j):
        """
        This function returns S_{j} using the recursive solution.
        S_{j} = R_{j} - \sum_{k=0}^{j-1} t_{j-k} S_{k}
        """
        S = self.R[j]
        for k in range(j):
            S -= self.t[j-k] * self.S[k]
        return S


if __name__ == "__main__":
    pass
