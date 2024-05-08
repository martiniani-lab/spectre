"""We define the class for simulating the Rock-Paper-Scissors-Lizard-Spock model."""

import numpy as np
import torch
import scipy.signal
from torchsde import sdeint
from torch.func import jacrev
import os
from ._dyn_models import _dyn_models
from spectre.utils.util_funs import dynm_fun
from spectre.utils.simulation_class import SDE
from spectre.spectrum_general.matrix_spectrum import matrix_solution
from spectre.spectrum_general.sim_spectrum import sim_solution
from spectre.spectrum_general.spectrum import element_wise

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class RPS(_dyn_models):
    def __init__(self, N, mu=0, eta=0.01, noise_type="multiplicative"):
        super(RPS, self).__init__()
        """
        This class implements a rock paper scissors type system with mutations,
        exhibiting a stable spiral solution. The model is asymmetric (rock always beats
        scissors). We consider that the populations follow a sociological strategy
        where they play the same strategy as the average of the population. We consider
        global mutations with a rate of mu. The system is described by the following
        set of equations (for a system of size n):
        da1/dt = a1*(f_a1 - phi) + mu*(-(n-1)*a1 + a2 + ... + an) + eta
        .
        .
        .
        dan/dt = an*(f_an - phi) + mu*(-(n-1)*an + a1 + a2 + ...) + eta
        where phi is the average payoff of the population (a1*f_a1 + a2*f_a2 + ...),
        f_ai is the payoff of strategy i which is calculated by according to the payoff
        matrix, eta is a random noise term, and mu is the mutation rate.

        Note that this system is dynamically a (n-1) dimensional system, since the
        sum of all populations is equal to 1.

            Parameters
            ----------
            N : int is the dimensionality of the system
        """
        self._N = N  # 5 for RPSLSp
        self.dim = self.N - 1
        self._mu = mu
        self._eta = eta
        self.phi = 0

        self.payoff = self.make_payoff_matrix()

        """Make the mutation weights"""
        self.mutation_weights = torch.ones(self.N, self.N) - self.N * torch.eye(self.N)

        """Type of noise"""
        self._noise_type = noise_type

        """Initialize the circuit"""
        self.initialize_circuit()
        self.make_noise_mats()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        if N > 0 and N % 2 == 1:
            self._N = N
            self.dim = self.N - 1
            self.initialize_circuit()
            self.make_noise_mats()
            self.mutation_weights = torch.ones(self.N, self.N) - self.N * torch.eye(
                self.N
            )
        else:
            raise ValueError("N must be an odd positive integer")

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        if 0 <= mu <= 1:
            self._mu = mu
            self.initialize_circuit()
        else:
            raise ValueError("mu should be between 0 and 1")

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, eta):
        if eta > 0:
            self._eta = eta
            self.make_noise_mats()
        else:
            raise ValueError("eta (S.D. of noise) should be positive")

    @property
    def noise_type(self):
        return self._noise_type

    @noise_type.setter
    def noise_type(self, noise_type):
        if noise_type in ["additive", "multiplicative", "cor_add", "cor_mul"]:
            self._noise_type = noise_type
            self.make_noise_mats()
        else:
            raise ValueError(
                "Noise type must be one of additive, multiplicative,"
                " cor_add or cor_mul"
            )

    def get_instance_variables(self):
        return (self.N, self.mu, self.eta)

    def make_payoff_matrix(self):
        n = self.N
        payoff = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    payoff[i, j] = 0
                elif i > j:
                    payoff[i, j] = (-1) ** (i + j + 1)
                else:
                    payoff[i, j] = (-1) ** (i + j)
        return payoff

    def initialize_circuit(self):
        """
        This function makes the jacobian and the noise matrix of the circuit.
        :return: None
        """
        self.payoff = self.make_payoff_matrix()

        """Make the jacobian"""
        ss = self.steady_state()
        _ = self.jacobian_autograd(ss)
        return None

    def jacobian_autograd(self, ss):
        """
        Calculates the Jacobian of the dynamical system using torch autograd.
        """
        J = jacrev(self._dynamical_fun, argnums=1)(0, ss)
        self.J = J
        return J

    def make_noise_mats(self):
        """
        This function creates the noise matrices for the SDE simulation.
        :return: None
        """
        ss = self.steady_state()
        self.L = self.make_Ly(0, ss)
        self.D = self.make_D()
        self.S = torch.sqrt(self.D)
        return

    def make_Ly(self, t, x):
        """
        This function creates the noise matrix.
        """
        L = self.eta * torch.diag(x)
        return L

    def make_D(self):
        """
        This function creates the D matrix for the noise.
        :return: The D matrix
        """
        D = torch.eye(self.dim)
        return D

    def steady_state(self):
        """
        This method calculates the steady state of the system.
        """
        steady_state = torch.ones(self.dim) / self.N
        return steady_state

    def jacobian(self):
        """
        This method makes the Jacobian matrix of the system.
        """
        n = self.N
        J = torch.zeros(self.dim, self.dim)
        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    J[i, j] = ((-1) ** (i + 1) / n) - n * self.mu
                elif i > j:
                    J[i, j] = max((-1) ** j, 0) * (-1) ** (i + 1) * 2 / n
                else:
                    J[i, j] = max((-1) ** (j - 1), 0) * (-1) ** (i + 1) * 2 / n
        self.J = J
        return J

    @dynm_fun
    def _dynamical_fun(self, t, x):
        """
        This function defines the dynamics of the RPSLSp model.
        :param x: The state of the system.
        :return: The derivative of the system at the current time-step.
        """
        x = x.squeeze(0)  # Remove the extra dimension during sde simulation
        x_new = torch.cat((x, (1 - torch.sum(x)).unsqueeze(0)), dim=0)
        dxdt = x_new * (self.payoff @ x_new - self.phi) + self.mu * (
            self.mutation_weights @ x_new
        )
        return dxdt[:-1]


if __name__ == "__main__":
    pass
