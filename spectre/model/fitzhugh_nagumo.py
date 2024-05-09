"""We define the class for simulating the Fitzhugh_Nagumo model."""

import numpy as np
import torch
import math
from ._dyn_models import _dyn_models
from ..utils.util_funs import dynm_fun


class FHN(_dyn_models):
    def __init__(self, I=0.265, eta1=0, eta2=0.001, method="euler", run_jacobian=True):
        super(FHN, self).__init__()
        """
        This function initializes the various parameters of the Fitzhugh-Nagumo model. 
        See supplementary material for details.
        """
        """Strength of noise"""
        self.N = 2
        self.dim = self.N

        self._eta1 = eta1
        self._eta2 = eta2

        """Relative Time constant"""
        self._epsilon = 0.08

        """Parameters"""
        self._alpha = 0.7
        self._beta = 0.75

        """External stimulus"""
        self._I = I

        """Type of noise"""
        self._noise_type = "multiplicative"

        """Initialize the circuit"""
        self.method = method
        self.run_jacobian = run_jacobian
        self.initialize_circuit(method=method, run_jacobian=run_jacobian)

    @property
    def eta1(self):
        return self._eta1

    @eta1.setter
    def eta1(self, eta):
        if eta >= 0:
            self._eta1 = eta
            self.make_noise_mats()
        else:
            raise ValueError("Noise S.D. must be a positive float")

    @property
    def eta2(self):
        return self._eta2

    @eta2.setter
    def eta2(self, eta):
        if eta >= 0:
            self._eta2 = eta
            self.make_noise_mats()
        else:
            raise ValueError("Noise S.D. must be a positive float")

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        if epsilon > 0:
            self._epsilon = epsilon
            self.initialize_circuit(method=self.method, run_jacobian=self.run_jacobian)
        else:
            raise ValueError("Relative time constant must be a positive float")

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if alpha > 0:
            self._alpha = alpha
            self.initialize_circuit(method=self.method, run_jacobian=self.run_jacobian)
        else:
            raise ValueError("Alpha must be a positive float")

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        if beta > 0:
            self._beta = beta
            self.initialize_circuit(method=self.method, run_jacobian=self.run_jacobian)
        else:
            raise ValueError("Beta must be a positive float")

    @property
    def I(self):
        return self._I

    @I.setter
    def I(self, I):
        if I >= 0:
            self._I = I
            self.initialize_circuit(method=self.method, run_jacobian=self.run_jacobian)
        else:
            raise ValueError("External stimulus must be a positive float")

    @property
    def noise_type(self):
        return self._noise_type

    @noise_type.setter
    def noise_type(self, noise_type):
        if noise_type in ["additive", "multiplicative"]:
            self._noise_type = noise_type
            self.make_noise_mats()
        else:
            raise ValueError("Noise type must be either additive or multiplicative")

    def get_instance_variables(self):
        return (self.eta1, self.eta2, self.epsilon, self.alpha, self.beta, self.I)

    def initialize_circuit(self, method="euler", run_jacobian=True):
        """
        This function makes the jacobian and the noise matrix of the circuit.
        :return: None
        """
        """Make the jacobian"""
        if run_jacobian:
            tau = 1.0
            time = tau * 200
            dt = 0.05 * tau
            points = int(time / dt)
            _ = self.jacobian_autograd(time=time, points=points, method=method)

        """Make noise matrices"""
        self.make_noise_mats()
        return

    def make_Ly(self, t, x):
        """
        This function creates the noise matrix.
        :param eta1: The strength (coefficient) of the noise in v.
        :param eta2: The strength (coefficient) of the noise in w.
        """
        L = torch.zeros((self.N, self.N))
        L[0, 0] = torch.abs(x[0]) * self.eta1
        L[1, 1] = torch.abs(x[1]) * self.eta2
        return L

    def make_noise_mats(self):
        """
        This function creates the noise matrices for the SDE simulation.
        :return: None
        """
        self.L = self.make_L(time=None, points=None, method=self.method)
        self.D = self.make_D()
        self.S = torch.sqrt(self.D)
        return

    def make_D(self):
        """
        This function creates the D matrix for the noise.
        :return: The D matrix
        """
        D = torch.eye(self.N)
        return D

    def analytical_ss(self):
        """This function returns the analytical steady state of the circuit."""
        delta = (1 / self.beta - 1) ** 3 + (9 / 4) * (
            self.alpha / self.beta - self.I
        ) ** 2
        t = -(3 / 2) * (self.alpha / self.beta - self.I)
        v_e = np.cbrt(t - math.sqrt(delta)) + np.cbrt(t + math.sqrt(delta))
        w_e = (v_e + self.alpha) / self.beta
        return torch.tensor([v_e, w_e])

    def psd_v(self, freq):
        """This function returns the power spectral density of v."""
        om = 2 * np.pi * freq
        v_e, w_e = self.analytical_ss()
        numerator = w_e**2 * self.eta2**2
        denominator = (
            (self.epsilon + (v_e**2 - 1) * self.beta * self.epsilon) ** 2
            + ((v_e**2 - 1) ** 2 - 2 * self.epsilon + self.beta**2 * self.epsilon**2)
            * om**2
            + om**4
        )
        return numerator / denominator

    @dynm_fun
    def _dynamical_fun(self, t, x):
        """
        This function defines the dynamics of the Fitzhugh-Nagumo model.
        :param x: The state of the circuit.
        :return: The derivative of the circuit at the current time-step.
        """
        x = x.squeeze(0)  # Remove the extra dimension during sde simulation
        v = x[0:1]
        w = x[1:2]
        dvdt = v - v**3 / 3 - w + self.I
        dwdt = self.epsilon * (v + self.alpha - self.beta * w)
        return torch.cat((dvdt, dwdt))


if __name__ == "__main__":
    pass
