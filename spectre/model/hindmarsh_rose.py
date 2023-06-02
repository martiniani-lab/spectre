"""We define the class for simulating the 3D Hindmarsh-Rose model."""
import numpy as np
import torch
import torch.nn as nn
import torch.func as ftorch
from ._dyn_models import _dyn_models
from spectre.utils.util_funs import dynm_fun
import math
import scipy.signal
from torchdiffeq import odeint
from torchsde import sdeint
import matplotlib.pyplot as plt
from spectre.utils.simulation_class import SDE
from spectre.spectrum_general.matrix_spectrum import matrix_solution
from spectre.spectrum_general.sim_spectrum import sim_solution
from spectre.spectrum_general.spectrum import element_wise
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class HR(_dyn_models):
    def __init__(self, I=3.5, eta1=0.001):
        super(HR, self).__init__()
        """
        This function initializes the various parameters of the Hindmarsh-Rose model. 
        See supplementary material for details.
        """
        """Strength of noise"""
        self.N = 3
        self.dim = self.N

        self._eta1 = eta1

        """Parameters"""
        self._b = 0.5
        self._mu = 0.01
        self._x_rest = -1.6
        self._s = 4

        """External stimulus"""
        self._I = I

        """Type of noise"""
        self._noise_type = "additive"
        # self._noise_type = "multiplicative"

        """Initialize the circuit"""
        self.initialize_circuit()

    @property
    def eta1(self):
        return self._eta1

    @eta1.setter
    def eta1(self, eta):
        if eta > 0:
            self._eta1 = eta
            self.make_noise_mats()
        else:
            raise ValueError("Noise S.D. must be a positive float")

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        if b > 0:
            self._b = b
            self.initialize_circuit()
        else:
            raise ValueError("b parameter be a positive float")

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        if mu > 0:
            self._mu = mu
            self.initialize_circuit()
        else:
            raise ValueError("Relative time constant must be a positive float")

    @property
    def x_rest(self):
        return self._x_rest

    @x_rest.setter
    def x_rest(self, x_rest):
        self._x_rest = x_rest
        self.initialize_circuit()

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, s):
        if s > 0:
            self._s = s
            self.initialize_circuit()
        else:
            raise ValueError("s parameter must be positive")

    @property
    def I(self):
        return self._I

    @I.setter
    def I(self, I):
        if I >= 0:
            self._I = I
            self.initialize_circuit()
        else:
            raise ValueError("External stimulus must be positive")

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
        return (self.eta1, self.b, self.mu, self.x_rest, self.s, self.I, self.noise_type)

    def initialize_circuit(self):
        """
        This function makes the jacobian and the noise matrix of the circuit.
        :return: None
        """
        """Make the jacobian"""
        time = 1000
        points = 10000
        _ = self.jacobian_autograd(time=time, points=points)

        """Make noise matrices"""
        self.make_noise_mats()
        return

    def make_noise_mats(self):
        """
        This function creates the noise matrices for the SDE simulation.
        :return: None
        """
        self.L = self.make_L(time=1000, points=10000)
        self.D = self.make_D()
        self.S = torch.sqrt(self.D)
        return

    def make_Ly(self, t, x):
        """
        This function creates the noise matrix.
        :param eta1: The strength (coefficient) of the noise in x.
        """
        L = torch.zeros((self.N, self.N))
        L[0, 0] = self.eta1
        return L

    def make_D(self):
        """
        This function creates the D matrix for the noise.
        :return: The D matrix
        """
        D = torch.eye(self.N)
        return D

    def psd_x(self, freq):
        """
        This function returns the psd of the x variable.
        :param x: The state of the circuit.
        :return: The psd of x
        """
        om = 2 * np.pi * freq

        """First we find the steady state"""
        sim_obj = sim_solution(self)
        time = 1000
        points = 10000
        ss = sim_obj.steady_state(time=time, points=points)
        x_e = ss[0]

        """The power spectral density of x"""
        numerator = self.eta1**2 * (self.mu**2 + (1 + self.mu**2) * om ** 2 + om ** 4)
        denominator = (self.mu ** 2) * ((x_e * (3 * x_e - 2 * self.b + 10) + self.s) ** 2)\
                    + ((self.mu ** 2) * (((x_e * (3 * x_e - 2 * self.b) + self.s) ** 2)
                    - 20 * x_e + 1) + (x_e ** 2) * ((3 * x_e - 2 * self.b + 10) ** 2) \
                    - 2 * self.mu * self.s + 20 * self.mu * self.s * x_e) * om ** 2 \
                    + (x_e * ((x_e * (3 * x_e - 2 * self.b) ** 2 - 20)) + self.mu ** 2
                    - 2 * self.mu * self.s + 1) * om ** 4 + om ** 6
        return numerator / denominator

    @dynm_fun
    def _dynamical_fun(self, t, xx):
        """
        This function defines the dynamics of the Hindmarsh-Rose model.
        :param x: The state of the circuit.
        :return: The derivative of the circuit at the current time-step.
        """
        xx = xx.squeeze(0)  # Remove the extra dimension during sde simulation
        x = xx[0:1]
        y = xx[1:2]
        z = xx[2:3]
        dxdt = y - x**3 + self.b * x**2 + self.I - z
        dydt = 1 - 5 * x**2 - y
        dzdt = self.mu * (self.s * (x - self.x_rest) - z)
        return torch.cat((dxdt, dydt, dzdt))


if __name__ == '__main__':
    pass
