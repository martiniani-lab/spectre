"""We define the class for simulating the Stabilized-supralinear-network (SSN) model."""
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import functorch as ftorch
from model_classes.dyn_models import dyn_models
from util.util_funs import dynm_fun
import scipy.signal
from torchdiffeq import odeint
from torchsde import sdeint
import matplotlib.pyplot as plt
from util.simulation_class import SDE
from spectrum_general.matrix_spectrum import matrix_solution
from spectrum_general.sim_spectrum import sim_solution
from spectrum_general.spectrum import PSD
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class SSN_new(dyn_models):
    def __init__(self, N=11, length=9, c=50, eta=0.01):
        super(SSN_new, self).__init__()
        """
        This function initializes the various parameters of the SSN model. See 
        supplementary material for details.
        """
        self._N = N # Number of units in the network
        self.dim = 2 * self._N
        self._eta = eta # Strength of the noise
        self._tauE = 0.006
        self._tauI = 0.004

        """Separation between units"""
        self._delta_x = 3.0

        """Parameters of the weights"""
        self._J_EE = 2.0
        self._J_IE = 2.25
        self._J_EI = 0.9
        self._J_II = 0.5

        """Sigma parameters"""
        self.sigma_RF = None
        self._sigma_EE = 4
        self._sigma_IE = 8

        """Stimulus length and contrast"""
        self._l = length
        self._c = c

        """Supralinear activation parameters"""
        self._k = 0.01
        self._n = 2.2

        """Make the input stimulus"""
        self.input = None

        """Make the weight matrices"""
        self.W_EE = None
        self.W_IE = None
        self.W_EI = None
        self.W_II = None

        """Type of noise"""
        self._noise_type = "additive"

        """Initialize the circuit"""
        self.initialize_circuit()
        self.make_noise_mats()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        if N > 0:
            self._N = N
            self.dim = self.calculate_dim()
            self.initialize_circuit()
            self.make_noise_mats()
        else:
            raise ValueError("N must be a positive integer")

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, eta):
        if eta >= 0:
            self._eta = eta
            self.make_noise_mats()
        else:
            raise ValueError("Noise S.D. must be a positive float")

    @property
    def tauE(self):
        return self._tauE

    @tauE.setter
    def tauE(self, tauE):
        if tauE > 0:
            self._tauE = tauE
            self.initialize_circuit()
            self.make_noise_mats()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def tauI(self):
        return self._tauI

    @tauI.setter
    def tauI(self, tauI):
        if tauI > 0:
            self._tauI = tauI
            self.initialize_circuit()
            self.make_noise_mats()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def delta_x(self):
        return self._delta_x

    @delta_x.setter
    def delta_x(self, delta_x):
        if delta_x > 0:
            self._delta_x = delta_x
            self.initialize_circuit()
        else:
            raise ValueError("delta_x must be a positive float")

    @property
    def J_EE(self):
        return self._J_EE

    @J_EE.setter
    def J_EE(self, J_EE):
        self._J_EE = J_EE
        self.initialize_circuit()

    @property
    def J_IE(self):
        return self._J_IE

    @J_IE.setter
    def J_IE(self, J_IE):
        self._J_IE = J_IE
        self.initialize_circuit()

    @property
    def J_EI(self):
        return self._J_EI

    @J_EI.setter
    def J_EI(self, J_EI):
        self._J_EI = J_EI
        self.initialize_circuit()

    @property
    def J_II(self):
        return self._J_II

    @J_II.setter
    def J_II(self, J_II):
        self._J_II = J_II
        self.initialize_circuit()

    @property
    def sigma_EE(self):
        return self._sigma_EE

    @sigma_EE.setter
    def sigma_EE(self, sigma_EE):
        if sigma_EE > 0:
            self._sigma_EE = sigma_EE
            self.initialize_circuit()
        else:
            raise ValueError("Sigma must be a positive float")

    @property
    def sigma_IE(self):
        return self._sigma_IE

    @sigma_IE.setter
    def sigma_IE(self, sigma_IE):
        if sigma_IE > 0:
            self._sigma_IE = sigma_IE
            self.initialize_circuit()
        else:
            raise ValueError("Sigma must be a positive float")

    @property
    def l(self):
        return self._l

    @l.setter
    def l(self, l):
        if l > 0:
            self._l = l
            self.initialize_circuit()
        else:
            raise ValueError("Length of stimulus must be a positive float")

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c):
        if c > 0:
            self._c = c
            self.initialize_circuit()
        else:
            raise ValueError("Contrast of stimulus must be a positive float")

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        if k > 0:
            self._k = k
            self.initialize_circuit()
        else:
            raise ValueError("Supralinear activation parameter must be a positive float")

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        if n > 1:
            self._n = n
            self.initialize_circuit()
        else:
            raise ValueError("Sublinear activation must be larger than 1")

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
        return (self.N, self.eta, self.tauE, self.tauI, self.delta_x, self.J_EE,
                self.J_IE, self.J_EI, self.J_II, self.sigma_EE, self.sigma_IE,
                self.l, self.c, self.k, self.n)

    def calculate_dim(self):
        """
        This function calculates the dimension of the network
        """
        return self._N * 2

    @staticmethod
    def make_weight_matrix(N, J, delta_x, sigma):
        """
        This function generates the weight matrix for the network.
        :param N: The number of units in the network
        :param J: The strength of the weights
        :param delta_x: Distance separation between units
        :param sigma: The spread of the weights
        :return: The N x N weight matrix
        """
        x = delta_x * torch.linspace(1, N, N) - (N+1)/2
        x = torch.reshape(x, (N, 1))
        W = J * SSN_new.gaussian(x, x.T, sigma)
        return W

    @staticmethod
    def gaussian(x, y, sigma):
        """
        This function creates the gaussian function.
        """
        d = torch.abs(x - y)
        return torch.exp(-torch.pow(d / sigma, 2) / 2)

    @staticmethod
    def check_eig(mat):
        """Checks if all the eigenvalues of a matrix have real part<0, ei, they describe
         a stable dynamical system."""
        return torch.all(torch.real(torch.linalg.eigvals(mat)) < 0)

    def initialize_circuit(self):
        """
        This function makes the input stimulus, the weight matrices and the jacobian
        corresponding to the system.
        :return: None
        """
        self.sigma_RF = 0.125 * self._delta_x

        """Make the input stimulus"""
        self.input = self.make_input()

        """Make the weight matrices"""
        self.W_EI = self.J_EI * torch.eye(self.N)
        self.W_II = self.J_II * torch.eye(self.N)
        self.W_EE = self.make_weight_matrix(self.N, self.J_EE, self.delta_x, self.sigma_EE)
        self.W_IE = self.make_weight_matrix(self.N, self.J_IE, self.delta_x, self.sigma_IE)

        """Make the jacobian"""
        time = 1
        points = 10000
        _ = self.jacobian_autograd(time=time, points=points)
        return

    def make_input(self):
        """
        This function generates the input stimulus for the network.
        :param N: The number of units in the network
        :param delta_x: Distance separation between units
        :param sigma: The spread of the stimulus
        :param l: The length of the stimulus
        :param c: The contrast of the stimulus
        :param theta: The orientation angle of the stimulus. Doesn't require theta now
        but in future maybe (to change stimulus position in space).
        :return: the N dimensional input stimulus.
        """
        N = self.N; delta_x = self.delta_x; sigma = self.sigma_RF; l = self.l; c = self.c
        x = (torch.linspace(1, N, N) - (N+1)/2 ) * delta_x
        input = c * (1 / (1 + torch.exp(- (x + l/2) / sigma))) * (1 - 1 / (1 + torch.exp(- (x - l/2) / sigma)))
        return input

    def make_Ly(self, t, x):
        """
        This function defines the most general form of the noise matrix.
        :param N: The number of units in the network.
        :param eta: The strength (coefficient) of the noise.
        :return: The N x N noise matrix.
        """
        return self.eta * torch.eye(self.dim)

    def make_D(self):
        """
        This function creates the D matrix for the noise.
        :return: The D matrix
        """
        D = torch.zeros(self.dim)
        D[0:self.N] = 1/self.tauE
        D[self.N:] = 1/self.tauI
        return torch.diag(D)

    @dynm_fun
    def _dynamical_fun(self, t, x):
        """
        This function defines the dynamics of the SSN model.
        :param x: The state of the network.
        :return: The derivative of the network at the current time-step.
        """
        x = x.squeeze(0) # Remove the extra dimension during sde simulation
        E = x[0:self.N]
        I = x[self.N:2*self.N]
        inputE = self.input + torch.matmul(self.W_EE, E) - torch.matmul(self.W_EI, I)
        inputI = self.input + torch.matmul(self.W_IE, E) - torch.matmul(self.W_II, I)
        dEdt = (1 / self.tauE) * (-E + self.k * (F.relu(inputE))**self.n)
        dIdt = (1 / self.tauI) * (-I + self.k * (F.relu(inputI))**self.n)
        return torch.cat((dEdt, dIdt))


if __name__ == '__main__':
    # create the network
    net = SSN_new()
    t = torch.linspace(0, 1, 10000)
    sim = net.simulate(t)
    # plot the results
    plt.plot(t, sim[:, 0:net.N])
    plt.show()