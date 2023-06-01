"""We define the class for simulating the Wilson-Cowan 4D model."""
import numpy as np
import torch
import torch.nn as nn
import functorch as ftorch
from model.dyn_models import dyn_models
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


class WC4D(dyn_models):
    def __init__(self, c=0.5, eta1=0.001, eta2=0.002):
        super(WC4D, self).__init__()
        """
        This function initializes the various parameters of the 4D Wilson-Cowan model. 
        See supplementary material for details.
        """
        """Strength of noise"""
        self.N = 4
        self.dim = self.N
        self._eta1 = eta1
        self._eta2 = eta2

        """Time constants"""
        self._tauE = 0.002
        self._tauI = 0.008
        self._tauSE = 0.01
        self._tauSI = 0.01

        """Weights"""
        self._W_EE = 5.0
        self._W_EI = 5.0
        self._W_IE = 3.5
        self._W_II = 3.0
        
        """More parameters"""
        self._gammaE = 1.0
        self._gammaI = 2.0
        self._thetaE = 0.4
        self._thetaI = 0.4
        self._kappaE = 0.2
        self._kappaI = 0.02

        """Stimulus contrast"""
        self._c = c

        """Type of noise"""
        self._noise_type = "additive"

        """Inputs"""
        self.IE = None
        self.II = None
        self.sE = None
        self.sI = None

        """Initialize the circuit"""
        self.initialize_circuit()

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
    def tauE(self):
        return self._tauE

    @tauE.setter
    def tauE(self, tauE):
        if tauE > 0:
            self._tauE = tauE
            self.initialize_circuit()
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
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def tauSE(self):
        return self._tauSE

    @tauSE.setter
    def tauSE(self, tauSE):
        if tauSE > 0:
            self._tauSE = tauSE
            self.initialize_circuit()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def tauSI(self):
        return self._tauSI

    @tauSI.setter
    def tauSI(self, tauSI):
        if tauSI > 0:
            self._tauSI = tauSI
            self.initialize_circuit()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def W_EE(self):
        return self._W_EE

    @W_EE.setter
    def W_EE(self, W_EE):
        if W_EE > 0:
            self._W_EE = W_EE
            self.initialize_circuit()
        else:
            raise ValueError("Weight must be a positive float")

    @property
    def W_IE(self):
        return self._W_IE

    @W_IE.setter
    def W_IE(self, W_IE):
        if W_IE > 0:
            self._W_IE = W_IE
            self.initialize_circuit()
        else:
            raise ValueError("Weight must be a positive float")

    @property
    def W_EI(self):
        return self._W_EI

    @W_EI.setter
    def W_EI(self, W_EI):
        if W_EI > 0:
            self._W_EI = W_EI
            self.initialize_circuit()
        else:
            raise ValueError("Weight must be a positive float")

    @property
    def W_II(self):
        return self._W_II

    @W_II.setter
    def W_II(self, W_II):
        if W_II > 0:
            self._W_II = W_II
            self.initialize_circuit()
        else:
            raise ValueError("Weight must be a positive float")

    @property
    def gammaE(self):
        return self._gammaE

    @gammaE.setter
    def gammaE(self, gammaE):
        if gammaE > 0:
            self._gammaE = gammaE
            self.initialize_circuit()
        else:
            raise ValueError("Gamma must be a positive float")

    @property
    def gammaI(self):
        return self._gammaI

    @gammaI.setter
    def gammaI(self, gammaI):
        if gammaI > 0:
            self._gammaI = gammaI
            self.initialize_circuit()
        else:
            raise ValueError("Gamma must be a positive float")

    @property
    def thetaE(self):
        return self._thetaE

    @thetaE.setter
    def thetaE(self, thetaE):
        if thetaE > 0:
            self._thetaE = thetaE
            self.initialize_circuit()
        else:
            raise ValueError("Theta must be a positive float")

    @property
    def thetaI(self):
        return self._thetaI

    @thetaI.setter
    def thetaI(self, thetaI):
        if thetaI > 0:
            self._thetaI = thetaI
            self.initialize_circuit()
        else:
            raise ValueError("Theta must be a positive float")

    @property
    def kappaE(self):
        return self._kappaE

    @kappaE.setter
    def kappaE(self, kappaE):
        if kappaE > 0:
            self._kappaE = kappaE
            self.initialize_circuit()
        else:
            raise ValueError("Kappa must be a positive float")

    @property
    def kappaI(self):
        return self._kappaI

    @kappaI.setter
    def kappaI(self, kappaI):
        if kappaI > 0:
            self._kappaI = kappaI
            self.initialize_circuit()
        else:
            raise ValueError("Kappa must be a positive float")

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c):
        if c > 0:
            self._c = c
            self.initialize_circuit()
        else:
            raise ValueError("contrast must be a positive float")

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
        return (self.eta1, self.eta2, self.tauE, self.tauI, self.tauSE, self.tauSI,
                self.W_EE, self.W_EI, self.W_IE, self.W_II, self.gammaE,
                self.gammaI, self.thetaE, self.thetaI, self.kappaE, self.kappaI)

    @staticmethod
    def sigmoid(x):
        """
        This function creates the sigmoid function.
        """
        return 1 / (1 + torch.exp(-x))

    def initialize_circuit(self):
        """
        This function makes the input, the jacobian and the noise matrix of the circuit.
        :return: None
        """
        """Make the input"""
        self.make_input()

        """Make the jacobian"""
        time = 10
        points = 100000
        _ = self.jacobian_autograd(time=time, points=points)

        """Make noise matrices"""
        self.make_noise_mats()
        return

    def make_input(self):
        """
        This function generates the input stimulus for the circuit.
        :param c: The contrast of the stimulus
        :return: the set of inputs for the circuit.
        """
        self.IE = 2 * self.c
        self.II = self.IE / 2
        self.sE = 0.2
        self.sI = 0.05
        return

    def make_Ly(self, t, x):
        """
        This function creates the L matrix for the noise.
        :param t: The time
        :param x: The state of the circuit
        :return: The L matrix
        """
        L = torch.zeros(self.N, self.N)
        L[0, 0] = self.eta1
        L[1, 1] = self.eta1
        L[2, 2] = self.eta2
        L[3, 3] = self.eta2
        return L

    def make_D(self):
        """
        This function creates the D matrix for the noise.
        :return: The D matrix
        """
        D = torch.zeros(self.N, self.N)
        D[0, 0] = 1 / self.tauE
        D[1, 1] = 1 / self.tauI
        D[2, 2] = 1 / self.tauSE
        D[3, 3] = 1 / self.tauSI
        return D

    @dynm_fun
    def _dynamical_fun(self, t, x):
        """
        This function defines the dynamics of the Wilson-Cowan 4D circuit.
        :param x: The state of the circuit.
        :return: The derivative of the circuit at the current time-step.
        """
        x = x.squeeze(0) # Remove the extra dimension during sde simulation
        # write the code below by slicing the tensor x
        E = x[0:1]
        I = x[1:2]
        SE = x[2:3]
        SI = x[3:4]
        dEdt = (-E + WC4D.sigmoid((self.IE + self.W_EE * SE - self.W_EI * SI -
                                   self.thetaE) / self.kappaE)) / self.tauE
        dIdt = (-I + WC4D.sigmoid((self.II + self.W_IE * SE - self.W_II * SI -
                                   self.thetaI) / self.kappaI)) / self.tauI
        dSEdt = (-SE + self.gammaE * E * (1 - SE) + self.sE) / self.tauSE
        dSIdt = (-SI + self.gammaI * I * (1 - SI) + self.sI) / self.tauSI
        return torch.cat((dEdt, dIdt, dSEdt, dSIdt))


if __name__ == '__main__':
    # create the circuit
    net = WC4D()
    t = torch.linspace(0, 1, 10000)
    sim = net.simulate(t)
    # plot the results
    plt.plot(t, sim[:, 0:net.N])
    plt.show()