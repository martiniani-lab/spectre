"""This function calculates the auto and cross spectrum using simulation"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from torchsde import sdeint
from spectre.utils.simulation_class import SDE, SDE_mul, SDE_cor_mul
import scipy.signal


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_default_dtype(torch.float64)


class sim_solution:
    def __init__(self, obj=None):
        """
        This function initializes the nonlinear dynamical function and noise matrices
        required for SDE simulation.
        :param obj: Dynamical function object
        """
        """Object containing dynamical function to be used for simulation"""
        self.obj = obj
        self.noise_type = obj.noise_type

    def steady_state(self, time=1, points=10000, tol=1e-5, atol=1e-5, rtol=1e-5):
        """
        This function calculates the steady-state of the function by simulating the
        dynamical system over time.
        :return: The steady-state circuit. Raises an error if steady-state is not found.
        """
        t = torch.linspace(0, time, points)
        sim = self.simulate(t, atol, rtol)
        # find if steady state is reached in a given tolerance else raise exception
        if torch.all(torch.abs(sim[-1, :] - sim[-2, :]) < tol):
            return sim[-1, :]
        else:
            raise Exception("Steady state not found")

    def simulate(self, t, atol=1e-5, rtol=1e-5):
        """
        This function simulates the circuit dynamics over time.
        :param t: The time tensor to simulate over.
        :return: The state of the circuit over time
        """
        y0 = torch.zeros(self.obj.dim) + 1e-2
        return odeint(self.obj._dynamical_fun, y0, t, method='euler', atol=atol, rtol=rtol)

    def simulate_sde(self, t, n_points=int(1e5), time=10, dt=1e-4, save=True):
        """
        This function simulates the dynamical function using the SDE solver.
        :param t: The time at which the simulation is to be done.
        :param save: This param is true when you want to look from the saved simulations.
        :return: The simulation of the circuit.
        """
        key = self.obj.get_instance_variables() + (n_points, time, dt, self.noise_type, save)
        if key not in self.obj.simulation:
            if hasattr(self.obj, "steady_state"):
                x0 = self.obj.steady_state().unsqueeze(0)
            else:
                x0 = self.steady_state(time=time/10, points=n_points//10).unsqueeze(0)
            if self.noise_type == "additive" or self.noise_type == "cor_add":
                sde = SDE(self.obj)
            elif self.noise_type == "multiplicative":
                sde = SDE_mul(self.obj)
            elif self.noise_type == "cor_mul":
                sde = SDE_cor_mul(self.obj)
            else:
                raise Exception("Noise type not recognized")
            with torch.no_grad():
                sol = sdeint(sde, x0, t, dt=dt, method='euler')
            solution = sol.squeeze(1)
            if save:
                self.obj.simulation[key] = solution
        else:
            solution = self.obj.simulation[key]
        return solution

    @staticmethod
    def spectrum(i=None, j=None, nperseg=1000, sampling_freq=torch.tensor([1e4]), sol_sde=None):
        sol_sde = sol_sde - torch.mean(sol_sde, dim=0)
        sol_i = sol_sde[:, i]

        if j is None:
            freq, S = scipy.signal.welch(sol_i.numpy(), sampling_freq.numpy(),
                                         nperseg=nperseg, scaling='density',
                                         return_onesided=False)
        else:
            sol_j = sol_sde[:, j]
            freq, S = scipy.signal.csd(sol_i.numpy(), sol_j.numpy(),
                                       sampling_freq.numpy(), nperseg=nperseg,
                                       scaling='density', return_onesided=False)

        f = freq[freq >= 0]
        S = S[freq >= 0]
        return torch.from_numpy(S), torch.from_numpy(f)

    @staticmethod
    def coh_spectrum(i=None, j=None, nperseg=1000, sampling_freq=1e4, sol_sde=None):
        sol_sde = sol_sde - torch.mean(sol_sde, dim=0)
        sol_i = sol_sde[:, i]
        sol_j = sol_sde[:, j]

        freq, S = scipy.signal.coherence(sol_i.numpy(), sol_j.numpy(),
                                         sampling_freq.numpy(), nperseg=nperseg,
                                         return_onesided=False)

        f = freq[freq >= 0]
        S = S[freq >= 0]
        return torch.from_numpy(S), torch.from_numpy(f)

    def simulation_spectrum(self, i=None, j=None, ndivs=15, n_points=int(1e5),
                            time=10, dt=1e-4):
        """
        This function calculates the power spectral density of the ith variable
        or cross-power spectral density between the i and j variables using simulation
        of the circuit with noise.
        :param i: First index of the variable.
        :param j: Second index of the variable.
        :return: The power spectral density or cross-power spectral density between the
        ith and jth variable.
        """
        i = 0 if i is None else i
        t = torch.linspace(0, time, n_points)

        sol_sde = self.simulate_sde(t, n_points, time, dt)

        nperseg = n_points // ndivs
        sampling_freq = 1 / (t[1] - t[0])
        S, f = self.spectrum(i=i, j=j, sol_sde=sol_sde,
                             nperseg=nperseg, sampling_freq=sampling_freq)
        return S, f

    def simulation_coherence(self, i=None, j=None, ndivs=15, n_points=int(1e5),
                             time=10, dt=1e-4):
        """
        This function calculates the coherence between the ith and jth variables,
        using simulation of the circuit with noise.
        :param i: The index of the first variable.
        :param j: The index of the second variable.
        :return: The coherence: |Sxy|^2 / (Sxx * Syy), between variables i and j.
        """
        i = 0 if i is None else i
        j = 1 if j is None else j

        Sxy, f = self.simulation_spectrum(i=i, j=j, ndivs=ndivs, n_points=n_points,
                                          time=time, dt=dt)
        Sxx, _ = self.simulation_spectrum(i=i, j=None, ndivs=ndivs, n_points=n_points,
                                          time=time, dt=dt)
        Syy, _ = self.simulation_spectrum(i=j, j=None, ndivs=ndivs, n_points=n_points,
                                          time=time, dt=dt)

        coherence = (torch.abs(Sxy) ** 2) / (Sxx * Syy)
        return coherence, f


if __name__ == '__main__':
    pass
