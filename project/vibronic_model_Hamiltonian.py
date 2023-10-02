""" Module Description

Some explanation / notes should go here

"""


# system imports
# system imports
import io
import time
import os
from os.path import abspath, join, dirname, basename
import sys
import cProfile
import pstats


# third party imports
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
import matplotlib as mpl; mpl.use('pdf')
import matplotlib.pyplot as plt
import parse  # used for loading data files
import pandas as pd
#
import opt_einsum as oe

# local imports
# import the path to the package
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/hot-molecule/'))
sys.path.insert(0, project_dir)
from project.two_mode_model import model_two_mode


class vibronic_model_hamiltonian(object):
    """ vibronic model hamiltonian class implement TF-VECC approach to simulation thermal properties of vibronic models. """

    def __init__(self, freq, LCP, QCP, VE, TDM, num_mode, num_surface):
        """
        initialize hamiltonian parameters:
        freq: vibrational frequencies
        LCP: linear coupling_constants
        QCP: quadratic coupling constant
        VE" vertical energy
        TDM: transition dipole moments
        num_mode: number of vibration modes
        num_surface: number of electronic surfaces
        """

        # initialize the Hamiltonian parameters as object instances
        self.A = num_surface
        self.N = num_mode
        self.Freq = freq
        self.LCP = LCP
        self.QCP = QCP
        self.VE = VE
        self.TDM = TDM

        # Boltzmann constant (eV K-1)
        self.Kb = 8.61733326e-5

        # convert unit of energy
        self.hbar_per_eV = 6.582119569e-16
        self.s_per_fs = 1e-15
        self.unit = self.s_per_fs / self.hbar_per_eV

        # define Hamiltonian obrect as a python dictionary where the keys are the rank of the Hamiltonian
        # and we represent the Hamitlnian in the form of second quantization
        self.H = dict()
        # constant
        self.H[(0, 0)] = VE
        # first order
        self.H[(1, 0)] = LCP
        self.H[(0, 1)] = LCP
        # second order
        self.H[(1, 1)] = 4 * QCP
        for a in range(self.A):
            self.H[(1, 1)][a, a, :] += 4 * np.diag(self.Freq)

        self.H[(2, 0)] = 2 * QCP
        self.H[(0, 2)] = 2 * QCP

        print("number of vibrational mode {:}".format(self.N))
        print("##### Hamiltonian parameters ######")
        for rank in self.H.keys():
            print("Block {:}: \n {:}".format(rank, self.H[rank]))

        print("Boltzmann constant: {:} eV K-1".format(self.Kb))

        print("### End of Hamiltonian parameters ####")

    def construct_full_Hamiltonian_in_HO_basis(self, basis_size=10):
        """construct matrix elements for the full Hamiltonian in H.O, basis"""
        print("### construct FCI Hamiltonian ####")
        # initialize the FCI Hamiltonian
        H_FCI = np.zeros([self.A, basis_size ** self.N, self.A, basis_size ** self.N])
        # construct matrix elements block by block
        for a in range(self.A):
            for b in range(self.A):
                model = model_two_mode(self.H[(0, 0)][a, b], self.H[(1, 0)][a, b, :], self.H[(1, 1)][a, b, :], self.H[(0, 2)][a, b, :])
                H_FCI[a, :, b][:] = model.sos_solution(basis_size=basis_size).copy()
        self.H_FCI = H_FCI.reshape(self.A * basis_size**self.N, self.A * basis_size**self.N)
        print("### Check hermicity of the full Hamiltonian in FCI basis")
        assert np.allclose(self.H_FCI, self.H_FCI.transpose())
        print("### diagnalize H and store its eigenvalues and eigenvectors")
        self.E, self.V = np.linalg.eigh(self.H_FCI)
        print("### FCI Hamiltonian successfully constructed ###")
        return
    def calculate_ACF_from_FCI(self, time, basis_size, name):
        """calculation time-correlation from FCI"""
        # compute ACF form FCI
        E, V = self.E, self.V
        unit = self.unit
        ACF = np.zeros_like(time, dtype=complex)
        for c in range(self.A):
             for d in range(self.A):
                for n in range(self.A * basis_size**self.N):
                    ACF += self.TDM[c] * V[(basis_size ** self.N * c, n)] * np.exp(-E[n] * time * unit * 1j) * V[(basis_size ** self.N * d, n)] * self.TDM[d]
        # normalize time-correlation Function
        ACF /= ACF[0]
        data = {'time': time, 'Re': ACF.real, 'Im': ACF.imag, 'Abs': abs(ACF)}
        df = pd.DataFrame(data)

        # store ACF data to csv format
        df.to_csv(name+".csv", index=False)

        # store ACF data in autospec format
        with open(name+".txt", 'w') as file:
            file.write('#    time[fs]         Re(autocorrel)     Im(autocorrel)     Abs(autocorrel)\n')
            tmp = df.to_records(index=False)
            for t, Re, Im, Abs in tmp:
                x1 = '{:.{}f}'.format(t, 8)
                x2, x3, x4 = ['{:.{}f}'.format(e, 14) for e in (Re, Im, Abs)]
                string = '{:>15} {:>22} {:>18} {:>18}'.format(x1, x2, x3, x4)
                file.write(string+'\n')

        return

    def calculate_state_pop_from_FCI(self, time, basis_size):
        """calculate state population as a function of time from FCI"""
        # unit converstion
        unit = self.unit
        print("### compute state population from FCI ###")
        def Cal_state_pop(E, V, b, time, basis_size=10):
            pop = np.zeros([self.A, len(time)], dtype=complex)
            V = V.reshape([self.A, basis_size, basis_size, self.A * basis_size**self.N])
            for a in range(self.A):
                for l in range(len(E)):
                    for n_1 in range(basis_size):
                        for n_2 in range(basis_size):
                            pop[a, :] += np.exp(-1j * E[l] * time * unit) * V[a, n_1, n_2, l] * V[b, 0, 0, l]
            return abs(pop)**2 / sum(abs(pop)**2)

        E, V = self.E, self.V
        print("###construct state population one surface at a time ###")
        state_pop = np.zeros([self.A, self.A, len(time)])
        for b in range(self.A):
            state_pop[b, :] = Cal_state_pop(E, V, b, time, basis_size=basis_size)
        print("### store state pululation data to csv ###")
        # store state pop data
        for b in range(self.A):
            temp = {"time": time}
            for a in range(self.A):
                temp[str(a)] = state_pop[b, a, :]
            df = pd.DataFrame(temp)
            name = "state_pop_for_surface_{:}_from_FCI.csv".format(b)
            df.to_csv(name, index=False)
        print("### state population is successfully computed from FCI")

        return
