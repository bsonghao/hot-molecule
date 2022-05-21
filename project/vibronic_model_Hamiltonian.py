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
project_dir = abspath(join(dirname(__file__), '/home/paulie/hot-molecule'))
sys.path.insert(0, project_dir)
from project.two_mode_model import model_two_mode


class vibronic_model_hamiltonian(object):
    """ vibronic model hamiltonian class implement TF-VECC approach to simulation thermal properties of vibronic models. """

    def __init__(self, freq, LCP, QCP, VE, num_mode):
        """ initialize hamiltonian parameters:
        freq: vibrational frequencies
        LCP: linear coupling_constants
        QCP: quadratic coupling constant
        VE" vertical energy
        num_mode: number of vibration modes
        """

        # initialize the Hamiltonian parameters as object instances
        self.N = num_mode
        self.Freq = freq
        self.LCP = LCP
        self.QCP = QCP
        self.VE = VE

        # Boltzmann constant (eV K-1)
        self.Kb = 8.61733326e-5

        # define Hamiltonian obrect as a python dictionary where the keys are the rank of the Hamiltonian
        # and we represent the Hamitlnian in the form of second quantization
        self.H = dict()
        # constant
        self.H[(0, 0)] = VE + 0.5 * np.trace(QCP)
        # first order
        self.H[(1, 0)] = LCP / np.sqrt(2) * np.ones(self.N)
        self.H[(0, 1)] = LCP / np.sqrt(2) * np.ones(self.N)
        # second order
        self.H[(1, 1)] = np.diag(freq)
        self.H[(1, 1)] += QCP

        self.H[(2, 0)] = QCP / 2
        self.H[(0, 2)] = QCP / 2

        print("number of vibrational mode {:}".format(self.N))
        print("##### Hamiltonian parameters ######")
        for rank in self.H.keys():
            print("Block {:}: \n {:}".format(rank, self.H[rank]))

        print("Boltzmann constant: {:} eV K-1".format(self.Kb))

        # initialize Hamiltonian in FCI basis
        model = model_two_mode(self.H[(0, 0)], self.H[(1, 0)], self.H[(1, 1)], self.H[(0, 2)])
        self.H_FCI = model.sos_solution(basis_size=40)

        print("### End of Hamiltonian parameters ####")

    def thermal_field_transformation(self, Temp):
        """conduct Bogoliubov transformation of input Hamiltonian and determine thermal field reference state"""
        # calculate inverse temperature
        beta = 1. / (self.Kb * Temp)
        # define Bogliubov transformation based on Bose-Einstein statistics
        self.cosh_theta = 1. / np.sqrt((np.ones(self.N) - np.exp(-beta * self.Freq)))
        self.sinh_theta = np.exp(-beta * self.Freq / 2.) / np.sqrt(np.ones(self.N) - np.exp(-beta * self.Freq))

        # Bogliubov tranform that Hamiltonian
        self.H_tilde = dict()

        # constant term???
        self.H_tilde[(0, 0)] = self.H[(0, 0)] + sum(self.sinh_theta**2)

        # linear terns
        self.H_tilde[(1, 0)] = {
                               "a": self.cosh_theta * self.H[(1, 0)],
                               "b": self.sinh_theta * self.H[(0, 1)]
                               }

        self.H_tilde[(0, 1)] = {
                               "a": self.sinh_theta * self.H[(0, 1)],
                               "b": self.cosh_theta * self.H[(1, 0)]
                               }

        # quadratic terms
        self.H_tilde[(1, 1)] = {
                                "aa": np.einsum('i,j,ij->ij', self.cosh_theta, self.cosh_theta, self.H[(1, 1)]),
                                "ab": np.einsum('i,j,ij->ij', self.cosh_theta, self.sinh_theta, self.H[(2, 0)]),
                                "ba": np.einsum('i,j,ij->ij', self.sinh_theta, self.cosh_theta, self.H[(0, 2)]),
                                "bb": np.einsum('i,j,ij->ij', self.sinh_theta, self.sinh_theta, self.H[(1, 1)])
                               }

        self.H_tilde[(2, 0)] = {
                                "aa": np.einsum('i,j,ij->ij', self.cosh_theta, self.cosh_theta, self.H[(2, 0)]),
                                "ab": np.einsum('i,j,ij->ij', self.cosh_theta, self.sinh_theta, self.H[(1, 1)]),
                                "ba": np.zeros_like(self.H[2, 0]),
                                "bb": np.einsum('i,j,ij->ij', self.sinh_theta, self.sinh_theta, self.H[(0, 2)]),
                               }

        self.H_tilde[(0, 2)] = {
                                "aa": np.einsum('i,j,ij->ij', self.cosh_theta, self.cosh_theta, self.H[(0, 2)]),
                                "ab": np.zeros_like(self.H[(0, 2)]),
                                "ba": np.einsum('i,j,ij->ij', self.sinh_theta, self.cosh_theta, self.H[(1, 1)]),
                                "bb": np.einsum('i,j,ij->ij', self.sinh_theta, self.sinh_theta, self.H[(2, 0)])
                               }

        print("###### Bogliubov transformed Hamiltonian ########")
        for rank in self.H_tilde.keys():
            if rank == (0, 0):
                print("Rank:{:}\n{:}".format(rank, self.H_tilde[rank]))
            else:
                for block in self.H_tilde[rank].keys():
                    print("Rank:{:} Block:{:}\n{:}".format(rank, block, self.H_tilde[rank][block]))

        return

    def _map_initial_T_amplitude(self, T_initial):
        """map initial T amplitude from Bose-Einstein statistics at high temperature"""
        def map_t1_amplitude():
            """map t_1 amplitude from linear coupling constant"""
            # initialize t1 amplitude
            t_1 = np.zeros(2 * N)
            t_1[:N] += self.LCP / np.sqrt(2) / (2 * self.cosh_theta)
            t_1[N:] += self.LCP / np.sqrt(2) / (2 * self.sinh_theta)

            return t_1

        def map_t2_amplitude(RDM_2, t1):
            """map t_2 amplitude from cumulant expression of 2-RDM"""
            # initialize t2 amplitude
            t_2 = np.zeros([2 * N, 2 * N])
            t_2[N:, N:] += RDM_2
            t_2 -= np.einsum('p,q->pq', t1, t1)

            return t_2

        N = self.N
        beta_initial = 1. / (self.Kb * T_initial)
        # calculate two particle density matrice from Bose-Einstein statistics
        two_RDM = np.diag(np.exp(beta_initial * self.Freq))

        initial_T_amplitude = {}
        initial_T_amplitude["t1"] = map_t1_amplitude()
        initial_T_amplitude["t2"] = map_t2_amplitude(two_RDM, initial_T_amplitude['t1'])

        print("initial single T amplitude:\n{:}".format(initial_T_amplitude["t1"]))
        print("initial double T amplitude:\n{:}".format(initial_T_amplitude["t2"]))

        return initial_T_amplitude

    def reduce_H_tilde(self):
        """merge the a, b blocks of the Bogliubov transformed Hamiltonian into on tensor"""
        N = self.N
        # initialize
        self.H_tilde_reduce = {
            (0, 0): self.H_tilde[(0, 0)],
            }

        def merge_linear(input_tensor):
            """ merge linear terms of the Hamiltonian """
            output_tensor = np.zeros(2 * N)
            output_tensor[:N] = input_tensor['a'].copy()
            output_tensor[N:] = input_tensor['b'].copy()

            return output_tensor

        def merge_quadratic(input_tensor):
            """ merge quadratic_terms of the Hamiltonian """
            output_tensor = np.zeros([2 * N, 2 * N])
            output_tensor[:N, :N] = input_tensor["aa"].copy()
            output_tensor[:N, N:] = input_tensor["ab"].copy()
            output_tensor[N:, :N] = input_tensor["ba"].copy()
            output_tensor[N:, N:] = input_tensor["bb"].copy()

            return output_tensor

        # merge linear terms
        self.H_tilde_reduce[(1, 0)] = merge_linear(self.H_tilde[(1, 0)])
        self.H_tilde_reduce[(0, 1)] = merge_linear(self.H_tilde[(0, 1)])
        self.H_tilde_reduce[(1, 1)] = merge_quadratic(self.H_tilde[(1, 1)])
        self.H_tilde_reduce[(2, 0)] = merge_quadratic(self.H_tilde[(2, 0)])
        self.H_tilde_reduce[(0, 2)] = merge_quadratic(self.H_tilde[(0, 2)])

        print("##### Bogliubov transformed (fictitous) Hamiltonian after merge blocks ######")
        for rank in self.H_tilde_reduce.keys():
            print("Block {:}: \n {:}".format(rank, self.H_tilde_reduce[rank]))

        return


    def CC_residue(self, H_args, T_args):
        """implement coupled cluster residue equations"""
        N = self.N

        def f_t_0(H, T):
            """return residue R_0"""

            # initialize as zero
            R = 0.

            # constant
            R += H[(0, 0)]

            # linear
            R += np.einsum('k,k->', H[(0, 1)], T[1])

            # quadratic
            R += 0.5 * np.einsum('kl,kl->', H[(2, 0)], T[2])
            R += 0.5 * np.einsum('kl,k,l->', H[(2, 0)], T[1], T[1])

            return R

        def f_t_I(H, T):
            """return residue R_I"""

            # initialize as zero
            R = np.zeros(N)

            # linear
            R += H[(0, 1)]

            # quadratic
            R += np.einsum('ik,k->i', H[(0, 2)], T[1])

            return R

        def f_t_i(H, T):
            """return residue R_i"""

            # initialize
            R = np.zeros(N, dtype=complex)

            # non zero initial value of R
            R += H[(1, 0)]

            # linear
            R += np.einsum('ik,k->i', H[(1, 1)], T[1])

            # quadratic
            R += np.einsum('k,ki->i', H[(0, 1)], T[2])
            R += np.einsum('kl,k,li->i', H[(0, 2)], T[1], T[2])

            return R

        def f_t_Ij(H, T):
            """return residue R_Ij"""

            # initialize
            R = np.zeros([N, N])

            # first term
            R += H[(1, 1)]

            # quadratic
            R += np.einsum('ik,kj->ij', H[(0, 2)], T[2])

            return R

        def f_t_IJ(H, T):
            """return residue R_IJ"""

            # initialize as zero
            R = np.zeros([N, N])

            # quadratic
            R += H[(0, 2)]
            return R

        def f_t_ij(H, T):
            """return residue R_ij"""

            # # initialize as zero
            R = np.zeros([N, N], dtype=complex)

            # if self.hamiltonian_truncation_order >= 2:

            # quadratic
            R += H[(2, 0)]  # h term
            R += np.einsum('jk,ki->ij', H[(1, 1)], T[2])
            R += np.einsum('ik,kj->ij', H[(1, 1)], T[2])
            R += 0.5 * np.einsum('kl,ki,lj->ij', H[(0, 2)], T[2], T[2])
            R += 0.5 * np.einsum('kl,kj,li->ij', H[(0, 2)], T[2], T[2])
            return R

        # compute similarity transformed Hamiltonian over e^T
        # sim_h = {}
        # sim_h[(0, 0)] = f_t_0(H_args, t_args)
        # sim_h[(0, 1)] = f_t_I(H_args, t_args)
        # sim_h[(1, 0)] = f_t_i(H_args, t_args)
        # sim_h[(1, 1)] = f_t_Ij(H_args, t_args)
        # sim_h[(0, 2)] = f_t_IJ(H_args, t_args)
        # sim_h[(2, 0)] = f_t_ij(H_args, t_args)

        residue = dict()

        residue[0] = f_t_0(H_args, T_args)
        residue[1] = f_t_i(H_args, T_args)
        residue[2] = f_t_ij(H_args, T_args)

        return residue

    def VECC_integration(self, t_final, num_steps):
        """ conduct VECC integration """
        dtau = t_final / num_steps
        # initialize auto-correlation function as an array
        time = np.linspace(0., t_final, num_steps+1)
        ACF = np.zeros(num_steps+1, dtype=complex)
        # initialize T amplitude as zeros
        T_amplitude = {
                   0: 0.,
                   1: np.zeros(self.N, dtype=complex),
                   2: np.zeros([self.N, self.N], dtype=complex)
        }
        for i in range(num_steps+1):
            # calculate ACF
            ACF[i] = np.exp(T_amplitude[0])
            # calculate CC residue
            residue = self.CC_residue(self.H, T_amplitude)
            # update T amplitude
            T_amplitude[0] -= dtau * residue[0] * 1j
            T_amplitude[1] -= dtau * residue[1] * 1j
            T_amplitude[2] -= dtau * residue[2] * 1j

            print("time:{:} ACF:{:}".format(time[i], ACF[i]))

        return time, ACF

    def FCI_solution(self, time):
        """ calculate ACF from exact diagonalization of the full Hamiltonian """
        print('### FCI program start ###')
        # check hermicity of Hamitlnian
        assert np.allclose(self.H_FCI, self.H_FCI.transpose())
        # diagonalize the full Hamiltonian
        E, V = np.linalg.eigh(self.H_FCI)
        print('Energy Eigenvalue')
        for i in range(10):
            print(E[i])

        # initilize auto correlation function
        ACF = np.zeros_like(time, dtype=complex)
        # compute ACF
        for n in range(40**self.N):
            ACF += V[(0, n)] * np.exp(-E[n] * time * 1j) * V[(0, n)].conjugate()

        # normalize ACF
        ACF /= ACF[0]

        print('### End of Sum Over States Program ###')
        return time, ACF

    def store_ACF_data(self, time, ACF, name):
        """ store ACF data in a format that adapt with autospec """
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


    def TFCC_integration(self):
        """conduct TFCC imaginary time integration to calculation thermal perperties"""
        return
