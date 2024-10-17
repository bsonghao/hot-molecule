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
import itertools as it


# third party imports
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
# import matplotlib as mpl; mpl.use('pdf')
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

    def __init__(self, freq, LCP, QCP, VE, num_mode, temperature):
        """ initialize hamiltonian parameters:
        freq: vibrational frequencies
        LCP: linear coupling_constants
        QCP: quadratic coupling constant
        VE" vertical energy
        num_mode: number of vibration modes
        temperature: tempuration of the simulation
        """

        # initialize the Hamiltonian parameters as object instances
        self.N = num_mode
        self.Freq = freq
        self.LCP = LCP
        self.QCP = QCP
        self.VE = VE
        self.temperature = temperature

        # Boltzmann constant (eV K-1)
        self.Kb = 8.61733326e-5

        # define Hamiltonian obrect as a python dictionary where the keys are the rank of the Hamiltonian
        # and we represent the Hamitlnian in the form of second quantization
        self.H = dict()
        # constant
        self.H[(0, 0)] = VE + 0.5 * np.trace(QCP)
        # first order
        self.H[(1, 0)] = LCP / np.sqrt(2) * np.ones(self.N, dtype=complex)
        self.H[(0, 1)] = LCP / np.sqrt(2) * np.ones(self.N, dtype=complex)
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

        # Bogoliubov transform the Hamiltonian
        self.thermal_field_transformation(Temp=self.temperature)
        self.reduce_H_tilde()

        # initialize the vibrational Hamiltonian (for hot band)
        # initialize h_0
        self.h_0 = {
                    (0, 0): 0+0j,
                    (1, 1): np.zeros((self.N, self.N), dtype=complex),
                }

        # H.O. ground state energy
        self.h_0[(0, 0)] += 0.5 * sum(freq)

        # frequencies
        for j in range(self.N):
            self.h_0[(1, 1)][j, j] += freq[j]
        # calcuate Bogoliubov transformed h_0
        self.h_tilde_0 = dict()
        self.h_tilde_0[(0, 0)] = self.h_0[(0, 0)] + np.einsum('ii,i,i->', self.h_0[(1, 1)], self.sinh_theta, self.sinh_theta)

        self.h_tilde_0[(1, 1)] = {
        "aa": np.einsum('i,j,ij->ij', self.cosh_theta, self.cosh_theta, self.h_0[(1, 1)]),
        "ab": np.zeros((self.N, self.N), dtype=complex),
        'ba': np.zeros((self.N, self.N), dtype=complex),
        "bb": np.einsum('i,j,ji->ij', self.sinh_theta, self.sinh_theta, self.h_0[(1, 1)])
                   }

        self.h_tilde_0[(2, 0)] = {
        "aa": np.zeros((self.N, self.N), dtype=complex),
        "ab": np.einsum('i,j,ij->ij', self.cosh_theta, self.sinh_theta, self.h_0[(1, 1)]),
        "ba": np.einsum('i,j,ji->ij', self.sinh_theta, self.cosh_theta, self.h_0[(1, 1)]),
        "bb": np.zeros((self.N, self.N), dtype=complex)
        }

        self.h_tilde_0[(0, 2)] = {
        "aa": np.zeros((self.N, self.N), dtype=complex),
        "ab": np.einsum('i,j,ji->ij', self.cosh_theta, self.sinh_theta, self.h_0[(1, 1)]),
        "ba": np.einsum('i,j,ij->ij', self.sinh_theta, self.cosh_theta, self.h_0[(1, 1)]),
        "bb": np.zeros((self.N, self.N), dtype=complex)
        }
        # log.info("h_11 unmerged")
        # for key in self.h_tilde_0[(1, 1)].keys():
            # log.info("Block {:}: \n {:}".format(key, self.h_tilde_0[(1, 1)][key]))

        self.h_tilde_0[(1, 1)] = self.merge_quadratic(self.h_tilde_0[(1, 1)])
        self.h_tilde_0[(2, 0)] = self.merge_quadratic(self.h_tilde_0[(2, 0)])
        self.h_tilde_0[(0, 2)] = self.merge_quadratic(self.h_tilde_0[(0, 2)])


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
        self.H_tilde[(0, 0)] = self.H[(0, 0)] + np.einsum('ii,i,i->', self.H[(1, 1)], self.sinh_theta, self.sinh_theta)

        # linear terns
        self.H_tilde[(1, 0)] = {
                               "a": self.cosh_theta * self.H[(1, 0)],
                               "b": self.sinh_theta * self.H[(0, 1)]
                               }

        self.H_tilde[(0, 1)] = {
                               "a": self.cosh_theta * self.H[(0, 1)],
                               "b": self.sinh_theta * self.H[(1, 0)]
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
                                "ba": np.einsum('i,j,ji->ij', self.sinh_theta, self.cosh_theta, self.H[(1, 1)]),
                                "bb": np.einsum('i,j,ij->ij', self.sinh_theta, self.sinh_theta, self.H[(0, 2)]),
                               }

        self.H_tilde[(0, 2)] = {
                                "aa": np.einsum('i,j,ij->ij', self.cosh_theta, self.cosh_theta, self.H[(0, 2)]),
                                "ab": np.einsum('i,j,ji->ij', self.cosh_theta, self.sinh_theta, self.H[(1, 1)]),
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

    def merge_linear(self, input_tensor):
        """ merge linear terms of the Hamiltonian """
        N = self.N
        output_tensor = np.zeros(2 * N, dtype=complex)
        output_tensor[:N] = input_tensor['a'].copy()
        output_tensor[N:] = input_tensor['b'].copy()

        return output_tensor

    def merge_quadratic(self, input_tensor):
        """ merge quadratic_terms of the Hamiltonian """
        N = self.N
        output_tensor = np.zeros([2*N,  2*N], dtype=complex)
        output_tensor[:N, :N] = input_tensor["aa"].copy()
        output_tensor[:N, N:] = input_tensor["ab"].copy()
        output_tensor[N:, :N] = input_tensor["ba"].copy()
        output_tensor[N:, N:] = input_tensor["bb"].copy()

        return output_tensor


    def reduce_H_tilde(self):
        """merge the a, b blocks of the Bogliubov transformed Hamiltonian into on tensor"""
        N = self.N
        # initialize
        self.H_tilde_reduce = {
            (0, 0): self.H_tilde[(0, 0)],
            }


        # merge linear terms
        self.H_tilde_reduce[(1, 0)] = self.merge_linear(self.H_tilde[(1, 0)])
        self.H_tilde_reduce[(0, 1)] = self.merge_linear(self.H_tilde[(0, 1)])
        self.H_tilde_reduce[(1, 1)] = self.merge_quadratic(self.H_tilde[(1, 1)])
        self.H_tilde_reduce[(2, 0)] = self.merge_quadratic(self.H_tilde[(2, 0)])
        self.H_tilde_reduce[(0, 2)] = self.merge_quadratic(self.H_tilde[(0, 2)])

        print("##### Bogliubov transformed (fictitous) Hamiltonian after merge blocks ######")
        for rank in self.H_tilde_reduce.keys():
            print("Block {:}: \n {:}".format(rank, self.H_tilde_reduce[rank]))

        return


    def CC_residue(self, H_args, T_args, CI_flag=False, mix_flag=False, proj_flag=False):
        """implement coupled cluster residue equations"""
        N = 2 * self.N
        if proj_flag:
            T_proj = np.conjugate(T_args[1])
        else:
            T_proj=None

        def f_t_0(H, T, CI_flag=CI_flag, proj_flag=False, T_proj=None):
            """return residue R_0"""

            # initialize as zero
            R = 0.

            # constant
            if not CI_flag:
                R += H[(0, 0)]
            else:
                R += H[(0, 0)] * T[0]

            # linear
            R += np.einsum('k,k->', H[(0, 1)], T[1])

            # quadratic
            R += 0.5 * np.einsum('kl,kl->', H[(0, 2)], T[2])

            if not CI_flag:
                R += 0.5 * np.einsum('kl,k,l->', H[(0, 2)], T[1], T[1])

            if proj_flag:

                R += np.einsum('i, i ->', T_proj, T[1]) * H[(0, 0)]

                R += (1 / 2) * np.einsum('i, j, ij ->', T_proj, T_proj, T[2]) * H[(0, 0)]

                R += np.einsum('i, i ->', T_proj, H[(1, 0)]) * T[0]

                R += (
                    np.einsum('i, ij, j ->', T_proj, H[(1, 1)], T[1]) +
                    np.einsum('i, j, j, i ->', T_proj, T_proj, H[(1, 0)], T[1])
                     )

                R += (
                    np.einsum('i, j, ij ->', T_proj, H[(0, 1)], T[2]) +
                    np.einsum('i, j, jk, ik ->', T_proj, T_proj, H[(1, 1)], T[2])
                     )
                R += (1 / 2) * np.einsum('i, j, k, k, ij ->', T_proj, T_proj, T_proj, H[(1, 0)], T[2])

                R += (1 / 2) * np.einsum('i, j, ij ->', T_proj, T_proj, H[(2, 0)]) * T[0]

                R += (1 / 2) * np.einsum('i, j, k, jk, i ->', T_proj, T_proj, T_proj, H[(2, 0)], T[1])

                R += (1 / 4) * np.einsum('i, j, k, l, kl, ij ->', T_proj, T_proj, T_proj, T_proj, H[(2, 0)], T[2])


            return R

        def f_t_I(H, T):
            """return residue R_I"""

            # initialize as zero
            R = np.zeros(N, dtype=complex)

            # linear
            R += H[(0, 1)]

            # quadratic
            R += np.einsum('ik,k->i', H[(0, 2)], T[1])

            return R

        def f_t_i(H, T, CI_flag=CI_flag, proj_flag=False, T_proj=None):
            """return residue R_i"""
            # initialize
            R = np.zeros(N, dtype=complex)

            # non zero initial value of R
            if not CI_flag:
                R += H[(1, 0)]
            else:
                R += H[(1, 0)] * T[0]
                R += H[(0, 0)] * T[1]


            # linear
            R += np.einsum('ik,k->i', H[(1, 1)], T[1])

            # quadratic
            R += np.einsum('k,ki->i', H[(0, 1)], T[2])
            if not CI_flag:
                R += np.einsum('kl,k,li->i', H[(0, 2)], T[1], T[2])

            if proj_flag:

                R += np.einsum('i, iz -> z', T_proj, T[2]) * H[(0, 0)]

                R += (
                    np.einsum('i, z, i -> z', T_proj, H[(1, 0)], T[1]) +
                    np.einsum('i, i, z -> z', T_proj, H[(1, 0)], T[1])
                     )

                R += (
                    np.einsum('i, ij, jz -> z', T_proj, H[(1, 1)], T[2]) +
                    np.einsum('i, jz, ij -> z', T_proj, H[(1, 1)], T[2]) +
                    np.einsum('i, j, j, iz -> z', T_proj, T_proj, H[(1, 0)], T[2])
                     )
                R += (1 / 2) * np.einsum('i, j, z, ij -> z', T_proj, T_proj, H[(1, 0)], T[2])

                R += np.einsum('i, iz -> z', T_proj, H[(2, 0)]) * T[0]

                R += np.einsum('i, j, jz, i -> z', T_proj, T_proj, H[(2, 0)], T[1])

                R += (1 / 2) * np.einsum('i, j, ij, z -> z', T_proj, T_proj, H[(2, 0)], T[1])

                R += (1 / 2) * (
                            np.einsum('i, j, k, jk, iz -> z', T_proj, T_proj, T_proj, H[(2, 0)], T[2]) +
                            np.einsum('i, j, k, kz, ij -> z', T_proj, T_proj, T_proj, H[(2, 0)], T[2])
                            )

            return R

        def f_t_Ij(H, T):
            """return residue R_Ij"""

            # initialize
            R = np.zeros([N, N], dtype=complex)

            # first term
            R += H[(1, 1)]

            # quadratic
            R += np.einsum('ik,kj->ij', H[(0, 2)], T[2])

            return R

        def f_t_IJ(H, T):
            """return residue R_IJ"""

            # initialize as zero
            R = np.zeros([N, N], dtype=complex)

            # quadratic
            R += H[(0, 2)]

            return R

        def f_t_ij(H, T, CI_flag=CI_flag, proj_flag=False, T_proj=None):
            """return residue R_ij"""
            # # initialize as zero
            R = np.zeros([N, N], dtype=complex)

            # if self.hamiltonian_truncation_order >= 2:

            # quadratic
            if not CI_flag:
                R += H[(2, 0)]  # h term
            else:
                R += H[(2, 0)] * T[0]
                R += H[(0, 0)] * T[2]
                R += np.einsum('i,j->ij', H[(1, 0)], T[1])
                R += np.einsum('j,i->ij', H[(1, 0)], T[1])

            R += np.einsum('jk,ki->ij', H[(1, 1)], T[2])
            R += np.einsum('ik,kj->ij', H[(1, 1)], T[2])
            if not CI_flag:
                R += 0.5 * np.einsum('kl,ki,lj->ij', H[(0, 2)], T[2], T[2])
                R += 0.5 * np.einsum('kl,kj,li->ij', H[(0, 2)], T[2], T[2])

            if proj_flag:
                R += np.einsum('i, z, iy -> zy', T_proj, H[(1, 0)], T[2])
                R += np.einsum('i, y, iz -> zy', T_proj, H[(1, 0)], T[2])

                R += np.einsum('i, i, zy -> zy', T_proj, H[(1, 0)], T[2])

                R += np.einsum('i, iz, y -> zy', T_proj, H[(2, 0)], T[1])
                R += np.einsum('i, iy, z -> zy', T_proj, H[(2, 0)], T[1])

                R += np.einsum('i, zy, i -> zy', T_proj, H[(2, 0)], T[1])

                R += np.einsum('i, j, jz, iy -> zy', T_proj, T_proj, H[(2, 0)], T[2])
                R += np.einsum('i, j, jy, iz -> zy', T_proj, T_proj, H[(2, 0)], T[2])


                R += (1 / 2) * (
                        np.einsum('i, j, zy, ij -> zy', T_proj, T_proj, H[(2, 0)], T[2]) +
                        np.einsum('i, j, ij, zy -> zy', T_proj, T_proj, H[(2, 0)], T[2])
                    )
            return R

        # similarity transfrom the Hamiltonian
        sim_h = {}
        sim_h[(0, 0)] = f_t_0(H_args, T_args)
        sim_h[(0, 1)] = f_t_I(H_args, T_args)
        sim_h[(1, 0)] = f_t_i(H_args, T_args)
        sim_h[(1, 1)] = f_t_Ij(H_args, T_args)
        sim_h[(0, 2)] = f_t_IJ(H_args, T_args)
        sim_h[(2, 0)] = f_t_ij(H_args, T_args)

        return sim_h


        # if not mix_flag:
            # residue = dict()

            # residue[0] = f_t_0(H_args, T_args)
            # residue[1] = f_t_i(H_args, T_args)
            # residue[2] = f_t_ij(H_args, T_args)

            # return residue
        # else:
            # similarity transform the Hamiltonian
            # sim_h = {}
            # sim_h[(0, 0)] = f_t_0(H_args, T_args)
            # sim_h[(0, 1)] = f_t_I(H_args, T_args)
            # sim_h[(1, 0)] = f_t_i(H_args, T_args)
            # sim_h[(1, 1)] = f_t_Ij(H_args, T_args)
            # sim_h[(0, 2)] = f_t_IJ(H_args, T_args)
            # sim_h[(2, 0)] = f_t_ij(H_args, T_args)

            # equate t_1 residue to (1, 0) block of the similairty transformed Hamiltonian
            # t_residue = sim_h[(1, 0)].copy()

            # calculate net residue based on similairty transformed Hamiltnoian
            # net_R_0 = f_t_0(sim_h, Z_args, CI_flag=True, proj_flag=proj_flag, T_proj=T_proj)
            # net_R_1 = f_t_i(sim_h, Z_args, CI_flag=True, proj_flag=proj_flag, T_proj=T_proj)
            # net_R_2 = f_t_ij(sim_h, Z_args, CI_flag=True, proj_flag=proj_flag, T_proj=T_proj)

            # z_residue = {}

            # calculate constant Z residue

            # double Z residue
            # z_residue[2] = net_R_2
            # z_residue[2] -= np.einsum('i,j->ij', t_residue, Z_args[1])
            # z_residue[2] -= np.einsum('j,i->ij', t_residue, Z_args[1])
            # if proj_flag:
                # X = np.einsum('k,k->', T_proj, t_residue)
                # z_residue[2] -= X * Z_args[2]
                # z_residue[2] -= np.einsum('k,i,kj->ij', T_proj, t_residue, Z_args[2])
                # z_residue[2] -= np.einsum('k,j,ki->ij', T_proj, t_residue, Z_args[2])


            # single Z residue
            # z_residue[1] = net_R_1
            # z_residue[1] -= t_residue * Z_args[0]
            # if proj_flag:
                # z_residue[1] -= X * Z_args[1]
                # z_residue[1] -= np.einsum('k,i,k->i', T_proj, t_residue, Z_args[1])
                # z_residue[1] -= X * np.einsum('l,li->i', T_proj, Z_args[2])
                # z_residue[1] -= np.einsum('k,ki->i', T_proj, z_residue[2])
                # z_residue[1] -= 0.5 * np.einsum('k,l,i,kl->i', T_proj, T_proj, t_residue, Z_args[2])

            # constant Z residue
            # z_residue[0] = net_R_0
            # if proj_flag:
                # z_residue[0] -= X * Z_args[0]
                # z_residue[0] -= X * np.einsum('l,l->', T_proj, Z_args[1])
                # z_residue[0] -= np.einsum('k,k->', T_proj, z_residue[1])
                # z_residue[0] -= X * 0.5 * np.einsum('l,m,lm->', T_proj, T_proj, Z_args[2])
                # z_residue[0] -= 0.5 * np.einsum('k,l,kl->', T_proj, T_proj, z_residue[2])

            # return t_residue, z_residue


    def VECC_integration(self, t_final, num_steps, CI_flag=False, mix_flag=False, proj_flag=False):
        """ conduct VECC integration """
        dtau = t_final / num_steps
        # initialize auto-correlation function as an array
        time = np.linspace(0., t_final, num_steps+1)
        ACF = np.zeros(num_steps+1, dtype=complex)
        if not mix_flag:
            # initialize T amplitude as zeros
            T_amplitude = {
                       0: 0.,
                       1: np.zeros(2*self.N, dtype=complex),
                       2: np.zeros([2*self.N, 2*self.N], dtype=complex)
            }

            if CI_flag:
                T_amplitude[0] = 1.
            for i in range(num_steps+1):
                # calculate ACF
                if CI_flag:
                    ACF[i] = T_amplitude[0]
                else:
                    ACF[i] = np.exp(T_amplitude[0])
                # calculate CC residue
                args = ( self.H_tilde_reduce,
                T_amplitude,
                CI_flag, mix_flag,
                proj_flag
                )
                sim_trans_h = self.CC_residue(*args)
                # update T amplitude
                T_amplitude[0] -= dtau * (sim_trans_h[(0, 0)] - self.h_tilde_0[(0, 0)]) * 1j
                T_amplitude[1] -= dtau * sim_trans_h[(1, 0)] * 1j
                T_amplitude[2] -= dtau * (sim_trans_h[(2, 0)] - self.h_tilde_0[(2, 0)]) * 1j

                print("time:{:} ACF:{:}".format(time[i], ACF[i]))

        else:
            T_amplitude = {
                       0: 0.,
                       1: np.zeros(self.N, dtype=complex),
                       2: np.zeros([self.N, self.N], dtype=complex)
            }
            Z_amplitude = {
                       0: 1.,
                       1: np.zeros(self.N, dtype=complex),
                       2: np.zeros([self.N, self.N], dtype=complex)
            }
            for i in range(num_steps+1):
                # calculate ACF
                ACF[i] = Z_amplitude[0]
                # calculate CC residue
                args = ( self.H_tilde_reduce,
                T_amplitude, Z_amplitude,
                CI_flag, mix_flag,
                proj_flag
                )
                t_residue, z_residue = self.CC_residue(*args)
                # update T amplitude
                T_amplitude[1] -= dtau * t_residue * 1j
                # update Z amplitude
                Z_amplitude[0] -= dtau * z_residue[0] * 1j
                Z_amplitude[1] -= dtau * z_residue[1] * 1j
                Z_amplitude[2] -= dtau * z_residue[2] * 1j
                print("time:{:} ACF:{:}".format(time[i], ACF[i]))

        return time, ACF

    def FCI_solution(self, time, basis_size):
        """ calculate ACF from exact diagonalization of the full Hamiltonian """
        def cal_boltz_factor():
            """calculate Boltzmann factor"""
            energy = np.zeros((basis_size, basis_size), dtype=complex)
            for m,n in it.product(range(basis_size), repeat=2):
                energy[m, n] = self.Freq[0]*m + self.Freq[1]*n + 0.5 * sum(self.Freq)
            energy = energy.reshape(basis_size**self.N)
            beta = 1. / (self.Kb * self.temperature)
            factor = np.exp(-beta*energy)
            factor /= sum(factor)
            return factor, energy

        print('### FCI program start ###')
        # initialize Hamiltonian in FCI basis
        model = model_two_mode(self.H[(0, 0)].real, self.H[(1, 0)].real, self.H[(1, 1)].real, self.H[(0, 2)].real)
        self.H_FCI = model.sos_solution(basis_size=basis_size)
        # check hermicity of Hamitlnian
        assert np.allclose(self.H_FCI, self.H_FCI.transpose())
        # diagonalize the full Hamiltonian
        E, V = np.linalg.eigh(self.H_FCI)
        # print('Energy Eigenvalue')
        # for i in range(10):
            # print(E[i])
        Pn, E_0 = cal_boltz_factor()
        # initilize auto correlation function
        ACF = np.zeros_like(time, dtype=complex)
        # compute ACF
        for n,l in it.product(range(basis_size**self.N), repeat=2):
            ACF += Pn[n]*V[(n, l)] * np.exp(-(E[l]-E_0[n]) * time * 1j) * V[(n, l)].conjugate()

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
