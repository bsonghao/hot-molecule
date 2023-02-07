""" Module Description

Some explanation / notes should go here

"""

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

        self.H[(2, 0)] = QCP
        self.H[(0, 2)] = QCP

        print("number of vibrational mode {:}".format(self.N))
        print("##### Hamiltonian parameters ######")
        for rank in self.H.keys():
            print("Block {:}: \n {:}".format(rank, self.H[rank]))

        print("Boltzmann constant: {:} eV K-1".format(self.Kb))

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
                                "bb": np.einsum('i,j,ji->ij', self.sinh_theta, self.sinh_theta, self.H[(1, 1)])
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

    def _map_initial_amplitude(self, T_initial, mix_flag):
        """map initial T amplitude from Bose-Einstein statistics at high temperature"""
        def map_t_0_amplitude(beta):
            """map t_0 amplitude from H.O. partition function"""
            z = 1
            for i,w in enumerate(self.Freq):
                z *= 1 / (1 - np.exp(-beta * w))
            z *= np.exp(-beta * (self.H[(0, 0)]-sum(self.LCP**2 / self.Freq)/2.))
            if mix_flag:
                t_0 = z
            else:
                t_0 = np.log(z)
            return t_0

        def map_t1_amplitude():
            """
            map t_1 amplitude from linear coupling constant
            """
            # initialize t1 amplitude
            X_i = self.H[(1, 0)] / self.Freq
            t_1 = np.zeros(2 * N)
            t_1[:N] -= X_i / self.cosh_theta
            t_1[N:] -= X_i / self.sinh_theta

            return t_1

        def map_t2_amplitude():
            """map t_2 amplitude from cumulant expression of 2-RDM"""
            # initialize t2 amplitude
            t_2 = np.zeros([2*N, 2*N])

            # enter initial t_2 for ab block
            t_2[N:, :N] += np.diag((BE_occ - self.sinh_theta**2 - np.ones(N)) / self.cosh_theta / self.sinh_theta)
            t_2[:N, N:] += np.diag((BE_occ - self.cosh_theta**2) / self.cosh_theta / self.sinh_theta)

            # symmetrize t_2 amplitude
            t_2_new = np.zeros_like(t_2)
            for i, j in it.product(range(2*N), repeat=2):
                t_2_new[i, j] = 0.5 * (t_2[i, j] + t_2[j, i])

            return t_2_new

        N = self.N
        beta_initial = 1. / (self.Kb * T_initial)

        # calculation BE occupation number at initial beta
        BE_occ = np.ones(N) / (np.ones(N) - np.exp(-beta_initial * self.Freq))

        print("BE occupation number:{:}".format(BE_occ))

        initial_T_amplitude = {}
        initial_Z_amplitude = {}
        if not mix_flag:
            initial_T_amplitude[0] = map_t_0_amplitude(beta_initial)
            initial_T_amplitude[1] = map_t1_amplitude()
            initial_T_amplitude[2] = map_t2_amplitude()

            print("initial constant T ampltidue:\n{:}".format(initial_T_amplitude[0]))
            print("initial single T amplitude:\n{:}".format(initial_T_amplitude[1]))
            print("initial double T amplitude:\n{:}".format(initial_T_amplitude[2]))
        else:
            initial_Z_amplitude[0] = map_t_0_amplitude(beta_initial)
            initial_T_amplitude[1] = map_t1_amplitude()
            initial_Z_amplitude[2] = map_t2_amplitude()

            # set the result of the ampltiudes zeros
            initial_Z_amplitude[1] = np.zeros(2*N)


            print("initial single T amplitude:\n{:}".format(initial_T_amplitude[1]))
            print("initial constant Z ampltidue:\n{:}".format(initial_Z_amplitude[0]))
            print("initial double Z amplitude:\n{:}".format(initial_Z_amplitude[2]))

        return initial_T_amplitude, initial_Z_amplitude

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

    def CC_residue(self, H_args, T_args, Z_args=None, CI_flag=False, mix_flag=False, proj_flag=False):
        """implement coupled cluster residue equations"""
        N = self.N
        if proj_flag:
            T_proj = T_args[1]
        else:
            T_proj = None

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
            if not mix_flag:
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
            R = np.zeros(2*N)

            # linear
            R += H[(0, 1)]

            # quadratic
            R += np.einsum('ik,k->i', H[(0, 2)], T[1])

            return R

        def f_t_i(H, T, CI_flag=CI_flag, proj_flag=False, T_proj=None):
            """return residue R_i"""
            # initialize
            R = np.zeros(2*N)

            # non zero initial value of R
            if not CI_flag:
                R += H[(1, 0)]
            else:
                R += H[(1, 0)] * T[0]
                R += H[(0, 0)] * T[1]


            # linear
            R += np.einsum('ik,k->i', H[(1, 1)], T[1])

            # quadratic
            if not mix_flag:
                R += np.einsum('k,ki->i', H[(0, 1)], T[2])
            if not CI_flag and not mix_flag:
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
            R = np.zeros([2*N, 2*N])

            # first term
            R += H[(1, 1)]

            # quadratic
            if not mix_flag:
                R += np.einsum('ik,kj->ij', H[(0, 2)], T[2])

            return R

        def f_t_IJ(H, T):
            """return residue R_IJ"""

            # initialize as zero
            R = np.zeros([2*N, 2*N])

            # quadratic
            R += H[(0, 2)]

            return R

        def f_t_ij(H, T, CI_flag=CI_flag, proj_flag=False, T_proj=None):
            """return residue R_ij"""

            # # initialize as zero
            R = np.zeros([2*N, 2*N])

            # if self.hamiltonian_truncation_order >= 2:

            # quadratic
            if not CI_flag:
                R += H[(2, 0)]  # h term
            else:
                R += H[(2, 0)] * T[0]
                R += H[(0, 0)] * T[2]
                R += np.einsum('i,j->ij', H[(1, 0)], T[1])
                R += np.einsum('j,i->ij', H[(1, 0)], T[1])
            if not mix_flag:
                R += np.einsum('kj,ki->ij', H[(1, 1)], T[2])
                R += np.einsum('ki,kj->ij', H[(1, 1)], T[2])
            if not CI_flag and not mix_flag:
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

        if not mix_flag:
            residue = dict()

            residue[0] = f_t_0(H_args, T_args)
            residue[1] = f_t_i(H_args, T_args)
            residue[2] = f_t_ij(H_args, T_args)

            return residue
        else:
            # similarity transform the Hamiltonian
            sim_h = {}
            sim_h[(0, 0)] = f_t_0(H_args, T_args)
            sim_h[(0, 1)] = f_t_I(H_args, T_args)
            sim_h[(1, 0)] = f_t_i(H_args, T_args)
            sim_h[(1, 1)] = f_t_Ij(H_args, T_args)
            sim_h[(0, 2)] = f_t_IJ(H_args, T_args)
            sim_h[(2, 0)] = f_t_ij(H_args, T_args)

            # equate t_1 residue to (1, 0) block of the similairty transformed Hamiltonian
            t_residue = sim_h[(1, 0)].copy()

            # calculate net residue based on similairty transformed Hamiltnoian
            net_R_0 = f_t_0(sim_h, Z_args, CI_flag=True, proj_flag=proj_flag, T_proj=T_proj)
            net_R_1 = f_t_i(sim_h, Z_args, CI_flag=True, proj_flag=proj_flag, T_proj=T_proj)
            net_R_2 = f_t_ij(sim_h, Z_args, CI_flag=True, proj_flag=proj_flag, T_proj=T_proj)

            z_residue = {}

            # calculate constant Z residue

            # double Z residue
            z_residue[2] = net_R_2
            z_residue[2] -= np.einsum('i,j->ij', t_residue, Z_args[1])
            z_residue[2] -= np.einsum('j,i->ij', t_residue, Z_args[1])
            if proj_flag:
                X = np.einsum('k,k->', T_proj, t_residue)
                z_residue[2] -= X * Z_args[2]
                z_residue[2] -= np.einsum('k,i,kj->ij', T_proj, t_residue, Z_args[2])
                z_residue[2] -= np.einsum('k,j,ki->ij', T_proj, t_residue, Z_args[2])


            # single Z residue
            z_residue[1] = net_R_1
            z_residue[1] -= t_residue * Z_args[0]
            if proj_flag:
                z_residue[1] -= X * Z_args[1]
                z_residue[1] -= np.einsum('k,i,k->i', T_proj, t_residue, Z_args[1])
                z_residue[1] -= X * np.einsum('l,li->i', T_proj, Z_args[2])
                z_residue[1] -= np.einsum('k,ki->i', T_proj, z_residue[2])
                z_residue[1] -= 0.5 * np.einsum('k,l,i,kl->i', T_proj, T_proj, t_residue, Z_args[2])

            # constant Z residue
            z_residue[0] = net_R_0
            if proj_flag:
                z_residue[0] -= X * Z_args[0]
                z_residue[0] -= X * np.einsum('l,l->', T_proj, Z_args[1])
                z_residue[0] -= np.einsum('k,k->', T_proj, z_residue[1])
                z_residue[0] -= X * 0.5 * np.einsum('l,m,lm->', T_proj, T_proj, Z_args[2])
                z_residue[0] -= 0.5 * np.einsum('k,l,kl->', T_proj, T_proj, z_residue[2])

            return t_residue, z_residue, net_R_0

    def TFCC_integration(self, output_path, T_initial, T_final, num_step, CI_flag=False, mix_flag=False, proj_flag=False):
        """
        conduct thermal field coupled cluster imiginary time integration
        T_initial: initial temperature of TFCC propagation
        T_final: final temperature of TFCC propagation
        num_step: nunber of steps for numerical integration
        """

        # map initial amplitudes
        T_amplitude, Z_amplitude = self._map_initial_amplitude(T_initial=T_initial, mix_flag=mix_flag)

        # create temperature grid for integration
        ## initialize auto-correlation function as an array
        beta_initial = 1. / (self.Kb * T_initial)
        beta_final = 1. / (self.Kb * T_final)
        step = (beta_final - beta_initial) / num_step
        self.temperature_grid = 1. / (self.Kb * np.linspace(beta_initial, beta_final, num_step))

        # create empty list to store thermal properties
        self.partition_function = []
        self.internal_energy = []

        if not mix_flag:
            if CI_flag:
                T_amplitude[0] = 1.
            for i in range(num_step):
               # calculate CC residue
                residue = self.CC_residue(self.H_tilde_reduce, T_amplitude, mix_flag=mix_flag, CI_flag=CI_flag, proj_flag=proj_flag)
                # calculate thermal properties
                E = residue[0]

                if CI_flag:
                    Z = T_amplitude[0]
                else:
                    Z = np.exp(T_amplitude[0])

                self.internal_energy.append(E)
                self.partition_function.append(Z)
                # update T amplitude
                T_amplitude[0] -= step * residue[0]
                T_amplitude[1] -= step * residue[1]
                T_amplitude[2] -= step * residue[2]

                print("step {:}:".format(i))
                print("max t_0 amplitude:{:}".format(T_amplitude[0]))
                print("max t_1 amplitude:{:}".format(abs(T_amplitude[1]).max()))
                print("max t_2 amplitude:{:}".format(abs(T_amplitude[2]).max()))
                print("Temperature: {:} K".format(self.temperature_grid[i]))
                print("thermal internal energy: {:} ev".format(E))
                print("partition function: {:}".format(Z))

        else:
            for i in range(num_step):
                # calculate CC residue
                t_residue, z_residue, net_R_0 = self.CC_residue(self.H_tilde_reduce, T_amplitude, Z_amplitude, CI_flag=CI_flag, mix_flag=mix_flag, proj_flag=proj_flag)
                # calculate thermal propeties
                Z = Z_amplitude[0]
                E = z_residue[0] / Z_amplitude[0]
                self.partition_function.append(Z)
                self.internal_energy.append(E)
                # update T amplitude
                T_amplitude[1] -= step * t_residue
                # update Z amplitude
                Z_amplitude[0] -= step * z_residue[0]
                Z_amplitude[1] -= step * z_residue[1]
                Z_amplitude[2] -= step * z_residue[2]

                print("step {:}:".format(i))
                print("max z_0 amplitude:{:}".format(Z_amplitude[0]))
                print("max z_1 amplitude:{:}".format(abs(Z_amplitude[1]).max()))
                print("max z_2 amplitude:{:}".format(abs(Z_amplitude[2]).max()))
                print("max t_1 ampltiude:{:}".format(abs(T_amplitude[1]).max()))
                print("Temperature: {:} K".format(self.temperature_grid[i]))
                print("thermal internal energy: {:} ev-1".format(E))
                print("partition function: {:}".format(Z))

        # store data
        thermal_data = {"temperature": self.temperature_grid, "internal energy": self.internal_energy, "partition function": self.partition_function}
        df = pd.DataFrame(thermal_data)
        if mix_flag and not proj_flag:
            name = "thermal_data_TFCC_mix.csv"
        elif proj_flag:
            name = "thermal_data_TFCC_mix_proj.csv"
        else:
            name = "thermal_data_TFCC.csv"

        df.to_csv(output_path+name, index=False)
        return

    def sum_over_states(self, output_path, basis_size=40, T_initial=10000, T_final=100, num_step=10000):
        """calculation thermal properties through sum over states"""
        def construct_full_Hamitonian():
            """construct full Hamiltonian in H.O. basis"""
            Hamiltonian = np.zeros((basis_size, basis_size, basis_size, basis_size))
            for a_1 in range(basis_size):
                for a_2 in range(basis_size):
                    for b_1 in range(basis_size):
                        for b_2 in range(basis_size):
                            if a_1 == b_1 and a_2 == b_2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = self.H[(0, 0)]
                                Hamiltonian[a_1, a_2, b_1, b_2] += self.H[(1, 1)][0, 0]*(b_1)+self.H[(1, 1)][1, 1]*(b_2)
                            if a_1 == b_1+1 and a_2 == b_2-1:
                                Hamiltonian[a_1, a_2, b_1, b_2] = self.H[(1, 1)][0, 1]*np.sqrt(b_1+1)*np.sqrt(b_2)
                            if a_1 == b_1-1 and a_2 == b_2+1:
                                Hamiltonian[a_1, a_2, b_1, b_2] = self.H[(1, 1)][1, 0]*np.sqrt(b_1)*np.sqrt(b_2+1)
                            if a_1 == b_1+1 and a_2 == b_2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = self.H[(1, 0)][0]*np.sqrt(b_1+1)
                            if a_1 == b_1 and a_2 == b_2+1:
                                Hamiltonian[a_1, a_2, b_1, b_2] = self.H[(1, 0)][1]*np.sqrt(b_2+1)
                            if a_1 == b_1-1 and a_2 == b_2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = self.H[(0, 1)][0]*np.sqrt(b_1)
                            if a_1 == b_1 and a_2 == b_2-1:
                                Hamiltonian[a_1, a_2, b_1, b_2] = self.H[(0, 1)][1]*np.sqrt(b_2)
                            if a_1 == b_1+2 and a_2 == b_2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = self.H[(2, 0)][0, 0]*np.sqrt(b_1+1)*np.sqrt(b_1+2)
                            if a_1 == b_1+1 and a_2 == b_2+1:
                                Hamiltonian[a_1, a_2, b_1, b_2] = (self.H[(2, 0)][0, 1] + self.H[(2, 0)][1, 0])*np.sqrt(b_1+1)*np.sqrt(b_2+1)
                            if a_1 == b_1 and a_2 == b_2+2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = self.H[(2, 0)][1, 1]*np.sqrt(b_2+1)*np.sqrt(b_2+2)
                            if a_1 == b_1-2 and a_2 == b_2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = self.H[(0, 2)][0, 0]*np.sqrt(b_1)*np.sqrt(b_1-1)
                            if a_1 == b_1-1 and a_2 == b_2-1:
                                Hamiltonian[a_1, a_2, b_1, b_2] = (self.H[(0, 2)][0, 1] + self.H[(0, 2)][1, 0])*np.sqrt(b_1)*np.sqrt(b_2)
                            if a_1 == b_1 and a_2 == b_2-2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = self.H[(0, 2)][1, 1]*np.sqrt(b_2)*np.sqrt(b_2-1)

            Hamiltonian = Hamiltonian.reshape(basis_size**self.N, basis_size**self.N)

            return Hamiltonian

        def Cal_partition_function(E, T):
            """ compute partition function """
            Z = sum(np.exp(-E / (self.Kb * T)))
            return Z

        def Cal_thermal_internal_energy(E, T, Z):
            """ compute thermal_internal_energy """
            energy = sum(E * np.exp(-E / (self.Kb * T))) / Z
            return energy

        print("### Sum over states program starts! ###")
        beta_initial = 1. / (T_initial * self.Kb)
        beta_final = 1. / (T_final * self.Kb)
        T_grid = 1. / (self.Kb * np.linspace(beta_initial, beta_final, num_step))
        self.T_FCI = T_grid
        # contruct Hamiltonian in H.O. basis
        H = construct_full_Hamitonian()
        # check Hermicity of the Hamitonian in H. O. basis
        assert np.allclose(H, H.transpose())
        # diagonalize the Hamiltonian
        E, V = np.linalg.eigh(H)
        # store eigenvector and eigenvalues as class instances
        self.E_val = E
        self.V_val = V
        # calculate partition function
        partition_function = np.zeros_like(T_grid)
        for i, T in enumerate(T_grid):
            partition_function[i] = Cal_partition_function(E, T)

        # calculate thermal internal energy
        thermal_internal_energy = np.zeros_like(T_grid)
        for i, T in enumerate(T_grid):
            thermal_internal_energy[i] = Cal_thermal_internal_energy(E, T, partition_function[i])

        # store thermal data
        thermal_data = {"T(K)": T_grid, "partition function": partition_function, "internal energy": thermal_internal_energy}
        df = pd.DataFrame(thermal_data)
        df.to_csv(output_path+"thermal_data_FCI.csv", index=False)
        print("### sum over state program terminate normally ###")

        return
