""" Module Description

Some explanation / notes should go here

"""


# system imports
import os
from os.path import join
import itertools as it
import types

# third party imports
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
# import matplotlib as mpl; mpl.use('pdf')
import matplotlib.pyplot as plt
import parse  # used for loading data files
import pandas as pd
import itertools as it

#
import opt_einsum as oe


class vibronic_model_hamiltonian(object):
    """ vibronic model hamiltonian class implement TNOE approach to simulation thermal properties of vibronic models. """

    def __init__(self, freq, LCP, VE, num_mode, num_surf, FC=False):
        """ initialize hamiltonian parameters:
        freq: vibrational frequencies
        LCP: linear coupling_constants
        VE: vertical energy
        num_mode: number of vibrational modes
        num_surf: number of electronic surfaces
        """

        # initialize the Hamiltonian parameters as object instances
        self.A = num_surf
        self.N = num_mode
        self.Freq = freq
        self.LCP = LCP
        self.VE = VE
        self.FC = FC

        # Boltzmann constant (eV K-1)
        self.Kb = 8.61733326e-5

        # define Hamiltonian object as a python dictionary where the keys are the rank of the Hamiltonian
        # and we represent the Hamitlnian in the form of second quantization
        self.H = dict()
        # constant
        self.H[(0, 0)] = np.zeros([self.A, self.A])
        for i in range(self.A):
            self.H[(0, 0)][i, i] = VE[i, i] + 0.5 * sum(freq)
        # first order
        self.H[(1, 0)] = np.array(LCP).transpose(1, 2, 0) / np.sqrt(2)
        self.H[(0, 1)] = np.array(LCP).transpose(1, 2, 0) / np.sqrt(2)

        # exclude off-diagonal elements if FC flag is True
        if self.FC:
            for a, b in it.product(range(self.A), repeat=2):
                if a != b:
                    self.H[(1, 0)][a, b, :] = np.zeros(self.N)
                    self.H[(0, 1)][a, b, :] = np.zeros(self.N)

        # second order
        self.H[(1, 1)] = np.zeros([self.A, self.A, self.N, self.N])
        for a in range(self.A):
            for i in range(self.N):
                self.H[(1, 1)][a, a, i, i] = freq[i]

        self.H[(2, 0)] = np.zeros([self.A, self.A, self.N, self.N])
        self.H[(0, 2)] = np.zeros([self.A, self.A, self.N, self.N])

        print("number of vibrational mode {:}".format(self.N))
        print("##### Hamiltonian parameters ######")
        for rank in self.H.keys():
            print("Block {:}:".format(rank))
            if rank != (0, 0):
                for a, b in it.product(range(self.A), repeat=2):
                    print("surface ({:}, {:}):\n{:}".format(a, b, self.H[rank][a, b, :]))
            else:
                print(self.H[rank])


        print("Boltzmann constant: {:} eV K-1".format(self.Kb))

        print("### End of Hamiltonian parameters ####")

    def thermal_field_transform(self, T_ref):
        """
        conduct Bogoliubov transfrom of the physical hamiltonian
        T_ref: temperature for the thermal field reference state
        """
        # calculate inverse temperature
        self.T_ref = T_ref
        beta = 1. / (self.Kb * T_ref)
        # define Bogliubov transformation based on Bose-Einstein statistics
        self.cosh_theta = 1. / np.sqrt((np.ones(self.N) - np.exp(-beta * self.Freq)))
        self.sinh_theta = np.exp(-beta * self.Freq / 2.) / np.sqrt(np.ones(self.N) - np.exp(-beta * self.Freq))

        # Bogliubov tranform that Hamiltonian
        self.H_tilde = dict()

        # constant terms
        self.H_tilde[(0, 0)] = self.H[(0, 0)] + np.einsum('abii,i,i->ab', self.H[(1, 1)], self.sinh_theta, self.sinh_theta)

        # linear termss
        self.H_tilde[(1, 0)] = {
                               "a": np.einsum('i,abi->abi', self.cosh_theta, self.H[(1, 0)]),
                               "b": np.einsum('i,abi->abi', self.sinh_theta, self.H[(0, 1)])
                               }

        self.H_tilde[(0, 1)] = {
                               "a": np.einsum('i,abi->abi', self.cosh_theta, self.H[(0, 1)]),
                               "b": np.einsum('i,abi->abi', self.sinh_theta, self.H[(1, 0)])
                               }

        # quadratic terms
        self.H_tilde[(1, 1)] = {
                                "aa": np.einsum('i,j,abij->abij', self.cosh_theta, self.cosh_theta, self.H[(1, 1)]),
                                "ab": np.einsum('i,j,abij->abij', self.cosh_theta, self.sinh_theta, self.H[(2, 0)]),
                                "ba": np.einsum('i,j,abij->abij', self.sinh_theta, self.cosh_theta, self.H[(0, 2)]),
                                "bb": np.einsum('i,j,abji->abij', self.sinh_theta, self.sinh_theta, self.H[(1, 1)])
                               }

        self.H_tilde[(2, 0)] = {
                                "aa": np.einsum('i,j,abij->abij', self.cosh_theta, self.cosh_theta, self.H[(2, 0)]),
                                "ab": np.einsum('i,j,abij->abij', self.cosh_theta, self.sinh_theta, self.H[(1, 1)]),
                                "ba": np.einsum('i,j,abji->abij', self.sinh_theta, self.cosh_theta, self.H[(1, 1)]),
                                "bb": np.einsum('i,j,abij->abij', self.sinh_theta, self.sinh_theta, self.H[(0, 2)]),
                               }

        self.H_tilde[(0, 2)] = {
                                "aa": np.einsum('i,j,abij->abij', self.cosh_theta, self.cosh_theta, self.H[(0, 2)]),
                                "ab": np.einsum('i,j,abji->abij', self.cosh_theta, self.sinh_theta, self.H[(1, 1)]),
                                "ba": np.einsum('i,j,abij->abij', self.sinh_theta, self.cosh_theta, self.H[(1, 1)]),
                                "bb": np.einsum('i,j,abij->abij', self.sinh_theta, self.sinh_theta, self.H[(2, 0)])
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
        """ merge linear terms of the Bogliubov transformed tensor """
        A, N = self.A, self.N
        output_tensor = np.zeros([A, A, 2 * N])
        for x, y in it.product(range(A), repeat=2):
            output_tensor[x, y, :N] += input_tensor['a'][x, y, :]
            output_tensor[x, y, N:] += input_tensor['b'][x, y, :]

        return output_tensor

    def merge_quadratic(self, input_tensor):
        """ merge quadratic_terms of the Bogliubov transformed tensor """
        A, N = self.A, self.N
        output_tensor = np.zeros([A, A, 2 * N, 2 * N])
        for x, y in it.product(range(A), repeat=2):
            output_tensor[x, y, :N, :N] += input_tensor["aa"][x, y, :]
            output_tensor[x, y, :N, N:] += input_tensor["ab"][x, y, :]
            output_tensor[x, y, N:, :N] += input_tensor["ba"][x, y, :]
            output_tensor[x, y, N:, N:] += input_tensor["bb"][x, y, :]
        return output_tensor

    def reduce_H_tilde(self):
        """merge the a, b blocks of the Bogliubov transformed Hamiltonian into on tensor"""
        A, N = self.A, self.N
        # initialize
        self.H_tilde_reduce = {
            (0, 0): self.H_tilde[(0, 0)],
            }

        self.H_tilde_reduce[(1, 0)] = self.merge_linear(self.H_tilde[(1, 0)])
        self.H_tilde_reduce[(0, 1)] = self.merge_linear(self.H_tilde[(0, 1)])
        self.H_tilde_reduce[(1, 1)] = self.merge_quadratic(self.H_tilde[(1, 1)])
        self.H_tilde_reduce[(2, 0)] = self.merge_quadratic(self.H_tilde[(2, 0)])
        self.H_tilde_reduce[(0, 2)] = self.merge_quadratic(self.H_tilde[(0, 2)])

        print("##### Bogliubov transformed (fictitous) Hamiltonian after merge blocks ######")
        for rank in self.H_tilde_reduce.keys():
            print("Block {:}: \n {:}".format(rank, self.H_tilde_reduce[rank]))

        return


    def sum_over_states(self, output_path, basis_size=40, T_initial=10000, T_final=100, N_step=10000, compare_with_TNOE=False):
        """calculation thermal properties through sum over states"""
        A, N = self.A, self.N
        def _construct_vibrational_Hamitonian(h):
            """construct vibrational Hamiltonian in H.O. basis"""
            Hamiltonian = np.zeros((basis_size, basis_size, basis_size, basis_size))
            for a_1 in range(basis_size):
                for a_2 in range(basis_size):
                    for b_1 in range(basis_size):
                        for b_2 in range(basis_size):
                            if a_1 == b_1 and a_2 == b_2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = h[(0, 0)]
                                Hamiltonian[a_1, a_2, b_1, b_2] += h[(1, 1)][0, 0]*(b_1)+h[(1, 1)][1, 1]*(b_2)
                            if a_1 == b_1+1 and a_2 == b_2-1:
                                Hamiltonian[a_1, a_2, b_1, b_2] = h[(1, 1)][0, 1]*np.sqrt(b_1+1)*np.sqrt(b_2)
                            if a_1 == b_1-1 and a_2 == b_2+1:
                                Hamiltonian[a_1, a_2, b_1, b_2] = h[(1, 1)][1, 0]*np.sqrt(b_1)*np.sqrt(b_2+1)
                            if a_1 == b_1+1 and a_2 == b_2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = h[(1, 0)][0]*np.sqrt(b_1+1)
                            if a_1 == b_1 and a_2 == b_2+1:
                                Hamiltonian[a_1, a_2, b_1, b_2] = h[(1, 0)][1]*np.sqrt(b_2+1)
                            if a_1 == b_1-1 and a_2 == b_2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = h[(0, 1)][0]*np.sqrt(b_1)
                            if a_1 == b_1 and a_2 == b_2-1:
                                Hamiltonian[a_1, a_2, b_1, b_2] = h[(0, 1)][1]*np.sqrt(b_2)
                            if a_1 == b_1+2 and a_2 == b_2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = h[(2, 0)][0, 0]*np.sqrt(b_1+1)*np.sqrt(b_1+2)
                            if a_1 == b_1+1 and a_2 == b_2+1:
                                Hamiltonian[a_1, a_2, b_1, b_2] = (h[(2, 0)][0, 1] + h[(2, 0)][1, 0])*np.sqrt(b_1+1)*np.sqrt(b_2+1)
                            if a_1 == b_1 and a_2 == b_2+2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = h[(2, 0)][1, 1]*np.sqrt(b_2+1)*np.sqrt(b_2+2)
                            if a_1 == b_1-2 and a_2 == b_2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = h[(0, 2)][0, 0]*np.sqrt(b_1)*np.sqrt(b_1-1)
                            if a_1 == b_1-1 and a_2 == b_2-1:
                                Hamiltonian[a_1, a_2, b_1, b_2] = (h[(0, 2)][0, 1] + h[(0, 2)][1, 0])*np.sqrt(b_1)*np.sqrt(b_2)
                            if a_1 == b_1 and a_2 == b_2-2:
                                Hamiltonian[a_1, a_2, b_1, b_2] = h[(0, 2)][1, 1]*np.sqrt(b_2)*np.sqrt(b_2-1)

            Hamiltonian = Hamiltonian.reshape(basis_size**N, basis_size**N)

            return Hamiltonian

        def construct_full_Hamitonian():
            """contruction the full vibronic Hamiltonian in FCI H.O. basis"""
            # initialize the Hamiltonian
            H_FCI = np.zeros([A, basis_size**N, A, basis_size**N])
            # contruction the full Hamiltonian surface by surface
            for a, b in it.product(range(A), repeat=2):
                h = {}
                for block in self.H.keys():
                    if block != (0, 0):
                        h[block] = self.H[block][a, b, :]
                    else:
                        h[block] = self.H[block][a, b]

                H_FCI[a, :, b][:] += _construct_vibrational_Hamitonian(h)

            H_FCI = H_FCI.reshape(A*basis_size**N, A*basis_size**N)

            return H_FCI

        def Cal_partition_function(E, T):
            """ compute partition function """
            Z = sum(np.exp(-E / (self.Kb * T)))
            return Z

        def Cal_thermal_internal_energy(E, T, Z):
            """ compute thermal_internal_energy """
            energy = sum(E * np.exp(-E / (self.Kb * T))) / Z
            return energy

        print("### Start sum over state calculation! ###")

        if compare_with_TNOE:
            T_grid = self.temperature_grid
        else:
            beta_initial = 1. / (T_initial * self.Kb)
            beta_final = 1. / (T_final * self.Kb)
            T_grid = 1. / (self.Kb * np.linspace(beta_initial, beta_final, N_step))
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
        print("### Sum over state calculation terminated gracefully! ###")
        return

    def _map_initial_amplitude(self, T_initial=1000):
        """map initial T amplitude from Bose-Einstein statistics at high temperature"""
        def map_z0_amplitude(beta):
            """map t_0 amplitude from H.O. partition function"""
            z_0 = np.eye(A)
            for a in range(A):
                for i in range(N):
                    z_0[a, a] *= 1 / (1 - np.exp(-beta * self.Freq[i]))
                z_0[a, a] *= np.exp(-beta*(self.H[(0, 0)][a, a]-sum(self.H[(1, 0)][a, a, :]**2 / self.Freq)))

            return z_0

        def map_t1_amplitude():
            """map initial t_1 and t^1 amplitude from linear displacements"""
            # initialzie t_i and t_I tensors
            t_i = np.zeros([A, 2*N])
            for x in range(A):
                t_i[x, :N] -= self.H[(0, 1)][x, x, :] / self.Freq / self.cosh_theta
                t_i[x, N:] -= self.H[(0, 1)][x, x, :] / self.Freq / self.sinh_theta

            return t_i

        def map_t2_amplitude():
            """map t_2 amplitude from BE statistics and cumulant expression of 2-RDM"""
            # initialize t2 amplitude
            t_2 = np.zeros([A, 2 * N, 2 * N])

            # enter initial t_2 for ab block
            for x in range(A):
                t_2[x, N:, :N] += np.diag((BE_occ - self.sinh_theta**2 - np.ones(N)) / self.cosh_theta / self.sinh_theta)
                t_2[x, :N, N:] += np.diag((BE_occ - self.cosh_theta**2) / self.cosh_theta / self.sinh_theta)

            # symmetrize t_2 amplitude
            t_2_new = np.zeros_like(t_2)
            for x in range(N):
                for i, j in it.product(range(2 * N), repeat=2):
                    t_2_new[x, i, j] = 0.5 * (t_2[x, i, j] + t_2[x, j, i])

            return t_2_new

        N, A = self.N, self.A
        beta_initial = 1. / (self.Kb * T_initial)

        # calculation BE occupation number at initial beta
        BE_occ = np.ones(self.N) / (np.ones(self.N) - np.exp(-beta_initial * self.Freq))
        print("BE occupation number:{:}".format(BE_occ))

        # map initial T amplitudes
        initial_T_amplitude = {}
        initial_T_amplitude[1] = map_t1_amplitude()
        initial_T_amplitude[2] = map_t2_amplitude()


        # map initial Z amplitudes
        initial_Z_amplitude={}
        initial_Z_amplitude[0] = map_z0_amplitude(beta_initial)
        initial_Z_amplitude[1] = np.zeros([A, A, 2*N])
        initial_Z_amplitude[2] = np.zeros([A, A, 2*N, 2*N])

        print("### initialize T amplitude ###")
        for block in initial_T_amplitude.keys():
            print("Block:{:}\n{:}".format(block, initial_T_amplitude[block]))

        print("### initialize Z amplitude ###")
        for block in initial_Z_amplitude.keys():
            print("Block:{:}\n{:}".format(block, initial_Z_amplitude[block]))

        print("###  T and Z amplitude initialized successfully! ###")

        return initial_T_amplitude, initial_Z_amplitude

    def sim_trans_H(self, H_args, T_args):
        """calculate similarity transformed Hamiltonian (H{e^S})_connected"""
        A, N = self.A, self.N

        def f_t_0(H, T):
            """return residue R_0: (0, 0) block"""

            # initialize as zero
            R = np.zeros([A, A])

            # constant
            R += H[(0, 0)]

            # linear
            R += np.einsum('abk,k->ab', H[(0, 1)], T[1])

            # quadratic
            R += 0.5 * np.einsum('abkl,kl->ab', H[(0, 2)], T[2])
            R += 0.5 * np.einsum('abkl,k,l->ab', H[(0, 2)], T[1], T[1])

            return R

        def f_t_i(H, T):
            """return residue R_I: (0, 1) block"""

            # initialize as zero
            R = np.zeros([A, A, 2*N])

            # linear
            R += H[(0, 1)]

            # quadratic
            R += np.einsum('abik,k->abi', H[(0, 2)], T[1])

            return R

        def f_t_I(H, T):
            """return residue R_i: (1, 0) block"""

            # initialize
            R = np.zeros([A, A, 2*N])

            # non zero initial value of R
            R += H[(1, 0)]

            # linear
            R += np.einsum('abik,k->abi', H[(1, 1)], T[1])

            # quadratic
            R += np.einsum('abk,ki->abi', H[(0, 1)], T[2])
            R += np.einsum('abkl,k,li->abi', H[(0, 2)], T[1], T[2])

            return R

        def f_t_Ij(H, T):
            """return residue R_Ij: (1, 1) block"""

            # initialize
            R = np.zeros([A, A, 2*N, 2*N])

            # first term
            R += H[(1, 1)]

            # quadratic
            R += np.einsum('abjk,ki->abij', H[(0, 2)], T[2])

            return R

        def f_t_ij(H, T):
            """return residue R_IJ: (0, 2) block"""

            # initialize as zero
            R = np.zeros([A, A, 2*N, 2*N])

            # quadratic
            R += H[(0, 2)]

            return R

        def f_t_IJ(H, T):
            """return residue R_ij: (2, 0) block"""

            # initialize as zero
            R = np.zeros([A, A, 2*N, 2*N])

            # h term
            R += H[(2, 0)]

            # quadratic
            R += np.einsum('abkj,ki->abij', H[(1, 1)], T[2])
            R += np.einsum('abki,kj->abij', H[(1, 1)], T[2])
            R += np.einsum('abkl,ki,lj->abij', H[(0, 2)], T[2], T[2])

            return R

        # compute similarity transformed Hamiltonian over e^T
        sim_h = {}
        sim_h[(0, 0)] = f_t_0(H_args, T_args)
        sim_h[(0, 1)] = f_t_i(H_args, T_args)
        sim_h[(1, 0)] = f_t_I(H_args, T_args)
        sim_h[(1, 1)] = f_t_Ij(H_args, T_args)
        sim_h[(0, 2)] = f_t_ij(H_args, T_args)
        sim_h[(2, 0)] = f_t_IJ(H_args, T_args)

        return sim_h

    def _cal_G_matrix(self, H_bar_args, T_args, debug_flag=False):
        """calculate a second similarity tranformation (e^{T^dagger}H_bar)_f.c."""
        A, N = self.A, self.N
        def cal_G_0():
            """calculate (0, 0) block of G"""
            R = np.zeros([A, A])
            R += H_bar_args[(0, 0)]
            R += np.einsum('k,abk->ab', T_args[1], H_bar_args[(1, 0)])
            R += 0.5 * np.einsum('k,l,abkl->ab', T_args[1], T_args[1], H_bar_args[(2, 0)])
            R += 0.5 * np.einsum('kl,abkl->ab', T_args[2], H_bar_args[(2, 0)])
            return R

        def cal_G_I():
            """calculate (1, 0) block of G"""
            R = np.zeros([A, A, 2*N])
            R += H_bar_args[(1, 0)]
            R += np.einsum('k,abki->abi', T_args[1], H_bar_args[(2, 0)])
            return R

        def cal_G_i():
            """calculate (0, 1) block of G"""
            R = np.zeros([A, A, 2*N])
            R += H_bar_args[(0, 1)]
            R += np.einsum('k,abki->abi', T_args[1], H_bar_args[(1, 1)])
            R += np.einsum('ki,abk->abi', T_args[2], H_bar_args[(1, 0)])
            R += np.einsum('k,li,abkl->abi', T_args[1], T_args[2], H_bar_args[(2, 0)])
            return R

        def cal_G_Ij():
            """calculate (1, 1) block of G"""
            R = np.zeros([A, A, 2*N, 2*N])
            R += H_bar_args[(1, 1)]
            R += np.einsum('jk,abik->ij', T_args[2], H_bar_args[(2, 0)])
            return R

        def cal_G_IJ():
            """calculate (2, 0) block of G"""
            R = np.zeros([A, A, 2*N, 2*N])
            R += H_bar_args[(2, 0)]
            return R

        def cal_G_ij():
            """calculate (0, 2) block of G"""
            R = np.zeros([A, A, 2*N, 2*N])
            R += H_bar_args[(0, 2)]
            R += np.einsum('kj,abki->abij', T_args[2], H_bar_args[(1, 1)])
            R += np.einsum('ki,abkj->abij', T_args[2], H_bar_args[(1, 1)])
            R += np.einsum('ki,lj,abkl->abij', T_args[2], T_args[2], H_bar_args[(2, 0)])
            return R

        # initialize G as a python dictionary
        G_args = {}

        # calculation matrix element of G block by block
        G_args[(0, 0)] = cal_G_0()
        G_args[(1, 0)] = cal_G_I()
        G_args[(0, 1)] = cal_G_i()
        G_args[(1, 1)] = cal_G_Ij()
        G_args[(2, 0)] = cal_G_IJ()
        G_args[(0, 2)] = cal_G_ij()

        if debug_flag:
            print("### G matrix")
            for block in G_args.keys():
                print("Block {:}:\n {:}".format(block, G_args[block]))

        return G_args

    def _cal_C_matrix(self, Z_args, T_args, debug_flag=False):
        """calculate C matrix: (e^{T^dagger}Z)_f.c."""
        A, N = self.A, self.N
        def cal_C_0():
            """0 block of C matrix"""
            R = np.zeros(A)
            R += Z_args[0]
            R += np.einsum('k,ak->a', T_args[1], Z_args[1])
            R += 0.5 * np.einsum('k,l,akl->a', T_args[1], T_args[1], Z_args[2])
            R += 0.5 * np.einsum('kl,akl->a', T_args[2], Z_args[2])
            return R

        def cal_C_1():
            """1 block of C matrix"""
            R = np.zeros([A, 2*N])
            R += Z_args[1]
            R += np.einsum('k,aki->ai', T_args[1], Z_args[2])
            return R

        def cal_C_2():
            """2 block of C matrix"""
            R = np.zeros([A, 2*N, 2*N])
            R += Z_args[2]
            return R

        # intialize C amplitude a a python library
        C_args = {}

        C_args[0] = cal_C_0()
        C_args[1] = cal_C_1()
        C_args[2] = cal_C_2()

        if debug_flag:
            print("### rho matrix")
            for block in C_args.keys():
                print("Block {:}:\n{:}".format(block, C_args[block]))

        return C_args

    def _cal_rho_matrix(self, dT_args, T_args, debug_flag=False):
        """calculate rho matrix: (e^{T^dagger}dT)_f.c."""
        A, N = self.A, self.N
        def cal_rho_0():
            """0 block of rho matrix"""
            R = 0
            R += np.einsum('k,k->', T_args[1], dT_args[1])
            R += 0.5 * np.einsum('k,l,kl->', T_args[1], T_args[1], dT_args[2])
            R += 0.5 * np.einsum('kl,kl->', T_args[2], dT_args[2])
            return R

        def cal_rho_1():
            """1 block of rho matrix"""
            R = np.zeros([2*N])
            R += dT_args[1]
            R += np.einsum('k,ki->i', T_args[1], dT_args[2])
            return R

        def cal_rho_2():
            """2 block of rho matrix"""
            R = np.zeros([2*N, 2*N])
            R += dT_args[2]
            return R

        # intialize rho amplitude a a python library
        rho_args = {}

        rho_args[0] = cal_rho_0()
        rho_args[1] = cal_rho_1()
        rho_args[2] = cal_rho_2()

        if debug_flag:
            print("### rho matrix")
            for block in rho_args.keys():
                print("Block:{:}:\n{:}".format(block, rho_args[block]))

        return rho_args

    def cal_net_residual(self, H_args, Z_args, debug_flag=False):
        """calculation net residual <\omega H_bar Z>"""
        A, N = self.A, self.N

        def f_z_0(H, Z):
            """return residue R_0: 0 block"""

            # initialize as zero
            R = np.zeros(A)

            R += np.einsum('abk,bk->a', H[(0, 1)], Z[1])
            R += 0.5 * np.einsum('abkl,bkl->a', H[(0, 2)], Z[2])

            # disconneted CI term
            R += np.einsum('ab,b->a', H[(0, 0)], Z[0])

            return R

        def f_z_I(H, Z):
            """return residue R_i: 1 block"""

            # initialize
            R = np.zeros([A, 2*N])

            R += np.einsum('abik,bk->ai', H[(1, 1)], Z[1])
            R += np.einsum('abk,bki->ai', H[(0, 1)], Z[2])

            # disconnected CI terms
            R += np.einsum('ab,bi->ai', H[(0, 0)], Z[1])
            R += np.einsum('abi,b->ai', H[(1, 0)], Z[0])

            return R

        def f_z_IJ(H, Z):
            """return residue R_ij: (2, 0) block"""

            # # initialize as zero
            R = np.zeros([A, 2*N, 2*N])
            R += np.einsum('abjk,bki->aij', H[(1, 1)], Z[2])
            R += np.einsum('abik,bkj->aij', H[(1, 1)], Z[2])

            # disconnected CI terms
            R += np.einsum('abij,b->aij', H[(2, 0)], Z[0])
            R += np.einsum('abi,bj->aij', H[(1, 0)], Z[1])
            R += np.einsum('abj,bi->aij', H[(1, 0)], Z[1])
            R += np.einsum('ab,bij->aij', H[(0, 0)], Z[2])

            return R

        residual = {}
        # calculate net residual block by block
        residual[0] = f_z_0(H_args, Z_args)
        residual[1] = f_z_I(H_args, Z_args)
        residual[2] = f_z_IJ(H_args, Z_args)

        if debug_flag:
            print("### Net residual:")
            for block in residual.keys():
                print("Block :{:}\n{:}".format(block, residual[block]))

        return residual

    def cal_T_Z_residual(self, T_args, Z_args):
        """calculation T and Z residual"""
        N = self.N
        def cal_T_residual(H_args, Z_args, debug_flag=False):
            """calculation T residual from Ehrenfest parameterization"""

            def cal_dT_1():
                """1 block of T residual"""
                R = np.einsum('a,abi,b->i', Z_args[0], H_args[(1, 0)], Z_args[0]) / np.einsum('a,a->', Z_args[0], Z_args[0])
                return R

            def cal_dT_2():
                """2 block of T residual"""
                R = np.einsum('a,abij,b->ij', Z_args[0], H_args[(2, 0)], Z_args[0]) / np.einsum('a,a->', Z_args[0], Z_args[0])
                return R

            residual = {}
            residual[1] = cal_dT_1()
            residual[2] = cal_dT_2()

            if debug_flag:
                print("### Ehrenfest parameteried T residual:")
                for block in residual.keys():
                    print("Block{:}:\n{:}".format(block, residual[block]))

            return residual

        def cal_Z_residual(R_args, rho_args, C_args, debug_flag=False):
            """calculation Z residual by strustracting T residual from the net residual"""
            def cal_rho_C():
                """calculate (<y,theta|Omega^dagger_v rho C|x,theta> term in the Z constribution)"""

                # initialize as an python dictionary
                R ={}

                # 0 block contribution
                R[0] = rho_args[0] * C_args[0]

                # 1 block contribution
                R[1] = np.einsum('i,y->yi', rho_args[1], C_args[0])
                R[1] += rho_args[0] * C_args[1]

                # 2 blolck contribution
                R[2] = np.einsum('i,yj->yij', rho_args[1], C_args[1])
                R[2] += 0.5 * rho_args[0] * C_args[2]

                return R

            def cal_D_0():
                """0 block of D matrix"""
                R = np.zeros(A)
                R += np.einsum('k,ak->a', T_args[1], dZ_args[1])
                R += 0.5 * np.einsum('k,l,akl->a', T_args[1], T_args[1], dZ_args[2])
                R += 0.5 * np.einsum('kl,akl->a', T_args[2], dZ_args[2])
                return R

            def cal_D_1():
                """1 block of D matrix"""
                R = np.zeros([A, 2*N])
                R += np.einsum('k,aki->ai', T_args[1], dZ_args[2])
                return R


            def cal_dZ_0():
                """(0, 0) block of Z residual"""
                R = R_args[0]
                R -= rho_C[0]
                R -= D_args[0]
                return R

            def cal_dZ_I():
                """(1, 0) block of Z residual"""
                R = R_args[1]
                R -= rho_C[1]
                R -= D_args[1]

                return R

            def cal_dZ_IJ():
                """(2, 0) block of Z residual"""
                R = R_args[2]
                R -= rho_C[2]

                return R

            A, N = self.A, self.N
            # calculation rho * C
            rho_C = cal_rho_C()

            # recursived solve equation for dZ / dbeta

            ## initialize D_args and Z_args as python dictionnary
            D_args = {}
            dZ_args = {}

            # solve equation for "2" block
            dZ_args[2] = cal_dZ_IJ()
            D_args[1] = cal_D_1()

            # solve equation for "1" block
            dZ_args[1] = cal_dZ_I()
            D_args[0] = cal_D_0()

            # solve equation for "0" block
            dZ_args[0] = cal_dZ_0()

            if debug_flag:
                print("### Z residual:")
                for block in dZ_args.keys():
                    print("Block{:}:\n{:}".format(block, dZ_args[block]))

            return dZ_args

        # calculate similarity transfromed Hamiltonian
        H_bar = self.sim_trans_H(self.H_tilde_reduce, T_args)

        # perform a second similarity transformation of the Hamiltonian
        G_args = self._cal_G_matrix(H_bar, T_args, debug_flag=True)

        # calculate C matrix
        C_args = self._cal_C_matrix(Z_args, T_args, debug_flag=True)

        # calculation net residual
        net_residual = self.cal_net_residual(G_args, C_args, debug_flag=True)

        # calculate T residual
        t_residual = cal_T_residual(G_args, C_args, debug_flag=True)

        # calculation rho matrix
        rho_args = self._cal_rho_matrix(t_residual, T_args, debug_flag=True)

        # calculate Z residual
        z_residual = cal_Z_residual(net_residual, rho_args, C_args, debug_flag=True)

        return t_residual, z_residual


    def TFCC_integration(self, output_path, T_initial, T_final, N_step, debug_flag=False):
        """
        conduct TFCC imaginary time integration to calculation thermal
        properties
        T_initial: initial temperature of the integration
        T_final: final temperature of the integration
        N_step: number of the numerical steps
        """
        A, N = self.A, self.N
        # map initial T amplitude
        T_amplitude, Z_amplitude = self._map_initial_amplitude(T_initial=T_initial)

        # define the temperature grid for the integration
        beta_initial = 1. / (self.Kb * T_initial)
        beta_final = 1. / (self.Kb * T_final)
        step = (beta_final - beta_initial) / N_step
        self.temperature_grid = 1. / (self.Kb * np.linspace(beta_initial, beta_final, N_step))

        # define empty list to thermal properties
        self.partition_function = []
        self.internal_energy = []

        # thermal field imaginary time propagation
        for i in range(N_step):
            # initialize each block of T / Z residual as zeros
            T_residual, Z_residual = {}, {}
            for block in T_amplitude.keys():
                T_residual[block] = np.zeros_like(T_amplitude[block])
            for block in Z_amplitude.keys():
                Z_residual[block] = np.zeros_like(Z_amplitude[block])
            for x in range(A):
                # calculate residual one surface at a time
                t_amplitude, z_amplitude = {}, {}
                for block in T_amplitude.keys():
                    t_amplitude[block] = T_amplitude[block][x, :]
                for block in Z_amplitude.keys():
                    z_amplitude[block] = Z_amplitude[block][x, :]
                t_residual, z_residual = self.cal_T_Z_residual(t_amplitude, z_amplitude)
                for block in t_residual.keys():
                    T_residual[block][x, :] += t_residual[block]
                for block in z_residual.keys():
                    Z_residual[block][x, :] += z_residual[block]

            # update amplitudes
            for block in T_amplitude.keys():
                if debug_flag:
                    print("T{:}:{:}".format(block, T_residual[block]))
                T_amplitude[block] -= T_residual[block] * step
            for block in Z_amplitude.keys():
                if debug_flag:
                    print("Z{:}:{:}".format(block, Z_residual[block]))
                Z_amplitude[block] -= Z_residual[block] * step

            # calculate partition function and store data
            Z = np.trace(Z_amplitude[0])
            self.partition_function.append(Z)
            # calculate thermal internal energy and store data
            E = np.trace(Z_residual[0]) / Z
            self.internal_energy.append(E)

            # print out CC amplitude along the integration
            if True:
                print("step {:}:".format(i))
                print("max z_0 amplitude:{:}".format(abs(Z_amplitude[0]).max()))
                print("max z_1 amplitude:{:}".format(abs(Z_amplitude[1]).max()))
                print("max z_2 amplitude:{:}".format(abs(Z_amplitude[2]).max()))
                print("max t_1 ampltiude:{:}".format(abs(T_amplitude[1]).max()))
                print("max t_2 amplitude:{:}".format(abs(T_amplitude[2]).max()))
                print("Temperature: {:} K".format(self.temperature_grid[i]))
                print("thermal internal energy: {:} cm-1".format(E))
                print("partition function: {:}".format(Z))

        # store data
        thermal_data = {"temperature": self.temperature_grid, "internal energy": self.internal_energy, "partition function": self.partition_function}
        df = pd.DataFrame(thermal_data)
        df.to_csv(output_path+"thermal_data_TFCC.csv", index=False)

        return

    def plot_thermal(self):
        """plot thermal properties"""
        print(len(self.temperature_grid))
        print(len(self.internal_energy))
        plt.figure(figsize=(10, 10))
        plt.title("Plot of thermal internal energy", fontsize=40)
        plt.plot(self.temperature_grid, self.internal_energy)
        plt.xlabel("T(K)", fontsize=40)
        plt.ylabel("energy(cm-1)", fontsize=40)
        plt.show()
        # plt.savefig("energy.png")

        plt.figure(figsize=(10, 10))
        plt.title("Plot of partition function", fontsize=40)
        plt.plot(self.temperature_grid, self.partition_function)
        plt.xlabel("T(K)", fontsize=40)
        plt.ylabel("partition_function", fontsize=40)
        plt.show()
        # plt.savefig("partition_function.png")
