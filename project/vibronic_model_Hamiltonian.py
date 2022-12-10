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

    def __init__(self, freq, LCP, VE, num_mode, num_surf):
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

        return

    def _map_initial_amplitude(self, T_initial=1000):
        """map initial T amplitude from Bose-Einstein statistics at high temperature"""
        def map_z_0_amplitude(beta):
            """map t_0 amplitude from H.O. partition function"""
            z_0 = np.eye(A)
            for a in range(A):
                for i in range(N):
                    z_0[a, a] *= 1 / (1 - np.exp(-beta * self.Freq[i]))
                z_0[a, a] *= np.exp(-beta*(self.H[(0, 0)][a, a]-sum(self.H[(1, 0)][a, a, :]**2 / self.Freq)))

            return z_0

        def map_t_1_amplitude():
            """map initial t_1 and t^1 amplitude from linear displacements"""
            # initialzie t_i and t_I tensors
            t_i = np.zeros([A, N])
            t_I = np.zeros([A, N])
            for a in range(A):
                t_i[a, :] = -self.H[(0, 1)][a, a, :] / self.Freq
                t_I[a, :] = -self.H[(1, 0)][a, a, :] / self.Freq

            return t_i, t_I
        def map_t11_amplitude(beta):
            """map t_11 amplitude from Bose-Einstein occupation number"""
            # initialize t11 amplitude
            t_11 = np.zeros([A, N, N])
            for a, i in it.product(range(A), range(N)):
                t_11[a, i, i] = 1 / (np.exp(beta * self.Freq[i]) - 1)
            return t_11

        N, A = self.N, self.A
        beta_initial = 1. / (self.Kb * T_initial)

        # map linear amplitude
        t_i, t_I = map_t_1_amplitude()

        initial_T_amplitude = {}

        # initialize (1, 0) and (0, 1) T amplitude from linear displacements
        initial_T_amplitude[(1, 0)] = t_I
        initial_T_amplitude[(0, 1)] = t_i
        # initialize (0 ,0) and (1, 1) T amplitude from high T limit of BE statistics
        initial_T_amplitude[(1, 1)] = map_t11_amplitude(beta_initial)

        # initialize the rest of T amplitudes to be zeros
        initial_T_amplitude[(2, 0)] = np.zeros([A, N, N])
        initial_T_amplitude[(0, 2)] = np.zeros([A, N, N])

        initial_Z_amplitude = {}

        # initialize (0, 0) block of the Z amplitude from D.H.O
        initial_Z_amplitude[(0, 0)] = map_z_0_amplitude(beta_initial)

        # initializae rest of the Z amplitude to be zeros
        initial_Z_amplitude[(1, 0)] = np.zeros([A, A, N])
        initial_Z_amplitude[(0, 1)] = np.zeros([A, A, N])

        initial_Z_amplitude[(1, 1)] = np.zeros([A, A, N, N])
        initial_Z_amplitude[(2, 0)] = np.zeros([A, A, N, N])
        initial_Z_amplitude[(0, 2)] = np.zeros([A, A, N, N])

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
        N = self.N

        def f_t_0(H, T):
            """return residue R_0: (0, 0) block"""

            # initialize as zero
            R = 0.

            # constant
            R += H[(0, 0)]

            # linear
            R += np.einsum('abk,k->ab', H[(0, 1)], T[(1, 0)])

            # quadratic
            R += 0.5 * np.einsum('abkl,kl->ab', H[(0, 2)], T[(2, 0)])
            R += 0.5 * np.einsum('abkl,k,l->ab', H[(0, 2)], T[(1, 0)], T[(1, 0)])

            # terms associated with thermal

            # Linear
            R += np.einsum('abkl,lk->ab', H[(1, 1)], T[(1, 1)])
            R += np.einsum('abk,k->ab', H[(1, 0)], T[(0, 1)])
            R += 0.5 * np.einsum('abkl,kl->ab', H[(2, 0)], T[(0, 2)])
            # quadratic
            R += np.einsum('abkl,k,l->ab', H[(1, 1)], T[(0, 1)], T[(1, 0)])
            R += 0.5 * np.einsum('abkl,k,l->ab', H[(2, 0)], T[(0, 1)], T[(0, 1)])

            return R

        def f_t_i(H, T):
            """return residue R_I: (0, 1) block"""

            # initialize as zero
            R = np.zeros(N)

            # linear
            R += H[(0, 1)]

            # quadratic
            R += np.einsum('abik,k->abi', H[(0, 2)], T[(1, 0)])

            # terms associated with thermal

            # Linear
            R += np.einsum('abki,k->abi', H[(1, 1)], T[(0, 1)])
            R += np.einsum('abk,ki->abi', H[(1, 0)], T[(0, 2)])
            R += np.einsum('abk,ki->abi', H[(0, 1)], T[(1, 1)])
            # quadratic
            R += np.einsum('ablk,ki,l->abi', H[(1, 1)], T[(1, 1)], T[0, 1])
            R += np.einsum('ablk,k,li->abi', H[(1, 1)], T[(1, 0)], T[(0, 2)])
            R += np.einsum('abkl,k,li->abi', H[(0, 2)], T[(1, 0)], T[(1, 1)])
            R += np.einsum('abkl,k,li->abi', H[(2, 0)], T[(0, 1)], T[(0, 2)])

            return R

        def f_t_I(H, T):
            """return residue R_i: (1, 0) block"""

            # initialize
            R = np.zeros(N)

            # non zero initial value of R
            R += H[(1, 0)]

            # linear
            R += np.einsum('abik,k->abi', H[(1, 1)], T[(1, 0)])

            # quadratic
            R += np.einsum('abk,ki->abi', H[(0, 1)], T[(2, 0)])
            R += np.einsum('abkl,k,li->abi', H[(0, 2)], T[(1, 0)], T[(2, 0)])

            # terms associated with thermal

            # linear
            R += np.einsum('abk,ik->abi', H[(1, 0)], T[(1, 1)])
            R += np.einsum('abki,k->abi', H[(2, 0)], T[(0, 1)])

            # quadratic
            R += np.einsum('ablk,il,k->abi', H[(1, 1)], T[(1, 1)], T[(1, 0)])
            R += np.einsum('ablk,l,ki->abi', H[(1, 1)], T[(0, 1)], T[(2, 0)])
            R += np.einsum('abkl,k,il->abi', H[(2, 0)], T[(0, 1)], T[(1, 1)])

            return R

        def f_t_Ij(H, T):
            """return residue R_Ij: (1, 1) block"""

            # initialize
            R = np.zeros([N, N])

            # first term
            R += H[(1, 1)]

            # quadratic
            R += np.einsum('abik,kj->abij', H[(0, 2)], T[2, 0])

            # terms associated with thermal

            # linear
            R += np.einsum('abkj,ik->abij', H[(1, 1)], T[(1, 1)])
            R += np.einsum('abik,kj->abij', H[(1, 1)], T[(1, 1)])
            R += np.einsum('abjk,ik->abij', H[(2, 0)], T[(0, 2)])

            # quadratic
            R += np.einsum('ablk,kj,il->abij', H[(1, 1)], T[(1, 1)], T[(1, 1)])
            R += np.einsum('abkl,kj,li->abij', H[(1, 1)], T[(0, 2)], T[(2, 0)])
            R += np.einsum('abkl,lj,ik->abij', H[(2, 0)], T[(0, 2)], T[(1, 1)])
            R += np.einsum('abkl,li,kj->abij', H[(0, 2)], T[(2, 0)], T[(1, 1)])



            return R

        def f_t_ij(H, T):
            """return residue R_IJ: (0, 2) block"""

            # initialize as zero
            R = np.zeros([N, N])

            # quadratic
            R += H[(0, 2)]

            # terms associated with thermal

            R += np.einsum('abki,kj->abij', H[(1, 1)], T[(0, 2)])
            R += np.einsum('abkj,ki->abij', H[(1, 1)], T[(0, 2)])
            R += np.einsum('abki,kj->abij', H[(0, 2)], T[(1, 1)])
            R += np.einsum('abkj,ki->abij', H[(0, 2)], T[(1, 1)])

            R += np.einsum('ablk,kj,li->abij', H[(1, 1)], T[(1, 1)], T[(0, 2)])
            R += np.einsum('ablk,ki,lj->abij', H[(1, 1)], T[(1, 1)], T[(0, 2)])
            R += np.einsum('abkl,lj,ki->abij', H[(2, 0)], T[(0, 2)], T[(0, 2)])
            R += np.einsum('abkl,ki,lj->abij', H[(0, 2)], T[(1, 1)], T[(1, 1)])


            return R

        def f_t_IJ(H, T):
            """return residue R_ij: (2, 0) block"""

            # # initialize as zero
            R = np.zeros([N, N])

            # if self.hamiltonian_truncation_order >= 2:

            # quadratic
            R += H[(2, 0)]  # h term
            R += np.einsum('abkj,ki->abij', H[(1, 1)], T[2, 0])
            R += np.einsum('abki,kj->abij', H[(1, 1)], T[2, 0])
            R += np.einsum('abkl,ki,lj->abij', H[(0, 2)], T[2, 0], T[2, 0])

            # terms associated with thermal

            # linear
            R += np.einsum('abjk,ik->abij', H[(2, 0)], T[(1, 1)])
            R += np.einsum('abik,jk->abij', H[(2, 0)], T[(1, 1)])

            # quadratic
            R += np.einsum('ablk,ki,jl->abij', H[(1, 1)], T[(2, 0)], T[(1, 1)])
            R += np.einsum('ablk,kj,il->abij', H[(1, 1)], T[(2, 0)], T[(1, 1)])
            R += np.einsum('abkl,ik,jl->abij', H[(2, 0)], T[(1, 1)], T[(1, 1)])

            return R

        # compute similarity transformed Hamiltonian over e^T
        sim_h = {}
        sim_h[(0, 0)] = f_t_0(H_args, t_args)
        sim_h[(0, 1)] = f_t_i(H_args, t_args)
        sim_h[(1, 0)] = f_t_I(H_args, t_args)
        sim_h[(1, 1)] = f_t_Ij(H_args, t_args)
        sim_h[(0, 2)] = f_t_ij(H_args, t_args)
        sim_h[(2, 0)] = f_t_IJ(H_args, t_args)

        return sim_h

    def cal_net_residual(self, H_args, Z_args):
        """calculation net residual <\omega H_bar Z>"""
        A, N = self.A, self.N

        def f_z_0(H, T):
            """return residue R_0: (0, 0) block"""

            # initialize as zero
            R = np.zeros(A)

            R += np.einsum('ab,b->a', H[(0, 0)], T[(0, 0)])
            R += np.einsum('abk,bk->a', H[(0, 1)], T[(1, 0)])
            R += 0.5 * np.einsum('abkl,bkl->a', H[(0, 2)], T[(2, 0)])

            # terms associated with thermal
            R += np.einsum('abkl,blk->a', H[(1, 1)], T[(1, 1)])
            R += np.einsum('abk,bk->a', H[(1, 0)], T[(0, 1)])
            R += 0.5 * np.einsum('abkl,bkl->a', H[(2, 0)], T[(0, 2)])

            # disconneted CI term
            R += np.einsum('ab,b->a', H[(0, 0)], T[(0, 0)])

            return R

        def f_z_i(H, T):
            """return residue R_I: (0, 1) block"""

            # initialize as zero
            R = np.zeros([A, N])
            # R += H[(0, 1)]
            # R += np.einsum('abik,bk->ai', H[(0, 2)], T[(1, 0)])

            # terms associated with thermal
            # R += np.einsum('abki,bk->ai', H[(1, 1)], T[(0, 1)])
            R += np.einsum('abk,bki->ai', H[(1, 0)], T[(0, 2)])
            R += np.einsum('abk,bki->ai', H[(0, 1)], T[(1, 1)])
            # disconnect CI terms
            R += np.einsum('ab,bi->ai', H[(0, 0)], T[(0, 1)])
            # R += np.einsum('abi,b->ai', H[(0, 1)], T[(0, 0)])
            return R

        def f_z_I(H, T):
            """return residue R_i: (1, 0) block"""

            # initialize
            R = np.zeros([A, N])

            R += H[(1, 0)]
            R += np.einsum('abik,bk->ai', H[(1, 1)], T[(1, 0)])
            R += np.einsum('abk,bki->ai', H[(0, 1)], T[(2, 0)])

            # terms associated with thermal
            R += np.einsum('abk,bik->ai', H[(1, 0)], T[(1, 1)])
            R += np.einsum('abki,bk->ai', H[(2, 0)], T[(0, 1)])
            # disconnected CI terms
            R += np.einsum('ab,bi->ai', H[(0, 0)], T[(1, 0)])
            R += np.einsum('abi,b->ai', H[(1, 0)], T[(0, 0)])

            return R

        def f_z_Ij(H, T):
            """return residue R_Ij: (1, 1) block"""

            # initialize
            R = np.zeros([A, N, N])
            # R += H[(1, 1)]
            # R += np.einsum('abik,bkj->aij', H[(0, 2)], T[2, 0])

            # terms associated with thermal
            # R += np.einsum('abkj,ik->abij', H[(1, 1)], T[(1, 1)])
            R += np.einsum('abik,bkj->aij', H[(1, 1)], T[(1, 1)])
            R += np.einsum('abik,bjk->aij', H[(2, 0)], T[(0, 2)])
            # disconneted CI terms
            R += np.einsum('ab,bij->aij', H[(0, 0)], T[(1, 1)])
            # R += np.einsum('abi,bj->aij', H[(1, 0)], T[(0, 1)])
            R += np.einsum('abj,bi->aij', H[(0, 1)], T[(1, 0)])
            # R += np.einsum('abij,b->aij', H[(1, 1)], T[(0, 0)])

            return R

        def f_z_ij(H, T):
            """return residue R_IJ: (0, 2) block"""

            # initialize as zero
            R = np.zeros([A, N, N])

            # quadratic
            # R += H[(0, 2)]

            # terms associated with thermal
            # R += np.einsum('abki,kj->abij', H[(1, 1)], T[(0, 2)])
            # R += np.einsum('abkj,ki->abij', H[(1, 1)], T[(0, 2)])
            # R += np.einsum('abki,kj->abij', H[(0, 2)], T[(1, 1)])
            # R += np.einsum('abkj,ki->abij', H[(0, 2)], T[(1, 1)])

            # discnneted CI terms
            # R += np.einsum('abij,b->aij', H[(0, 2)], T[(0, 0)])
            # R += np.einsum('abi,bj->aij', H[(0, 1)], T[(0, 1)])
            # R += np.einsum('abj,bi->aij', H[(0, 1)], T[(0, 1)])
            R += np.einsum('ab,bij->aij', H[(0, 0)], T[(0, 2)])

            return R

        def f_z_IJ(H, T):
            """return residue R_ij: (2, 0) block"""

            # # initialize as zero
            R = np.zeros([A, N, N])
            R += H[(2, 0)]  # h term
            R += np.einsum('abjk,bki->aij', H[(1, 1)], T[(2, 0)])
            R += np.einsum('abik,bkj->aij', H[(1, 1)], T[(2, 0)])
            # terms associated with thermal
            R += np.einsum('abjk,bik->aij', H[(2, 0)], T[(1, 1)])
            R += np.einsum('abik,bjk->aij', H[(2, 0)], T[(1, 1)])

            # disconnected CI terms
            R += np.einsum('abij,b->aij', H[(2, 0)], T[(0, 0)])
            R += np.einsum('abi,bj->aij', H[(1, 0)], T[(1, 0)])
            R += np.einsum('abj,bi->aij', H[(1, 0)], T[(1, 0)])
            R += np.einsum('ab,bij->aij', H[(0, 0)], T[(2, 0)])

            return R

        residual = {}
        # calculate net residual block by block
        residual[(0, 0)] = f_z_0(H_args, Z_args)
        residual[(0, 1)] = f_z_i(H_args, Z_args)
        residual[(1, 0)] = f_z_I(H_args, Z_args)
        residual[(1, 1)] = f_z_Ij(H_args, Z_args)
        residual[(0, 2)] = f_z_ij(H_args, Z_args)
        residual[(2, 0)] = f_z_IJ(H_args, Z_args)

        return residual

    def cal_T_Z_residual(self, T_args, Z_args):
        """calculation T and Z residual"""

        def cal_T_residual(H_args, Z_args):
            """calculation T residual from Ehrenfest parameterization"""
            return residual

        def cal_Z_residual(R_args, dT_args):
            """calculation Z residual by strustracting T residual from the net residual"""
            return residual
        # calculate similarity transfromed Hamiltonian
        H_bar = self.sim_trans_H(self.H, T_args)
        # calculation net residual
        net_residual = cal_net_residual(H_bar, Z_args)
        # calculate T residual
        t_residual = cal_T_residual(H_bar, Z_args)
        # calculate Z residual
        z_residual = cal_Z_residual(net_residual, t_residual)

        return t_residual, z_residual


    def TFCC_integration(self, output_path, T_initial, T_final, N):
        """conduct TFCC imaginary time integration to calculation thermal properties"""

        # map initial T amplitude
        T_amplitude, Z_amplitude = self._map_initial_T_amplitude(T_initial=T_initial)

        beta_initial = 1. / (self.Kb * T_initial)
        beta_final = 1. / (self.Kb * T_final)
        step = (beta_final - beta_initial) / N
        self.temperature_grid = 1. / (self.Kb * np.linspace(beta_initial, beta_final, N))
        self.partition_function = []
        self.internal_energy = []
        # thermal field imaginary time propagation
        for i in range(N):
            # initialize each block of T / Z residual as zeros
            T_residual, Z_residual = {}, {}
            for block in T_amplitude.keys():
                T_residual[block] = np.zeros_like(T_amplitude[block])
            for block in Z_amplitude.keys():
                Z_residual[blck] = np.zeros_like(Z_amplitude[block])
            for x in range(A):
                # calculate residual one surface at a time
                t_amplitude, z_amplitude = {}, {}
                for block in T_amplitude.keys():
                    t_amplitude[block] = T_amplitude[block][x, :]
                for block in Z_amplitude.keys():
                    z_amplitude[block] = Z_amplitude[block][x, :]
                t_residual, z_residual = self.cal_T_Z_residual(self.H, t_amplitude, z_amplitude)
                for block in t_residual.keys():
                    T_residual[block][x, :] += t_residual[block]
                for block in z_residual.keys():
                    Z_residual[block][x, :] += z_residual[block]

            # update amplitudes
            for block in T_amplitude.keys():
                T_amplitude[block] -= T_residual[block] * step
            for block in Z_amplitude.keys():
                Z_amplitude[block] -= Z_residual[block] * step

            # calculate partition function
            Z = np.trace(Z_amplitude[(0, 0)])
            self.partition_function.append(Z)
            # calculate thermal internal energy
            E = np.trace(Z_residual[(0, 0)]) / Z
            self.thermal_internal_energy.append(E)

            print("step {:}:".format(i))
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
