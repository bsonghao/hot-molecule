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
import matplotlib as mpl; mpl.use('pdf')
import matplotlib.pyplot as plt
import parse  # used for loading data files
import pandas as pd
#
import opt_einsum as oe


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

        print("### End of Hamiltonian parameters ####")

    def sum_over_states(self, output_path, basis_size=40, T_grid=np.linspace(100, 1000, 10000)):
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



        compare_with_TFCC = False
        if compare_with_TFCC:
            T_grid = self.temperature_grid
        # contruct Hamiltonian in H.O. basis
        H = construct_full_Hamitonian()
        # check Hemicity of the Hamitonian in H. O. basis
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

    def thermal_field_transformation(self, Temp):
        """conduct Bogoliubov transformation of input Hamiltonian and determine thermal field reference state"""
        # calculate inverse temperature
        self.T_ref = Temp
        beta = 1. / (self.Kb * Temp)
        # define Bogliubov transformation based on Bose-Einstein statistics
        self.cosh_theta = 1. / np.sqrt((np.ones(self.N) - np.exp(-beta * self.Freq)))
        self.sinh_theta = np.exp(-beta * self.Freq / 2.) / np.sqrt(np.ones(self.N) - np.exp(-beta * self.Freq))

        # Bogliubov tranform that Hamiltonian
        self.H_tilde = dict()

        # constant term???
        self.H_tilde[(0, 0)] = self.H[(0, 0)] + np.trace(np.einsum('ij,i,j->ij', self.H[(1, 1)], self.sinh_theta, self.sinh_theta))

        # linear terms
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
                                "bb": np.einsum('j,i,ij->ij', self.sinh_theta, self.sinh_theta, self.H[(1, 1)])
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

    def _map_initial_T_amplitude(self, T_initial=1000):
        """map initial T amplitude from Bose-Einstein statistics at high temperature"""
        def map_t_0_amplitude(beta):
            """map t_0 amplitude from partition function"""
            z = 1
            for i,w in enumerate(self.Freq):
                z *= np.exp(-beta * w / 2) / (1 - np.exp(-beta * w))
            t_0 = np.log(z)
            return t_0
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
        initial_T_amplitude[0] = map_t_0_amplitude(beta_initial)
        initial_T_amplitude[1] = map_t1_amplitude()
        initial_T_amplitude[2] = map_t2_amplitude(two_RDM, initial_T_amplitude[1])

        print("initial single T amplitude:\n{:}".format(initial_T_amplitude[1]))
        print("initial double T amplitude:\n{:}".format(initial_T_amplitude[2]))

        return initial_T_amplitude

    def merge_linear(self, input_tensor):
        """ merge linear terms of the Bogliubov transformed tensor """
        N = self.N
        output_tensor = np.zeros(2 * N)
        output_tensor[:N] = input_tensor['a'].copy()
        output_tensor[N:] = input_tensor['b'].copy()

        return output_tensor

    def merge_quadratic(self, input_tensor):
        """ merge quadratic_terms of the Bogliubov transformed tensor """
        N =self.N
        output_tensor = np.zeros([2 * N, 2 * N])
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


    def _map_initial_T_amplitude_from_FCI(self, T_initial=1000, basis_size=40):
        """map initial T amplitude from FCI"""
        def cal_rdm_i(E, V, Z, basis_size, beta):
            """calculate one body density matrix from FCI"""
            rdm_i = np.zeros(self.N)
            for l, e in enumerate(E):
                for i in range(self.N):
                    for n_i in range(basis_size):
                        for n_j in range(1, basis_size):
                            if i == 0:
                                rdm_i[i] += np.sqrt(n_j) * V[n_j, n_i, l] * V[n_j-1, n_i, l] * np.exp(-beta * e)
                            elif i == 1:
                                rdm_i[i] += np.sqrt(n_j) * V[n_i, n_j, l] * V[n_i, n_j-1, l] * np.exp(-beta * e)
            rdm_i /= Z
            return rdm_i

        def cal_rdm_I(E, V, Z, basis_size, beta):
            """calculate one body density matrix from FCI"""
            rdm_I = np.zeros(self.N)
            for l, e in enumerate(E):
                for i in range(self.N):
                    for n_i in range(basis_size):
                        for n_j in range(0, basis_size-1):
                            if i == 0:
                                rdm_I[i] += np.sqrt(n_j+1) * V[n_j+1, n_i, l] * V[n_j, n_i, l] * np.exp(-beta * e)
                            elif i == 1:
                                rdm_I[i] += np.sqrt(n_j+1) * V[n_i, n_j+1, l] * V[n_i, n_j, l] * np.exp(-beta * e)
            rdm_I /= Z
            return rdm_I


        def cal_rdm_ij(E, V, Z, basis_size, beta):
            """calculate two body density matrix from FCI"""
            rdm_ij = np.zeros([self.N, self.N])
            for i in range(self.N):
                for j in range(self.N):
                    #i=j
                    if i == j:
                        for n_i in range(2, basis_size):
                            for n_j in range(basis_size):
                                for l, e in enumerate(E):
                                    if i == 0:
                                        rdm_ij[i, i] += np.sqrt(n_i) * np.sqrt(n_i-1) *\
                                                   V[n_i-2, n_j, l] * V[n_i, n_j, l] *\
                                                   np.exp(-e * beta)
                                    else:
                                        rdm_ij[i, i] += np.sqrt(n_i) * np.sqrt(n_i-1) *\
                                                    V[n_j, n_i-2, l]* V[n_j, n_i, l] *\
                                                    np.exp(-e * beta)

                    #i!=j
                    else:
                        for n_i in range(1, basis_size):
                            for n_j in range(1, basis_size):
                                for l, e in enumerate(E):
                                    if i == 0 and j == 1:
                                        rdm_ij[i, j] += np.sqrt(n_i) * np.sqrt(n_j) *\
                                                    V[n_i-1, n_j-1, l]* V[n_i, n_j, l] *\
                                                    np.exp(-e * beta)
                                    elif i == 1 and j == 0:
                                        rdm_ij[i, j] += np.sqrt(n_i) * np.sqrt(n_j) *\
                                                    V[n_j-1, n_i-1, l] * V[n_j, n_i, l] *\
                                                    np.exp(-e * beta)
            rdm_ij /= Z
            return rdm_ij

        def cal_rdm_IJ(E, V, Z, basis_size, beta):
            """calculate two body density matrix from FCI"""
            rdm_IJ = np.zeros([self.N, self.N])
            for i in range(self.N):
                for j in range(self.N):
                    #i=j
                    if i == j:
                        for n_i in range(basis_size-2):
                            for n_j in range(basis_size):
                                for l, e in enumerate(E):
                                    if i == 0:
                                        rdm_IJ[i, i] += np.sqrt(n_i+1) * np.sqrt(n_i+2) *\
                                                   V[n_i+2, n_j, l] * V[n_i, n_j, l] *\
                                                   np.exp(-e * beta)
                                    else:
                                        rdm_IJ[i, i] += np.sqrt(n_i+1) * np.sqrt(n_i+2) *\
                                                    V[n_j, n_i+2, l] * V[n_j, n_i, l] *\
                                                    np.exp(-e * beta)

                    #i!=j
                    else:
                        for n_i in range(basis_size-1):
                            for n_j in range(basis_size-1):
                                for l, e in enumerate(E):
                                    if i == 0 and j == 1:
                                        rdm_IJ[i, j] += np.sqrt(n_i+1) * np.sqrt(n_j+1) *\
                                                    V[n_i+1, n_j+1, l] * V[n_i, n_j, l] *\
                                                    np.exp(-e * beta)
                                    elif i == 1 and j == 0:
                                        rdm_IJ[i, j] += np.sqrt(n_i+1) * np.sqrt(n_j+1) *\
                                                    V[n_j+1, n_i+1, l] * V[n_j, n_i, l] *\
                                                    np.exp(-e * beta)
            rdm_IJ /= Z
            return rdm_IJ

        def cal_rdm_Ij(E, V, Z, basis_size, beta):
            """calculate two body density matrix from FCI"""
            rdm_Ij = np.zeros([self.N, self.N])
            for i in range(self.N):
                for j in range(self.N):
                    #i=j
                    if i == j:
                        for n_i in range(basis_size):
                            for n_j in range(basis_size):
                                for l, e in enumerate(E):
                                    if i == 0:
                                        rdm_Ij[i, i] += n_i *\
                                                   V[n_i, n_j, l] * V[n_i, n_j, l] *\
                                                   np.exp(-e * beta)
                                    else:
                                        rdm_Ij[i, i] += n_i *\
                                                    V[n_j, n_i, l] * V[n_j, n_i, l] *\
                                                    np.exp(-e * beta)

                    #i!=j
                    else:
                        for n_i in range(basis_size-1):
                            for n_j in range(1, basis_size):
                                for l, e in enumerate(E):
                                    if i == 0 and j == 1:
                                        rdm_Ij[i, j] += np.sqrt(n_i+1) * np.sqrt(n_j) *\
                                                    V[n_i+1, n_j-1, l] * V[n_i, n_j, l] *\
                                                    np.exp(-e * beta)
                                    elif i == 1 and j == 0:
                                        rdm_Ij[i, j] += np.sqrt(n_i+1) * np.sqrt(n_j) *\
                                                    V[n_j-1, n_i+1, l] * V[n_j, n_i, l] *\
                                                    np.exp(-e * beta)
            rdm_Ij /= Z
            return rdm_Ij

        def cal_rdm_iJ(E, V, Z, basis_size, beta):
            """calculate two body density matrix from FCI"""
            rdm_iJ = np.zeros([self.N, self.N])
            for i in range(self.N):
                for j in range(self.N):
                    #i=j
                    if i == j:
                        for n_i in range(basis_size):
                            for n_j in range(basis_size):
                                for l, e in enumerate(E):
                                    if i == 0:
                                        rdm_iJ[i, i] += (n_i + 1) *\
                                                   V[n_i, n_j, l] * V[n_i, n_j, l] *\
                                                   np.exp(-e * beta)
                                    else:
                                        rdm_iJ[i, i] += (n_i + 1) *\
                                                    V[n_j, n_i, l] * V[n_j, n_i, l] *\
                                                    np.exp(-e * beta)

                    #i!=j
                    else:
                        for n_i in range(1, basis_size):
                            for n_j in range(basis_size-1):
                                for l, e in enumerate(E):
                                    if i == 0 and j == 1:
                                        rdm_iJ[i, j] += np.sqrt(n_i) * np.sqrt(n_j+1) *\
                                                    V[n_i-1, n_j+1, l] * V[n_i, n_j, l] *\
                                                    np.exp(-e * beta)
                                    elif i == 1 and j == 0:
                                        rdm_iJ[i, j] += np.sqrt(n_i) * np.sqrt(n_j+1) *\
                                                    V[n_j+1, n_i-1, l] * V[n_j, n_i, l] *\
                                                    np.exp(-e * beta)
            rdm_iJ /= Z
            return rdm_iJ


        def map_quasi_1_RDM(DM):
            """map quasi 1-RDM from physical density matrices"""
            RDM_1 = {}
            RDM_1["a"] = DM[(1, 0)] / self.cosh_theta
            RDM_1["b"] = DM[(0, 1)] / self.sinh_theta
            return RDM_1

        def map_quasi_2_RDM(DM):
            """map quasi 2-RDM from physical densitry matrix"""
            RDM_2 = {}
            RDM_2["ab"] = DM["iJ"] / np.einsum('i,j->ij', self.cosh_theta, self.sinh_theta)
            RDM_2["ba"] = DM["Ij"] / np.einsum('i,j->ij', self.sinh_theta, self.cosh_theta)
            RDM_2["aa"] = DM[(2, 0)] / np.einsum('i,j->ij', self.cosh_theta, self.cosh_theta)
            RDM_2["bb"] = DM[(0, 2)] / np.einsum('i,j->ij', self.sinh_theta, self.sinh_theta)
            return RDM_2

        def map_T_from_RDM(DM):
            """map initial T amplitude from quasi particle density matrix"""
            T = {}
            T[1] = DM[1]
            T[2] = DM[2] - np.einsum('i,j->ij', T[1], T[1])
            return T
        beta_initial = 1. / (self.Kb * T_initial)
        # calculation parttion function as normalization factor
        Z = sum(np.exp(-beta_initial * self.E_val))

        # reshape eigen vector
        self.V_val = self.V_val.reshape([basis_size, basis_size, basis_size**self.N])

        # initialize physical density matrix
        DM_phys = {}
        DM_phys[(1, 0)] = cal_rdm_I(self.E_val, self.V_val, Z, basis_size, beta_initial)
        DM_phys[(0, 1)] = cal_rdm_i(self.E_val, self.V_val, Z, basis_size, beta_initial)
        DM_phys['Ij'] = cal_rdm_Ij(self.E_val, self.V_val, Z, basis_size, beta_initial)
        DM_phys['iJ'] = cal_rdm_iJ(self.E_val, self.V_val, Z, basis_size, beta_initial)
        DM_phys[(2, 0)] = cal_rdm_IJ(self.E_val, self.V_val, Z, basis_size, beta_initial)
        DM_phys[(0, 2)] = cal_rdm_ij(self.E_val, self.V_val, Z, basis_size, beta_initial)

        print("Physical density matrix mapped from FCI:")
        for block in DM_phys.keys():
            print("{:}:\n{:}".format(block, DM_phys[block]))

        # initial quasi density matrix
        RDM_quasi = {}
        RDM_quasi[1] = self.merge_linear(map_quasi_1_RDM(DM_phys))
        RDM_quasi[2] = self.merge_quadratic(map_quasi_2_RDM(DM_phys))

        print("Quasi particle density matrix mapped from FCI")
        for block in RDM_quasi.keys():
            print("{:}:\n{:}".format(block, RDM_quasi[block]))

        # initial T amplitdue
        T_initial = map_T_from_RDM(RDM_quasi)
        T_initial[0] = np.log(Z)
        print("initial T amplitude")
        for block in T_initial.keys():
            print("{:}:\n{:}".format(block, T_initial[block]))

        return T_initial


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
            R += 0.5 * np.einsum('kl,kl->', H[(0, 2)], T[2])
            R += 0.5 * np.einsum('kl,k,l->', H[(0, 2)], T[1], T[1])

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

        def f_t_i(H, T):
            """return residue R_i"""

            # initialize
            R = np.zeros(2*N)

            # non zero initial value of R
            R += H[(1, 0)]

            # linear
            R += np.einsum('ki,k->i', H[(1, 1)], T[1])

            # quadratic
            R += np.einsum('k,ki->i', H[(0, 1)], T[2])
            R += np.einsum('kl,k,li->i', H[(0, 2)], T[1], T[2])

            return R

        def f_t_Ij(H, T):
            """return residue R_Ij"""

            # initialize
            R = np.zeros([2*N, 2*N])

            # first term
            R += H[(1, 1)]

            # quadratic
            R += np.einsum('ik,kj->ij', H[(0, 2)], T[2])

            return R

        def f_t_IJ(H, T):
            """return residue R_IJ"""

            # initialize as zero
            R = np.zeros([2*N, 2*N])

            # quadratic
            R += H[(0, 2)]
            return R

        def f_t_ij(H, T):
            """return residue R_ij"""

            # # initialize as zero
            R = np.zeros([2*N, 2*N])

            # if self.hamiltonian_truncation_order >= 2:

            # quadratic
            R += H[(2, 0)]  # h term
            R += np.einsum('kj,ki->ij', H[(1, 1)], T[2])
            R += np.einsum('ki,kj->ij', H[(1, 1)], T[2])
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

    def TFCC_integration(self, T_initial, T_final, N):
        """conduct TFCC imaginary time integration to calculation thermal perperties"""
        # map initial T amplitude
        T_amplitude = self._map_initial_T_amplitude(T_initial=T_initial)
        beta_initial = 1. / (self.Kb * T_initial)
        beta_final = 1. / (self.Kb * T_final)
        step = (beta_final - beta_initial) / N
        self.temperature_grid = 1. / (self.Kb * np.linspace(beta_initial, beta_final, N))
        self.partition_function = []
        self.internal_energy = []
        # thermal field imaginary time propagation
        for i in range(N):
            Residual = self.CC_residue(self.H_tilde_reduce, T_amplitude)
            # energy
            E = Residual[0]
            self.internal_energy.append(E)
            # partition function
            Z = np.exp(T_amplitude[0])
            self.partition_function.append(Z)
            # update amplitudes
            T_amplitude[0] -= Residual[0] * step
            T_amplitude[1] -= Residual[1] * step
            T_amplitude[2] -= Residual[2] * step
            print("step {:}:".format(i))
            print("Temperature: {:} K".format(self.temperature_grid[i]))
            print("thermal internal energy: {:} cm-1".format(E))
            print("partition function: {:}".format(Z))

        # store data
        thermal_data = {"temperature": self.temperature_grid, "internal energy": self.internal_energy, "partition function": self.partition_function}
        df = pd.DataFrame(thermal_data)
        df.to_csv("thermal_data_TFCC.csv", index=False)

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
        # plt.show()
        plt.savefig("partition_function.png")
