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

# local import
from project.vibronic import vIO, VMK
from project.log_conf import log
from project.temporary_trial_solver import new_solve_ivp # import the RK integrator



class vibronic_model_hamiltonian(object):
    """ vibronic model hamiltonian class implement TNOE approach to simulation
    thermal properties of vibronic models. """
    def __init__(self, model, name, truncation_order, FC=False):
        """ initialize hamiltonian parameters:
        model: an object the contains parameters of the vibronic model Hamiltonian
        name: name of the vibronic model
        truncation_order: truncation order of the vibronic model Hamiltonian
        GS_energy: ground state energy
        """

        # initialize the Hamiltonian parameters as object instances
        # number of potential energy surfaces
        self.A = model[VMK.A]

        # number of normal modes
        self.N = model[VMK.N]

        # Hamiltonian truncation order
        self.truncation_order = truncation_order

        # vibronic model
        self.model = model

        # name of the vibronic model
        self.name = name

        # A flag to determine if to turn on/off the off-diagonal elements
        self.FC = FC

        # Boltzmann constant (eV K-1)
        self.Kb = 8.61733326e-5

        # initialize the vibronic model Hamiltonian
        # define coefficient tensors
        A, N = self.A, self.N

        self.H = {
            (0, 0): np.zeros((A, A)),
            (0, 1): np.zeros((A, A, N)),
            (1, 0): np.zeros((A, A, N)),
            (1, 1): np.zeros((A, A, N, N)),
            (0, 2): np.zeros((A, A, N, N)),
            (2, 0): np.zeros((A, A, N, N))
        }
        log.info("Zero point energy:{:}".format(0.5 * np.sum(self.model[VMK.w])))
        # constant term
        self.H[(0, 0)] += self.model[VMK.E].copy()
        ## H.O. ground state energy
        for a in range(A):
            self.H[(0, 0)][a, a] += 0.5 * np.sum(self.model[VMK.w])

        # frequencies
        for a, j in it.product(range(A), range(N)):
            self.H[(1, 1)][a, a, j, j] += self.model[VMK.w][j]

        # linear terms
        if self.truncation_order >= 1:
            self.H[(0, 1)] += self.model[VMK.G1] / np.sqrt(2)
            self.H[(1, 0)] += self.model[VMK.G1] / np.sqrt(2)

        # quadratic terms
        if self.truncation_order >= 2:
            ## quadratic correction to H.O. ground state energy
            self.H[(0, 0)] += 0.25 * np.trace(self.model[VMK.G2], axis1=2, axis2=3)

            # quadratic terms
            self.H[(1, 1)] += self.model[VMK.G2]
            self.H[(2, 0)] += self.model[VMK.G2]
            self.H[(0, 2)] += self.model[VMK.G2]

        # if computing Frank-Condon(FC) model then
        # zero out all electronically-diagonal terms
        if self.FC:
            for a, b in it.product(range(A), repeat=2):
                if a != b:
                    self.H[(1, 1)][a, b, :] = np.zeros([N, N])
                    self.H[(2, 0)][a, b, :] = np.zeros([N, N])
                    self.H[(0, 2)][a, b, :] = np.zeros([N, N])
                    self.H[(1, 0)][a, b, :] = np.zeros([N, ])
                    self.H[(0, 1)][a, b, :] = np.zeros([N, ])

        log.info("number of vibrational mode {:}".format(self.N))
        log.info("##### Hamiltonian parameters ######")
        for rank in self.H.keys():
            log.info("Block {:}:".format(rank))
            if rank != (0, 0):
                for a, b in it.product(range(self.A), repeat=2):
                    log.info("surface ({:}, {:}):\n{:}".format(a, b, self.H[rank][a, b, :]))
            else:
                log.info(self.H[rank])

        # check hermicity of the Hamiltonian
        # constant term keys
        constants = [(0, 0)]
        # linear term keys
        linears = [(1, 0), (0, 1)]
        # quadratic term keys
        quadratics = [(2, 0), (1, 1), (0, 2)]
        # loop over each block of the Hamiltonian
        for rank in self.H.keys():
            if rank in constants:
                assert np.allclose(self.H[rank], np.transpose(self.H[rank]))
            elif rank in linears:
                assert np.allclose(self.H[rank], np.transpose(self.H[rank], (1, 0, 2)))
            elif rank in quadratics:
                assert np.allclose(self.H[rank], np.transpose(self.H[rank], (1, 0, 3, 2)))

        log.info("Boltzmann constant: {:} eV K-1".format(self.Kb))

        log.info("### End of Hamiltonian parameters ####")

    def thermal_field_transform(self, T_ref):
        """
        conduct Bogoliubov transfrom of the physical hamiltonian
        T_ref: temperature for the thermal field reference state
        """
        # calculate inverse temperature
        self.T_ref = T_ref
        beta = 1. / (self.Kb * T_ref)
        # define Bogliubov transformation based on Bose-Einstein statistics
        self.cosh_theta = 1. / np.sqrt((np.ones(self.N) - np.exp(-beta * self.model[VMK.w])))
        self.sinh_theta = np.exp(-beta * self.model[VMK.w] / 2.) / np.sqrt(np.ones(self.N) - np.exp(-beta * self.model[VMK.w]))

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

        log.info("###### Bogliubov transformed Hamiltonian ########")
        for rank in self.H_tilde.keys():
            if rank == (0, 0):
                log.info("Rank:{:}\n{:}".format(rank, self.H_tilde[rank]))
            else:
                for block in self.H_tilde[rank].keys():
                    log.info("Rank:{:} Block:{:}\n{:}".format(rank, block, self.H_tilde[rank][block]))
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

        log.info("##### Bogliubov transformed (fictitous) Hamiltonian after merge blocks ######")
        for rank in self.H_tilde_reduce.keys():
            log.info("Block {:}: \n {:}".format(rank, self.H_tilde_reduce[rank]))

        return

    def _map_initial_amplitude(self, T_initial=1000):
        """map initial T amplitude from Bose-Einstein statistics at high temperature"""
        def map_z0_amplitude(beta):
            """map t_0 amplitude from H.O. partition function"""
            z_0 = np.eye(A)
            for a in range(A):
                for i in range(N):
                    z_0[a, a] *= 1 / (1 - np.exp(-beta * self.model[VMK.w][i]))
                z_0[a, a] *= np.exp(-beta*(self.H[(0, 0)][a, a]-sum(self.H[(1, 0)][a, a, :]**2 / self.model[VMK.w])))

            return z_0

        def map_t1_amplitude():
            """map initial t_1 and t^1 amplitude from linear displacements"""
            # initialzie t_i and t_I tensors
            t_i = np.zeros([A, 2*N])
            for x in range(A):
                t_i[x, :N] -= self.H[(0, 1)][x, x, :] / self.model[VMK.w] / self.cosh_theta
                t_i[x, N:] -= self.H[(0, 1)][x, x, :] / self.model[VMK.w] / self.sinh_theta

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
            for x in range(A):
                for i, j in it.product(range(2 * N), repeat=2):
                    t_2_new[x, i, j] = 0.5 * (t_2[x, i, j] + t_2[x, j, i])

            return t_2_new

        N, A = self.N, self.A
        beta_initial = 1. / (self.Kb * T_initial)

        # calculation BE occupation number at initial beta
        BE_occ = np.ones(self.N) / (np.ones(self.N) - np.exp(-beta_initial * self.model[VMK.w]))
        log.info("BE occupation number:{:}".format(BE_occ))

        # map initial T amplitudes
        initial_T_amplitude = {}
        initial_T_amplitude[1] = map_t1_amplitude()
        initial_T_amplitude[2] = map_t2_amplitude()


        # map initial Z amplitudes
        initial_Z_amplitude={}
        initial_Z_amplitude[0] = map_z0_amplitude(beta_initial)
        initial_Z_amplitude[1] = np.zeros([A, A, 2*N])
        initial_Z_amplitude[2] = np.zeros([A, A, 2*N, 2*N])

        log.info("### initialize T amplitude ###")
        for block in initial_T_amplitude.keys():
            log.info("Block:{:}\n{:}".format(block, initial_T_amplitude[block]))

        log.info("### initialize Z amplitude ###")
        for block in initial_Z_amplitude.keys():
            log.info("Block:{:}\n{:}".format(block, initial_Z_amplitude[block]))

        log.info("###  T and Z amplitude initialized successfully! ###")

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

            # # initialize as zero
            R = np.zeros([A, A, 2*N, 2*N])

            # if self.hamiltonian_truncation_order >= 2:

            # quadratic
            R += H[(2, 0)]  # h term
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

    def cal_net_residual(self, H_args, Z_args):
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

        return residual

    def cal_T_Z_residual(self, T_args, Z_args):
        """calculation T and Z residual"""
        N = self.N
        def cal_T_residual(H_args, Z_args):
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

            return residual

        def cal_Z_residual(R_args, Z_args, dT_args):
            """calculation Z residual by strustracting T residual from the net residual"""
            def cal_dZ_0():
                """(0, 0) block of Z residual"""
                R = R_args[0]
                return R

            def cal_dZ_I():
                """(1, 0) block of Z residual"""
                R = R_args[1]
                R -= np.einsum('i,a->ai', dT_args[1], Z_args[0])
                return R

            def cal_dZ_IJ():
                """(2, 0) block of Z residual"""
                R = R_args[2]
                R -= np.einsum('i,aj->aij', dT_args[1], Z_args[1])
                R -= np.einsum('j,ai->aij', dT_args[1], Z_args[1])
                R -= np.einsum('ij,a->aij', dT_args[2], Z_args[0])
                return R

            residual = {}
            residual[0] = cal_dZ_0()
            residual[1] = cal_dZ_I()
            residual[2] = cal_dZ_IJ()

            return residual
        # calculate similarity transfromed Hamiltonian
        H_bar = self.sim_trans_H(self.H_tilde_reduce, T_args)
        # calculation net residual
        net_residual = self.cal_net_residual(H_bar, Z_args)
        # calculate T residual
        t_residual = cal_T_residual(H_bar, Z_args)
        # calculate Z residual
        z_residual = cal_Z_residual(net_residual, Z_args, t_residual)

        return t_residual, z_residual


    def TFCC_integration(self, output_path, T_initial, T_final, N_step, debug_flag=False):
        """
        conduct TFCC imaginary time integration to calculation thermal
        properties
        T_initial: initial temperature of the integration
        T_final: final temperature of the integration
        N_step: number of the numerical steps

        This is a primary implementation with first order Euler
        method with fixed numerical integration step
        """
        A, N = self.A, self.N
        # map initial T amplitude
        T_amplitude, Z_amplitude = self._map_initial_amplitude(T_initial=T_initial)

        beta_initial = 1. / (self.Kb * T_initial)
        beta_final = 1. / (self.Kb * T_final)
        step = (beta_final - beta_initial) / N_step
        self.temperature_grid = 1. / (self.Kb * np.linspace(beta_initial, beta_final, N_step))
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
                    log.info("T{:}:{:}".format(block, T_residual[block]))
                T_amplitude[block] -= T_residual[block] * step
            for block in Z_amplitude.keys():
                if debug_flag:
                    log.info("Z{:}:{:}".format(block, Z_residual[block]))
                Z_amplitude[block] -= Z_residual[block] * step

            # calculate partition function
            # log.info("Z[(0, 0)]:\n{:}".format(Z_amplitude[0, 0]))
            # log.info("step:{:}".format(step))
            Z = np.trace(Z_amplitude[0])
            self.partition_function.append(Z)
            # calculate thermal internal energy
            E = np.trace(Z_residual[0]) / Z
            self.internal_energy.append(E)
            if (i+1) / N_step * 100 % 1 == 0:
                log.info("step {:}:".format(i))
                log.info("Temperature: {:} K".format(self.temperature_grid[i]))
                log.info("max z_0 amplitude:{:}".format(abs(Z_amplitude[0]).max()))
                log.info("max z_1 amplitude:{:}".format(abs(Z_amplitude[1]).max()))
                log.info("max z_2 amplitude:{:}".format(abs(Z_amplitude[2]).max()))
                log.info("max t_1 ampltiude:{:}".format(abs(T_amplitude[1]).max()))
                log.info("max t_2 amplitude:{:}".format(abs(T_amplitude[2]).max()))
                log.info("thermal internal energy: {:} ev".format(E))
                log.info("partition function: {:}".format(Z))

        # store data
        thermal_data = {"temperature": self.temperature_grid, "internal energy": self.internal_energy, "partition function": self.partition_function}
        df = pd.DataFrame(thermal_data)
        df.to_csv(output_path+"{:}_thermal_data_TFCC.csv".format(self.name), index=False)

        return

    def _unravel_y_tensor(self, y_tensor):
        """ Restore the original shape of the flattened y tensor """

        A, N = self.A, self.N  # for brevity

        # all return tensors start as None
        Z = {0: None, 1: None, 2: None}
        T = {1: None, 2: None}

        # ------------------------------ restore z tensor ----------------------------

        # constant terms
        start_constant_slice_index = 0
        end_constant_slice_index = start_constant_slice_index + A * A
        Z[0] = np.reshape(
            y_tensor[start_constant_slice_index:end_constant_slice_index],
            newshape=(A, A)
        )

        # linear terms
        start_linear_slice_index = end_constant_slice_index
        end_linear_slice_index = start_linear_slice_index + A * A * 2 * N
        Z[1] = np.reshape(
            y_tensor[start_linear_slice_index: end_linear_slice_index],
            newshape=(A, A, 2 * N)
        )
        # quadratic terms
        start_quadratic_slice_index = end_linear_slice_index
        end_quadratic_slice_index = start_quadratic_slice_index + A * A * 2 * N * 2 *  N
        Z[2] = np.reshape(
            y_tensor[start_quadratic_slice_index: end_quadratic_slice_index],
            newshape=(A, A, 2 * N, 2 * N)
        )

        # ------------------------------ restore t tensor ----------------------------

        # need a bit of logic to figure out the final Z slice index

        start_linear_slice_index = end_quadratic_slice_index

        end_linear_slice_index = start_linear_slice_index + A * 2 * N
        T[1] = np.reshape(
            y_tensor[start_linear_slice_index: end_linear_slice_index],
            newshape=(A, 2 * N)
        )

        # quadratic terms



        start_quadratic_slice_index = end_linear_slice_index



        end_quadratic_slice_index = start_quadratic_slice_index + A * 2 * N * 2 * N



        T[2] = np.reshape(



            y_tensor[start_quadratic_slice_index: end_quadratic_slice_index],



            newshape=(A, 2 * N, 2 * N)



            )

        return Z, T

    def _ravel_y_tensor(self, Z, T):
        """ Flatten the `t` and `z` tensors into a 1D array """
        # ravel z tensor
        z_tensor_list = [Z[0].ravel(), ]
        z_tensor_list.append(Z[1].ravel())
        z_tensor_list.append(Z[2].ravel())

        # ravel t tensor
        t_tensor_list = []
        t_tensor_list.append(T[1].ravel())
        t_tensor_list.append(T[2].ravel())

        # the t tensor should come before the z tensor
        y_tensor = np.concatenate((*z_tensor_list, *t_tensor_list))

        return y_tensor

    def _print_integration_progress(self, time, t_init, t_final, *args):
        """ Prints to stdout every 1e4 steps or if current fs value is a multiple of (0.1 * `t_final`). """

        # unpack any args we wish to print
        Z, T = args

        self.counter += 1
        self.last_counter += 1

        delta_time = time - t_init
        delta_t_final = t_final - t_init
        percent = delta_time / delta_t_final * 100

        time_is_a_multiple_of_ten_percent_of_t_final = np.isclose((round(percent, 1) % 10), 0)

        percent_is_not_zero = (percent != 0)

        print_flag = bool(
            self.last_counter >= int(1e4)
            or (time_is_a_multiple_of_ten_percent_of_t_final
            and percent_is_not_zero)
        )

        if print_flag:
            log.info(
                f"On integration step {self.counter:<8d} at {1./(self.Kb * time):>9.4f}K\n"
                f"{percent:>9.3f} % integration is completed\n"
                f"Z = {self.partition_function[-1][1]:>9.4f}\n"
                f"E = {self.internal_energy[-1][1]:>9.4f}\n"
            )

            # -------------------------------------------------------------------------------------
            # print T amplitudes
            t_amplitude_values_string = f'\n'
            t_amplitude_values_string += f'max single   T[1]  amplitude: {abs(T[1]).max()}\n'
            t_amplitude_values_string += f'max double   T[2]  amplitude: {abs(T[2]).max()}\n'
            log.info(t_amplitude_values_string)
            # -------------------------------------------------------------------------------------
            # print Z amplitudes
            z_amplitude_values_string = f'\n'
            z_amplitude_values_string += f'max constant Z[0]   amplitude: {abs(Z[0]).max()}\n'
            z_amplitude_values_string += f'max single   z[1]   amplitude: {abs(Z[1]).max()}\n'
            z_amplitude_values_string += f'max double   z[2]  amplitude: {abs(Z[2]).max()}\n'
            log.info(z_amplitude_values_string)


            self.last_print = time
            self.last_counter = 0

        return


    def rk45_solve_ivp_integration_function(self, time, y_tensor, t_init, t_final):
        """ Integration function used by `solve_ivp` integrator inside `rk45_integration` method.

        `time` is a float, the value of time for the current integration step
        `y_tensor` is an (n, k) dimensional tensor where the n dimension counts the ode's
        that we are attempting to solve and k can represent multiple time steps to block integrate over
        at the moment we do not do any block integration so k is 1
        """
        A, N = self.A, self.N

        # restore the origin shape of t, z amplitudes from y_tensor
        Z_amplitude, T_amplitude = self._unravel_y_tensor(y_tensor)

        # printing progression
        self._print_integration_progress(time, t_init, t_final, Z_amplitude, T_amplitude)

        T_residual, Z_residual = {}, {}
        for block in T_amplitude.keys():
            T_residual[block] = np.zeros_like(T_amplitude[block])
        for block in Z_amplitude.keys():
            Z_residual[block] = np.zeros_like(Z_amplitude[block])

        for x in range(A):
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

        # calculate partition function
        Z = np.trace(Z_amplitude[0])
        # calculate thermal internal energy
        E = np.trace(Z_residual[0]) / Z

        # flatten the z, t tensors into a 1D array
        delta_y_tensor = self._ravel_y_tensor(Z_residual, T_residual)

        # store thermal properties data
        self.partition_function.append((time, Z))
        self.internal_energy.append((time, E))

        return -delta_y_tensor

    def _postprocess_rk45_integration_results(self, sol, output_path, debug=False):
        """ extract the relevant information from the integrator object `sol` """
        # number of integration steps accepted by the integrator

        log.info(f"RK45 preformed {len(self.partition_function)} integration calculations.")
        log.info(f"RK45 accepted  {len(sol.t)} of those as solutions to the ode's.")
        if debug:
            log.debug(f"Distance we reached when we stopped: {sol.t_events[0]}")

        # Copy the time value arrays
        self.t_cc = sol.t.copy()

        # initialize the arrays to store the thermal properties
        self.partition_function_cc = np.zeros_like(self.t_cc)
        self.internal_energy_cc = np.zeros_like(self.t_cc)

        # log.info(len(self.C_tau_cc))
        # log.info(len(self.C_tau_cc_store))

        # only extract the values which correspond to time steps in the solution
        C_dic_partition_function = {c[0]: c[1] for c in self.partition_function}
        for idx, t in enumerate(sol.t):
            self.partition_function_cc[idx] = C_dic_partition_function[t]

        C_dic_internal_energy = {c[0]: c[1] for c in self.internal_energy}
        for idx, t in enumerate(sol.t):
            self.internal_energy_cc[idx] = C_dic_internal_energy[t]

        log.debug(f"Status message: {sol.message}")  # description of termination reason
        log.debug(f"status: {sol.status}")  # -1: step failed, 0: reached end of tspan, 1: termination event occurred
        log.debug(f"Succeeded?: {sol.success}")  # bool if reached end of interval or termination event occurred

        # store data
        thermal_data = {"temperature": 1. / (self.Kb * sol.t), "internal energy": self.internal_energy_cc, "partition function": self.partition_function_cc}
        df = pd.DataFrame(thermal_data)
        df.to_csv(output_path+"{:}_thermal_data_TFCC.csv".format(self.name), index=False)

        return


    def rk45_integration(self, output_path, T_initial=0., T_final=10., density=1.0, nof_points=10000, debug_flag=False):
        """ Runge-Kutta imaginary time integration

        This is an advanced integration method with the RK numerical integration
        method with adapative numerical integration steps
        """

        # ------------------------------------------------------------------------
        # initialize integration parameters
        # ------------------------------------------------------------------------
        log.info(f"We are going to preform a RK4(5) integration")

        A, N = self.A, self.N  # to reduce line lengths, for conciseness

        # map initial T amplitude from classical limit at hight temperature
        initial_T, initial_Z = self._map_initial_amplitude(T_initial=T_initial)

        # used for debugging purposes to print out the integration steps every n% of integration
        self.counter = 0
        self.last_counter = 0
        self.last_print = 0

        # calculation initial step size for the numerical integration
        beta_init = 1. / (self.Kb * T_initial)
        beta_final = 1. / (self.Kb * T_final)
        step_size = (beta_final - beta_init) / nof_points


        # initialize thermal propeties as empty lists
        self.partition_function = []
        self.internal_energy = []


        # prepare the initial y_tensor
        initial_y_tensor = self._ravel_y_tensor(initial_Z, initial_T)

        # ------------------------------------------------------------------------
        # the integration function called by the `solve_ivp` integrator at each step of integration
        # ------------------------------------------------------------------------


        # set up tolerance for the RK integrator
        relative_tolerance = 1e-012
        absolute_tolerance = 1e-014
        # ------------------------------------------------------------------------
        # call the integrator
        # ------------------------------------------------------------------------

        integration_function = self.rk45_solve_ivp_integration_function

        sol = new_solve_ivp(
            fun=integration_function,  # the function we are integrating
            method="RK45",  # the integration method we are using
            first_step=step_size,  # fix the initial step size
            t_span=(
                beta_init,  # initial time
                beta_final,  # boundary time, integration end point
            ),
            y0=initial_y_tensor,  # initial state - shape (n, )
            args=(beta_init, beta_final, ),  # extra args to pass to `rk45_solve_ivp_integration_function`
            # max_step=self.step_size,  # maximum allowed step size
            rtol=relative_tolerance,  # relative tolerance
            atol=absolute_tolerance,  # absolute tolerance
            store_y_values=False,  # do not store the y values over the integration
            t_eval=None,  # store all the time values that we integrated over
            dense_output=False,  # extra debug information
            # we do not need to vectorize
            # this means to process multiple time steps inside the function `rk45_solve_ivp_integration_function`
            # it would be useful for a method which does some kind of block stepping
            vectorized=False,
        )

        # ------------------------------------------------------------------------
        # now we extract the relevant information from the integrator object `sol`
        # ------------------------------------------------------------------------
        self._postprocess_rk45_integration_results(sol, output_path, debug=debug_flag)

        return
