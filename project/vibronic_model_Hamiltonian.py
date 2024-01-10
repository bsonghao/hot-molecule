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
import itertools as it

# local imports
# import the path to the package
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/hot-molecule/'))
sys.path.insert(0, project_dir)
from project.two_mode_model import model_two_mode
from project.vibronic import vIO, VMK
from project.log_conf import log



class vibronic_model_hamiltonian(object):
    """ vibronic model hamiltonian class implement TF-VECC approach to simulation thermal properties of vibronic models. """

    def __init__(self, model, name, truncation_order):
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
        # number of potential energy surfaces
        self.A = model[VMK.A]
        # number of vibrational modes
        self.N = model[VMK.N]
        # transition dipole moments
        self.TDM = self.E_tdm = model[VMK.etdm][0]
        print("Transition dipole moment:\n{:}".format(self.TDM))
        # name of the model
        self.model_name = name

        # Hamiltonian truncation order
        self.truncation_order = truncation_order

        # vibronic model
        self.model = model

        # Boltzmann constant (eV K-1)
        self.Kb = 8.61733326e-5

        # convert unit of energy
        self.hbar_per_eV = 6.582119569e-16
        self.s_per_fs = 1e-15
        self.unit = self.s_per_fs / self.hbar_per_eV

        # define Hamiltonian obrect as a python dictionary where the keys are the rank of the Hamiltonian
        # and we represent the Hamitlnian in the form of second quantization
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
    def calculate_ACF_from_FCI(self, time, basis_size):
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
        df.to_csv(self.model_name+".csv", index=False)

        # store ACF data in autospec format
        with open(self.name+".txt", 'w') as file:
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
        A, N = self.A, self.N
        print("### compute state population from FCI ###")
        def Cal_state_pop(E, V, b, time, basis_size=10):
            """calculate state population from FCI"""

            V_dagger = np.transpose(V)
            V_dagger = V_dagger.reshape([A, basis_size**N, A * basis_size**N])

            # reshape the eignvector V
            V = V.reshape([A, basis_size**N, A * basis_size**N])

            propagator_matrix = np.zeros([len(E), len(E), len(time)], dtype=complex)
            for m, n in it.product(range(len(E), repeat=2):
                propagator_matrix[m, n, :] += np.exp(1j * unit * (E[m]-E[n]) * time)

            pop = np.einsum('lmt, l, xnl, xnm, m->xt', propagator_matrix, V[b, 0, : ], V, V, V[b, 0, :])
            if False:
                pop_ = np.zeros([A, len(time)], dtype=complex)
                for x, l, m, n in it.product(range(A), range(len(E)), range(len(E)), range(basis_size**N)):
                    pop_[x, :] += np.exp(1j * unit * (E[l] - E[m]) * time) * V[b, 0, l] * V[x, n, l] * V[x, n, m] * V[b, 0, m]
                assert np.allclose(pop, pop_)
            return pop

        E, V = self.E, self.V
        print("###construct state population one surface at a time ###")
        state_pop = np.zeros([self.A, self.A, len(time)], dtype=complex)
        for b in range(self.A):
            state_pop[b, :] = Cal_state_pop(E, V, b, time, basis_size=basis_size)
        print("### store state pululation data to csv ###")
        # store state pop data
        for b in range(self.A):
            temp = {"time": time}
            for a in range(self.A):
                temp[str(a)] = state_pop[b, a, :]
            df = pd.DataFrame(temp)
            name = "{:}_state_pop_for_surface_{:}_from_FCI.csv".format(self.model_name, b)
            df.to_csv(name, index=False)
        print("### state population is successfully computed from FCI")

        return

    def double_similarity_transform(self, T_args, CI_op, b):
        """
        Perform doubly similarity transformation of the origin vibronic Hamiltonian
        and Ehrenfest parameterized T residue
        """
        A, N = self.A, self.N
        T_conj = np.conj(T_args)

        def first_transform():
            """perform first similarity transformation: (H e^T)_conn"""
            def f_t_0(H, T):
                """return residue R_0"""

                # initialize as zero
                R = np.zeros([A, A], dtype=complex)

                # constant
                R += H[(0, 0)]

                # linear
                R += np.einsum('abk,k->ab', H[(0, 1)], T)
                # quadratic
                R += 0.5 * np.einsum('abkl,k,l->ab', H[(0, 2)], T, T)

                return R

            def f_t_I(H, T):
                """return residue R_I"""

                # initialize as zero
                R = np.zeros([A, A, N], dtype=complex)

                # linear
                R += H[(0, 1)]

                # quadratic
                R += np.einsum('abik,k->abi', H[(0, 2)], T)

                return R

            def f_t_i(H, T):
                """return residue R_i"""

                # initialize
                R = np.zeros([A, A, N], dtype=complex)

                # non zero initial value of R
                R += H[(1, 0)]

                # linear
                R += np.einsum('abki,k->abi', H[(1, 1)], T)

                return R

            def f_t_Ij(H, T):
                """return H[(1, 1)]"""

                return H[(1, 1)]

            def f_t_IJ(H, T):
                """return H[(0, 2)]"""
                return H[(0, 2)]

            def f_t_ij(H, T):
                """return H[(2, 0)]"""
                return H[(2, 0)]
            # compute similarity transformed Hamiltonian over e^T
            sim_h = {}
            sim_h[(0, 0)] = f_t_0(self.H, T_args)
            sim_h[(0, 1)] = f_t_I(self.H, T_args)
            sim_h[(1, 0)] = f_t_i(self.H, T_args)
            sim_h[(1, 1)] = f_t_Ij(self.H, T_args)
            sim_h[(0, 2)] = f_t_IJ(self.H, T_args)
            sim_h[(2, 0)] = f_t_ij(self.H, T_args)

            return sim_h


        def second_transform(H_bar):
            """perform second similarity transformation: (e^T* H_bar)_conn"""
            def f_s_0():
                """return constant residue"""
                # initialize as zero
                R = np.zeros([A, A], dtype=complex)

                R += H_bar[(0, 0)]

                R += np.einsum('k,abk->ab', T_conj, H_bar[(1, 0)])

                R += 0.5 * np.einsum('k,l,abkl->ab', T_conj, T_conj, H_bar[(2, 0)])

                return R

            def f_s_I():
                """return residue R_I"""
                R = np.zeros([A, A, N], dtype=complex)

                R += H_bar[(1, 0)]

                R += np.einsum('k,abik->abi', T_conj, H_bar[(2, 0)])

                return R

            def f_s_i():
                """return residue R_i"""
                R = np.zeros([A, A, N], dtype=complex)

                R += H_bar[(0, 1)]

                R += np.einsum('k,abki->abi', T_conj, H_bar[(1, 1)])

                return R

            def f_s_Ij():
                """return residue R_Ij"""
                return H_bar[(1, 1)]


            output_tensor = {
                    (0, 0): f_s_0(),
                    (1, 0): f_s_I(),
                    (0, 1): f_s_i(),
                    (1, 1): f_s_Ij(),
                    (2, 0): H_bar[(2, 0)],
                    (0, 2): H_bar[(0, 2)],
            }
            return output_tensor


        def _compute_t_residual_new():
            """compute t from Ehrenfest parameterization using the new scheme (weight C)"""

            C_0_conj = np.conj(CI_op[:,0])
            C_0 = CI_op[:,0]
            weight = np.einsum('y,y->', C_0_conj, C_0)

            # single t residue
            dT = np.einsum('y,yxi,x->i', C_0_conj, H_bar_tilde[(1, 0)], C_0) / weight

            return dT

        def _sim_trans_dT(dT):
            """ similarity transform dT"""
            output_tensor = {
                (0, 0): np.einsum('k,k->', T_conj, dT),
                (1, 0): dT
            }
            return output_tensor

        def Cal_G(rho, dT):
            """calcuation G by sustraction of T residual contribtion from the doubly transformed Hamiltonian"""
            dT_conj = np.conj(dT)

            G = H_bar_tilde.copy()
            G[(0, 0)] -= rho[(0, 0)] * np.eye(A)
            G[(1, 0)] -= np.einsum('i,xy->xyi', rho[(1, 0)], np.eye(A))
            G[(0, 1)] -= 1j * np.einsum('i, xy-> xyi', dT_conj, np.eye(A))

            return G

        # perform first similarity transfromation of the vibronic Hamiltonian H
        H_bar = first_transform()

        # perform second similarity transformation of the vibronic Hamiltonian
        H_bar_tilde = second_transform(H_bar)

        # calculate T residual using Ehrenfest parameterization
        T_residual = _compute_t_residual_new()

        # calculate G by sustraction of T residual contribution from the doubly transformed Hamiltonian
        # compute rho
        rho = _sim_trans_dT(T_residual)
        # substration rho and dT component from the doubly transformed Hamiltonian
        trans_H = Cal_G(rho, T_residual)

        return trans_H, T_residual

    def resolve_G(self, trans_H, CI_op, basis_size):
        """
        resolve the doubly transformed in finite H.O. basis and compute
        residual for CI operator C (resolved in H.O. basis)
        """
        A, N = self.A, self.N
        def _construct_vibrational_Hamitonian(h):
            """construct vibrational Hamiltonian in H.O. basis"""
            Hamiltonian = np.zeros((basis_size, basis_size, basis_size, basis_size), dtype=complex)
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
            H_FCI = np.zeros([A, basis_size**N, A, basis_size**N], dtype=complex)
            # contruction the full Hamiltonian surface by surface
            for a, b in it.product(range(A), repeat=2):
                h = {}
                for block in self.H.keys():
                    if block != (0, 0):
                        h[block] = trans_H[block][a, b, :]
                    else:
                        h[block] = trans_H[block][a, b]

                H_FCI[a, :, b][:] += _construct_vibrational_Hamitonian(h)

            # H_FCI = H_FCI.reshape(A*basis_size**N, A*basis_size**N)

            return H_FCI

        # resolve the similarity transformed Hamiltonian in H.O. basis
        G_FCI = construct_full_Hamitonian()
        # compute C residual
        # print("shape of G_FCI:{:}".format(G_FCI.shape))
        # print("shape of CI_op:{:}".format(CI_op.shape))
        dC = np.einsum('ynxm,xm->yn', G_FCI, CI_op)


        return dC

    def cal_state_pop(self, CI_op, b):
        """
        calculate the state population from CI operator resolved in finite H.O. basis
        """
        CI_op_conj = np.conj(CI_op)
        population = np.einsum('xn,yn->xy', CI_op_conj, CI_op)
        population /= np.einsum('xn,xn->', CI_op_conj, CI_op)
        return np.diag(population)


    def time_integration(self, t_final, num_steps, basis_size):
        """
        perform time integration
        t_final: time range of the integration
        num_steps: number of steps in the integration
        """
        A, N = self.A, self.N
        dtau = t_final / num_steps
        time = np.linspace(0, t_final, num_steps)
        # initialize T
        T = np.zeros([A, N], dtype=complex)
        # initialize C
        C = np.zeros([A, basis_size, basis_size, A], dtype=complex)
        for x, y in it.product(range(A), repeat=2):
            if x==y:
                C[x, 0, 0, y] = 1
        # merge the vibrational dimensions
        C = C.reshape(A, basis_size**N, A)
        # print("shape of C: {:}".format(C[0,:].shape))

        for b in range(A):
            pop_list = []
            for i in range(num_steps):
                # step 1: double similarity transform the Hamiltonian and calcuation dT
                G_args, dT = self.double_similarity_transform(T[b, :], C[:, :, b], b)
                # step 2: resolve the similarity transform Hamiltonian and CI operator in finite H.O. basis and calculate dC
                dC = self.resolve_G(G_args, C[:, : ,b], basis_size)
                # step 3: calcuate the state population from C in H.O. basis
                pop = self.cal_state_pop(C[:, :,b], b)
                pop_list.append(pop)
                # step 4: update T and C
                T[b, : ] -= dT * 1j * self.unit * dtau
                C[ :, :, b] -= dC * 1j * self.unit * dtau
                if i % 100 == 0:
                    print("At t= {:.4f} fs, state polution for state {:d}:\n{:}".format(dtau * i, b, pop))
            # store state population data
            pop_dic = {"time(fs)": time}
            for a in range(A):
                pop_dic[str(a)] = []
                for i in range(num_steps):
                    pop_dic[str(a)].append(pop_list[i][a])
            df = pd.DataFrame(pop_dic)
            name = "{:}_state_pop_for_surface_{:}_{:d}_basis_per_mode_from_VECC.csv".format(self.model_name, b,  basis_size)
            df.to_csv(name, index=False)


        return
