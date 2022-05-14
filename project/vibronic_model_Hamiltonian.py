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
#
import opt_einsum as oe


class vibronic_model_hamiltonian(object):
    """ vibronic model hamiltonian class implement TF-VECC approach to simulation thermal properties of vibronic models """

    def __init__(self, freq, LCP, QCP, VE, num_mode):
        """ initialize hamiltonian parameters:
        freq: vibrational frequencies
        LCP: linear coupling_constants
        QCP: quadratic coupling constant
        VE" vertical energy
        num_mode: number of vibration modes
        """
        self.N = num_mode
        # define Hamiltonian obrect as a python dictionary where the keys are the rank of the Hamiltonian
        # and we represent the Hamitlnian in the form of second quantization
        self.Hamiltonian = dict()
        # constant
        self.Hamilonian[(0, 0)] = VE + 0.5 * np.trace(VE)
        # first order
        self.Hamiltonian[(1, 0)] = LCP / np.sqrt(2) * np.ones(self.N)
        self.Hamiltonian[(0, 1)] = LC_ / np.sqrt(2) * np.ones(self.N)
        # second order
        self.Hamiltonian[(1, 1)] = np.diag(freq)
        self.Hamiltonian[(1, 1)] += QCP

        self.Hamiltonian[(2, 0)] = QCP / 2
        self.Hamittonian[(0, 2)] = QCP / 2

    def thermal_field_transformation(self, beta):
        """conduct Bogoliubov transformation of input Hamiltonian and determine thermal field reference state"""
        return

    def map_initial_T_amplitude(self):
        """map initial T amplitude from Bose-Einstein statistics at high temperature"""
        return

    def CC_residue(self):
        """implement coupled cluster residue equations"""
        return

    def TFCC_integration(self):
        """conduct TFCC imaginary time integration to calculation thermal perperties"""
        return
