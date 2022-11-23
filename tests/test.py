# system imports
import io
import time
import os
from os.path import abspath, join, dirname, basename
import sys
import cProfile
import pstats

# third party import
import numpy as np

# import the path to the package
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/hot-molecule'))
sys.path.insert(0, project_dir)
outputdir =  '/Users/pauliebao/hot-molecule/data/'

# local import
import Project
from Project.vibronic_model_Hamiltonian import vibronic_model_hamiltonian


def main():
    """main function that run TF-VECC simulation"""
    # Hamiltonian model parameters
    # define number of vibrational model
    num_mode = 2

    # constant term (in eV)
    VE = 1.3

    # linear coupling constant (in eV)
    LCP = np.array([0.32, 0.41])

    # quadratic coupling constant
    QCP = np.array([[0.030, 0.001],
                    [0.001, 0.040]])

    # frequancies (in eV)
    Freq = np.array([0.15, 0.20])

    # initialize the Hamiltonian
    model = vibronic_model_hamiltonian(Freq, LCP, QCP, VE, num_mode)
    model.thermal_field_transformation(Temp=1e3)
    model.reduce_H_tilde()
    model.TFCC_integration(T_initial=1e4, T_final=100, N=10000, output_path=outputdir)
    model.sum_over_states(basis_size=10, output_path=outputdir)
    model._map_initial_T_amplitude_from_FCI(T_initial=1000, basis_size=10)

    model.plot_thermal()
    #model._map_initial_T_amplitude(T_initial=1000.)

    return


if (__name__ == '__main__'):
    main()
