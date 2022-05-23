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
project_dir = abspath(join(dirname(__file__), '/home/paulie/hot-molecule'))
sys.path.insert(0, project_dir)

# local import
from project.vibronic_model_Hamiltonian import vibronic_model_hamiltonian


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
    # model.thermal_field_transformation(Temp=10000.)
    # model.reduce_H_tilde()
    # model._map_initial_T_amplitude(T_initial=1000.)
    # run FCI calculation of ACF
    # time_FCI, ACF_FCI = model.FCI_solution(time=np.linspace(0, 100, 10001))
    # store ACF data of FCI calculation
    # model.store_ACF_data(time_FCI, ACF_FCI, name="ACF_single_surface_model_FCI")
    # run VECC calculation of ACF
    time_CC, ACF_CC = model.VECC_integration(t_final=100, num_steps=10000, CI_flag=False, mix_flag=True)
    model.store_ACF_data(time_CC, ACF_CC, name="ACF_single_surface_model_mix_CC_CI")

    return


if (__name__ == '__main__'):
    main()
