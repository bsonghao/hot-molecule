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
    num_mode = 3

    # constant term (in eV)
    VE = 1.3

    # linear coupling constant (in eV)
    LCP = np.array([0.32, 0.41, 0.56])

    # quadratic coupling constant
    QCP = np.array([[0.030, 0.001, 0.002],
                    [0.001, 0.040, 0.005],
                    [0.002, 0.005, 0.050]])

    # frequancies (in eV)
    Freq = np.array([0.15, 0.20, 0.35])

    # initialize the Hamiltonian
    model = vibronic_model_hamiltonian(Freq, LCP, QCP, VE, num_mode)

    return


if (__name__ == '__main__'):
    main()
