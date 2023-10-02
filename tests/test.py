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
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/hot-molecule/'))
sys.path.insert(0, project_dir)

# local import
from project.vibronic_model_Hamiltonian import vibronic_model_hamiltonian


def main():
    """main function that run TF-VECC simulation"""
    # Hamiltonian model parameters
    # define number of vibrational model
    num_mode = 2

    num_surface = 2

    # constant term (in eV)
    VE = np.zeros([num_surface, num_surface])
    VE[0, 0] = 0.6
    VE[1, 1] = 3.1
    # linear coupling constant (in eV)
    LCP = np.zeros([num_surface, num_surface, num_mode])
    LCP[0, 0, :] = np.array([0.28, 0.61]) / 2
    LCP[0, 1, :] = np.array([0.18, 0.38]) / 2
    LCP[1, 0, :] = np.array([0.18, 0.38]) / 2
    LCP[1, 1, :] = np.array([0.34, 0.39]) / 2

    # quadratic coupling constant
    QCP = np.zeros([num_surface, num_surface, num_mode, num_mode])
    QCP[0, 0, :] = np.array([[0.007341, 0.0004], [0.0004, 0.008899]])*0
    QCP[0, 1, :] = np.array([[0.001867, 0.0005], [0.0005, 0.006013]])*0
    QCP[1, 0, :] = np.array([[0.001867, 0.0005], [0.0005, 0.006013]])*0
    QCP[1, 1, :] = np.array([[0.002257, 0.0001], [0.0001, 0.007940]])*0

    # frequancies (in eV)
    Freq = np.array([0.21, 0.43])

    # transition dipole moment
    TDM = np.array([0.1, 0.1])

    # initialize the Hamiltonian
    model = vibronic_model_hamiltonian(Freq, LCP, QCP, VE, TDM, num_mode, num_surface)
    model.construct_full_Hamiltonian_in_HO_basis(basis_size=10)
    # model.calculate_state_pop_from_FCI(time=np.linspace(0,100,10000), basis_size=10)
    model.calculate_ACF_from_FCI(time=np.linspace(0,100,10001), basis_size=10, name="FCI_ACF")


    return


if (__name__ == '__main__'):
    main()
