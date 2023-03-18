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
import json

# import the path to the package
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/hot-molecule'))
sys.path.insert(0, project_dir)
inputdir = '/Users/pauliebao/hot-molecule/data/input_json/'
outputdir =  '/Users/pauliebao/hot-molecule/data/'

# local import
import project
from project.vibronic_model_Hamiltonian import vibronic_model_hamiltonian

def read_in_model(dir, file_name):
    """read in model parameters from json file"""
    json_file =  open(dir+file_name)
    dict = json.load(json_file)
    N = dict["number of modes"]
    A = dict["number of surfaces"]
    VE = np.array(dict["energies"])
    Freq = np.array(dict["frequencies"]) * 10 # enlarge the frequencies by 10 time to mimics the true physical models
    LCP = np.array(dict["linear couplings"])
    json_file.close()

    return N, A, VE, Freq, LCP




def main():
    """main function that run TNOE simulation"""
    # Hamiltonian model parameters
    # define number of vibrational model
    name = "displaced_6.json"

    num_mode, num_surf, VE, Freq, LCP = read_in_model(inputdir, name)

    print("number of surfaces:{:}".format(num_surf))
    print("number of modes:{:}".format(num_mode))
    print("verticle energy (in eV):\n{:}".format(VE))
    print("Frequencies (in eV):\n{:}".format(Freq))
    print("Linear coupling constants (in eV):\n{:}".format(LCP))


    # initialize the Hamiltonian
    model = vibronic_model_hamiltonian(Freq, LCP, VE, num_mode, num_surf, FC=False)
    # Bogoliubov transform the Hamiltonian
    model.thermal_field_transform(T_ref=1e3)
    model.reduce_H_tilde()
    # model._map_initial_amplitude(T_initial=2e3)
    # model.sum_over_states(basis_size=40, output_path=outputdir, T_initial=2e3, T_final=1e2, N_step=10000, compare_with_TNOE=False)
    model.TFCC_integration(T_initial=2e3, T_final=1e2, N_step=10000, output_path=outputdir)
    # model._map_initial_amplitude(T_initial=1e4)

    # model.sum_over_states(basis_size=40, output_path=outputdir, compare_with_TFCC=False, T_grid=np.linspace(100, 10000, 10000))
    # model._map_initial_T_amplitude_from_FCI(T_initial=1000, basis_size=10)
    # model._map_initial_T_amplitude(T_initial=10000)
    # model.plot_thermal()
    #model._map_initial_T_amplitude(T_initial=1000.)

    return


if (__name__ == '__main__'):
    main()
