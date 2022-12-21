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
inputdir = '/Users/pauliebao/hot-molecule/data/vibronic_models/'
outputdir =  '/Users/pauliebao/hot-molecule/data/'

# local import
import Project
from Project.vibronic_model_Hamiltonian import vibronic_model_hamiltonian
from Project.vibronic import vIO, VMK

order_dict = {
    0: "constant",
    1: "linear",
    2: "quadratic",
    3: "cubic",
    4: "quartic",
}


def read_in_model(dir, model_name, order):
    """read in model parameters from MCTDH operator file"""
    # read in entire model
    file_name = "{:}{:}.op".format(dir, model_name)
    raw_model = vIO.read_raw_model_op_file(file_name, highest_order=order, dimension_of_dipole_moments=1)
    # remove any higher order terms
    vIO.remove_higher_order_terms(raw_model, highest_order=order)
    # write new model into the mctdh operator file
    vIO.write_raw_model_op_file(f"{dir}{model_name}_{order_dict[order]}.op", raw_model, highest_order=order)
    # read in our specific model
    path_op = join(inputdir, f"{model_name}_{order_dict[order]}.op")
    model = vIO.extract_excited_state_model_op(path_op, FC=False, highest_order= order,\
                    dimension_of_dipole_moments=1)
    vIO.prepare_model_for_cc_integration(model,order)

    return model




def main():
    """main function that run TNOE simulation"""
    # Hamiltonian model parameters
    # define number of vibrational model
    name = "h2o"

    model = read_in_model(inputdir, name, order=1)

    print("number of surfaces:{:}".format(model[VMK.A]))
    print("number of modes:{:}".format(model[VMK.N]))
    print("verticle energy (in eV):\n{:}".format(model[VMK.E]))
    print("Frequencies (in eV):\n{:}".format(model[VMK.w]))
    print("Linear coupling constants (in eV):\n{:}".format(model[VMK.G1]))


    # initialize the Hamiltonian
    model_hamiltonian = vibronic_model_hamiltonian(model, truncation_order = 1, FC=False)
    model_hamiltonian.TFCC_integration(T_initial=1e4, T_final=3e3, N_step=10000, output_path=outputdir)
    # model.sum_over_states(basis_size=40, output_path=outputdir, T_initial=10000, T_final=10, N_step=10000, compare_with_TNOE=True)
    # model._map_initial_amplitude(T_initial=1e4)

    # model.sum_over_states(basis_size=40, output_path=outputdir, compare_with_TFCC=False, T_grid=np.linspace(100, 10000, 10000))
    # model._map_initial_T_amplitude_from_FCI(T_initial=1000, basis_size=10)
    # model._map_initial_T_amplitude(T_initial=10000)
    # model.plot_thermal()
    #model._map_initial_T_amplitude(T_initial=1000.)

    return


if (__name__ == '__main__'):
    main()
