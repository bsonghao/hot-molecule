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
import project
from project.vibronic_model_Hamiltonian import vibronic_model_hamiltonian
from project.vibronic import vIO, VMK

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
    # Read in Hamiltonian model parameters
    # define number of vibrational model
    name = "CoF4"

    model = read_in_model(inputdir, name, order=2)

    print("number of surfaces:{:}".format(model[VMK.A]))
    print("number of modes:{:}".format(model[VMK.N]))
    print("verticle energy (in eV):\n{:}".format(model[VMK.E]))
    print("Frequencies (in eV):\n{:}".format(model[VMK.w]))
    print("Linear coupling constants (in eV):\n{:}".format(model[VMK.G1]))
    print("Quadratic coupling constants (in eV):\n{:}".format(model[VMK.G2]))

    # assert np.allclose(model[VMK.G2], np.transpose(model[VMK.G2], (1, 0, 3, 2)))

    # initialize the Hamiltonian
    model = vibronic_model_hamiltonian(model, name, truncation_order=2, FC=False)
    # Bogoliubov transform the Hamiltonian
    model.thermal_field_transform(T_ref=2e3)
    model.reduce_H_tilde()
    # run TFCC simulation
    model.TFCC_integration(T_initial=1e3, T_final=1e1, N_step=10000, output_path=outputdir)

    return


if (__name__ == '__main__'):
    main()
