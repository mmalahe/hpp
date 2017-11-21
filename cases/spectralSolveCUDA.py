import sys
import os
sys.path.append(os.path.join("..","build-release","lib","hpp"))
import hpppy as hpp

def spectralSolve(experiment_name,
                  database_filename,
                  unified_coeff_order,
                  ref_multiplier,
                  ncrystals,
                  nterms,
                  output_filename,
                  default_seed=False):
    
    # Method constants
    strain_increment = 2e-2

# Inputs
experiment_name = 'simple_shear'
database_filename = 'databaseSpectralOrderedUnified128.hdf5'
unified_coeff_order = True
ref_multiplier = 128
ncrystals = 1024
nterms = 2**12
output_filename = "output.hdf5"
default_seed=True

# Run
spectralSolve(experiment_name,
                  database_filename,
                  unified_coeff_order,
                  ref_multiplier,
                  ncrystals,
                  nterms,
                  output_filename,
                  default_seed)


