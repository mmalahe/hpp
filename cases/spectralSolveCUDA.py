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
    
    # Experiment parameters
    experiment = hpp.ExperimentF(experiment_name)
    
    # Method constants
    strain_increment = 2e-2
    
    # Derived method constants
    dt = abs(strain_increment/experiment.strainRate)
    
    # Crystal properties and initial conditions
    props = hpp.defaultCrystalPropertiesF()
    init = hpp.defaultCrystalInitialConditionsF()
    
    # Make the list of crystals
    crystal_list = []
    for i in range(ncrystals):
        crystal = hpp.SpectralCrystalCUDAF()
        crystal.s = init.s_0
        crystal_list.append(crystal)
    
    print(hpp.listDemo(crystal_list))

# Inputs
experiment_name = 'mihaila2014_simple_shear'
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


