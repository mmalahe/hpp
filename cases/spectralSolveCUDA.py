import sys
import os
sys.path.append(os.path.join("..","build-release","lib","hpp"))
import hpppy as hpp

def spectralSolve(experiment_name,
                  database_filename,
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
    props = hpp.CrystalPropertiesCUDAF12(hpp.defaultCrystalPropertiesF())
    init = hpp.defaultCrystalInitialConditionsF()
    
    # Make the list of crystals with their orientations
    crystal_list = []
    for i in range(ncrystals):
        crystal = hpp.SpectralCrystalCUDAF()                            # default crystal
        crystal.s = init.s_0                                            # initial deformation resistance
        crystal.angles = experiment.generateNextOrientationAngles()     # initial orientation
        crystal_list.append(crystal)
        
    # Choose the dataset IDs for the crystal response database
    dsetIDs = hpp.defaultCrystalSpectralDatasetIDs()
    
    # Load the crystal response database
    db = hpp.SpectralDatabaseUnifiedF(database_filename, dsetIDs, nterms, ref_multiplier)
    

# Inputs
experiment_name = 'mihaila2014_simple_shear'
database_filename = 'databases/databaseSpectralOrderedUnified128.hdf5'
ref_multiplier = 128
ncrystals = 1024
nterms = 2**12
output_filename = "output.hdf5"
default_seed=True

# Run
spectralSolve(experiment_name,
                  database_filename,
                  ref_multiplier,
                  ncrystals,
                  nterms,
                  output_filename,
                  default_seed)


