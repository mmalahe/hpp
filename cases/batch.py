"""A script for batch running and analysing the output of the solvers.
"""

import sys
import multiprocessing
sys.path.append("../src")
from profUtils import *
from runUtils import *
from plotting import *
from analysisUtils import *

setPlotDefaults('journal')

# Top level control
do_iterative_solve = False
do_spectral_solve = True

do_iterative_solve_plot = False
do_spectral_solve_plot = False

#~ error_study_type = None
error_study_type = 'n_terms'
#~ error_study_type = 'refinement_multiplier'
#~ error_study_type = 'db_dim'

spectral_performance_study_type = None
#~ spectral_performance_study_type = 'n_terms'
#~ spectral_performance_study_type = 'n_crystals'
#~ spectral_performance_study_type = 'refinement_multiplier'

spectral_mem_study_type = None
#~ spectral_mem_study_type = 'n_crystals'

# General parameters
general_params = {}
general_params['do_debug'] = False
general_params['do_memcheck'] = False
general_params['do_profile'] = False

# Problem parameters
problem_params = {}
problem_params['default_seed'] = True
#~ problem_params['n_crystals'] = 2*22*768*2**5
problem_params['n_crystals'] = 22*2*768
#~ problem_params['n_crystals'] = 2*768
#~ problem_params['n_crystals'] = [33*2**i for i in range(7,18,1)]
problem_params['experiment_name'] = []
problem_params['experiment_name'].append('mihaila2014_simple_shear')
#~ problem_params['experiment_name'].append('savage2015_plane_strain_compression')
#~ problem_params['experiment_name'].append('kalidindi1992_simple_shear')
#~ problem_params['experiment_name'].append('kalidindi1992_simple_compression')
#~ problem_params['experiment_name'].append('static')

# Spectral database parameters
spectral_db_params = {}
spectral_db_params['db_dir'] = "databases"
spectral_db_params['db_dim'] = 128
#~ spectral_db_params['db_dim'] = [8,16,32,64,128]
spectral_db_params['refinement_multiplier'] = 128
#~ spectral_db_params['refinement_multiplier'] = [1,2,4,8,16,32,64,128]
#~ spectral_db_params['refinement_multiplier'] = [1,4,16,64,128]
spectral_db_params['use_unified_coeff_order'] = True  

# Iterative solve parameters
iterative_solve_verbose = True
iterative_solve_params = {}
iterative_solve_params.update(general_params)
iterative_solve_params.update(problem_params)
iterative_solve_params['np'] = multiprocessing.cpu_count()

# Spectral solver parameters
spectral_solve_verbose = True
spectral_solve_params = {}
spectral_solve_params.update(general_params)
spectral_solve_params.update(problem_params)
spectral_solve_params.update(spectral_db_params)
spectral_solve_params['n_omp_threads'] = multiprocessing.cpu_count()
spectral_solve_params['use_gpu'] = True
#~ spectral_solve_params['n_terms'] = 2**14
spectral_solve_params['n_terms'] = [2**i for i in range(1,17)]
#~ spectral_solve_params['n_terms'] = [2**i for i in range(7,16)]

# Iterative solves
iterative_solve_base_runs = [IterativeSolveRun(iterative_solve_params, iterative_solve_verbose)]
iterative_solve_runs = expandRunsByAllListParameters(iterative_solve_base_runs)
for run in iterative_solve_runs:
    if do_iterative_solve:  
        run.run()
    else:
        run.updateDerivedParameters()

# Spectral solves
spectral_solve_base_runs = [SpectralSolveRun(spectral_solve_params, spectral_solve_verbose)]
spectral_solve_runs = expandRunsByAllListParameters(spectral_solve_base_runs)
for run in spectral_solve_runs:
    if do_spectral_solve:    
        run.run()
        run.getGigatermsComputationRate();
    else:
        run.updateDerivedParameters()

# Solution plots
if do_iterative_solve_plot or do_spectral_solve_plot:
    doIterativeSpectralPlots(do_iterative_solve_plot, iterative_solve_runs, do_spectral_solve_plot, spectral_solve_runs)
    
# Error studies
if error_study_type != None:    
    # Comparison with own solution
    doErrorStudy(spectral_solve_runs[-1], spectral_solve_runs[:-1], error_study_type)
    
    # Comparison with iterative solver
    #~ assert(len(iterative_solve_runs) == 1)
    #~ doErrorStudy(iterative_solve_runs[0], spectral_solve_runs, error_study_type)
    
# Performance studies
if spectral_performance_study_type != None:
    doSpectralPerformanceStudy(spectral_solve_runs, spectral_performance_study_type)

# Memory studies
if spectral_mem_study_type != None:
    doMemUsageStudy(spectral_solve_runs, spectral_mem_study_type)
