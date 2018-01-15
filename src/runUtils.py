import os
import copy
from subprocess import call
from numpy import array
from crystal import *
import h5py

CPU_DEBUG_ARGS = ["gdb","--args"]
GPU_DEBUG_ARGS = ["cuda-gdb","--args"]
CPU_MEMCHECK_ARGS = ["valgrind"]
GPU_MEMCHECK_ARGS = ["cuda-memcheck"]
CPU_PROFILE_ARGS  = ["perf","record","-F997","--call-graph","dwarf"]
GPU_PROFILE_ARGS = ["nvprof"]

def getCfgDir():
    src_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(src_dir)
    cfg_dir = os.path.join(root_dir, "config")
    return cfg_dir

def getBinDir(build_type="release"):
    cfg_dir = getCfgDir()
    cfg_filename = os.path.join(cfg_dir, "%s.txt" % (build_type))
    read_dict = {}
    exec(open(cfg_filename,'r').read(), read_dict)
    return read_dict['bin_dir']

def getSpectralDatabaseFilename(db_dir, db_dim, use_unified_coeff_order):
    if use_unified_coeff_order:
        db_basename = os.path.join(db_dir, "databaseSpectralOrderedUnified")
    else:
        db_basename = os.path.join(db_dir, "databaseSpectralOrdered")
    db_filename = db_basename+"%d.hdf5" % (db_dim)
    return db_filename

class GenericRun(object):
    def __init__(self, params={}, verbose=False):
        self.params = {}
        self.params['name'] = ''
        self.params.update(params)
        self.verbose = verbose

    def run(self):
        print("Run execution not defined for this subclass of GenericRun.")

    def __getitem__(self, key):
        return self.params[key]
        
    def __setitem__(self, key, value):
        self.params[key] = value

    def __repr__(self):
        """The unambiguous, potentially ugly print.
        
        This is called in an interactive prompt, or if a list containing
        instances of this class is printed. The goal is to have a string
        that would be sufficient for instantiating an object of this
        class in the same way that this instance was.
        """
        return "GenericRun(params=%s,verbose=%s)"%tuple([arg.__repr__() for arg in
        [self.params, self.verbose]])
        
    def __str__(self):
        """The pretty, potentially ambiguous print.
        
        This is called when this class is printed. The goal is for it
        to be human readable.
        """
        return self.__repr__()

def getStrainStressFromResultsFile(filename, stress_unit='MPa'):
    # Extract
    results = h5py.File(filename, "r")
    true_strain_history = results['trueStrainHistory']
    T_cauchy_history = []
    for i in range(results['TCauchyHistory'].shape[0]):
        T_cauchy_history.append(results['TCauchyHistory'][i,:,:])
    
    # Scale
    if stress_unit == 'MPa':
        T_cauchy_history = [array(a)*1e3 for a in T_cauchy_history]
    elif stress_unit == 'GPa':
        T_cauchy_history = [array(a) for a in T_cauchy_history]
    else:
        print("Haven't implemented stress unit \""+stress_unit+"\".")
    
    # Return
    return true_strain_history, T_cauchy_history
    
class IterativeSolveRun(GenericRun):
    """This class handles running and handling output from the iterative solver.
    """

    def __init__(self, params={}, verbose=False):
        # Base construction
        super(IterativeSolveRun, self).__init__(params, verbose)
    
    def updateDerivedParameters(self):
        if self['do_debug'] or self['do_memcheck']:
            bin_dir = getBinDir("debug")
        else:
            bin_dir = getBinDir("release")
        self.executable = os.path.join(bin_dir,"iterativeSolve")
        self['results_filename'] = self['name']+"_result_iterative.hdf5"
    
    def run(self):
        # Derived parameters
        self.updateDerivedParameters()
              
        # Call arguments
        args = []
        
        # Arguments for debugging, memory checking and profiling
        if self['do_debug']:
            args += CPU_DEBUG_ARGS
        if self['do_memcheck']:
            args += CPU_MEMCHECK_ARGS
        if self['do_profile']:
            args += CPU_PROFILE_ARGS
        
        # MPI
        if not self['do_debug']:
            args += ["mpirun","-np",str(self['np'])]            
        
        # Main call arguments
        args += [self.executable]
        args += ["-n", str(self['n_crystals'])]
        args += ["-o", self['results_filename']]
        args += ["-e", self['experiment_name']]
        
        # Default seed or not
        if self['default_seed']:
            args += ["-d"]
        
        # Call program
        if self.verbose:
            print(" ".join(args))
        call(args)
        
        # Detailed profiling stats
        if self['do_profile']:
            perfData = getPerfPercentages()
            print(perfData)
            
    def getStrainAndStress(self):
        return getStrainStressFromResultsFile(self['results_filename'])
        
    def getPoleHistograms(self):
        histograms = {}
        results = h5py.File(self['results_filename'], "r")
        for key in results.keys():
            if key.startswith('poleHistogram_'):
                pole_name = key[key.find('_')+1:]
                histograms[pole_name] = results[key]
        return histograms

class SpectralSolveRun(GenericRun):
    """This class handles running and handling output from the spectral solver.
    """

    def __init__(self, params={}, verbose=False):
        # Base construction
        super(SpectralSolveRun, self).__init__(params, verbose)
    
    def updateDerivedParameters(self):
        # Set executable
        if self['do_debug'] or self['do_memcheck']:
            bin_dir = getBinDir("debug")
        else:
            bin_dir = getBinDir("release")
        if self['use_gpu']:
            self.executable = os.path.join(bin_dir,"spectralSolveCUDA")
        else:
            self.executable = os.path.join(bin_dir,"spectralSolve")
        
        # Filename    
        self['results_filename'] = self['name']+"_result_spectral.hdf5"
        self['db_filename'] = getSpectralDatabaseFilename(self['db_dir'], self['db_dim'], self['use_unified_coeff_order'])  
    
    def run(self):
        # Derived parameters
        self.updateDerivedParameters()
        
        # Call arguments
        args = []
        
        # Arguments for debugging
        if self['do_debug']:
            if self['use_gpu']:
                args += GPU_DEBUG_ARGS
            else:
                args += CPU_DEBUG_ARGS
                
        # Arguments for memory checking
        if self['do_memcheck']:
            if self['use_gpu']:
                args += GPU_MEMCHECK_ARGS
            else:
                args += CPU_MEMCHECK_ARGS
        
        # Arguments for profiling
        if self['do_profile']:
            if self['use_gpu']:
                args += GPU_PROFILE_ARGS
            else:
                args += CPU_PROFILE_ARGS
        
        # Main call arguments
        args += [self.executable]
        args += ["-n", str(self['n_crystals'])]
        args += ["-i", self['db_filename']]
        args += ["-t", str(self['n_terms'])]
        args += ["-o", self['results_filename']]
        args += ["-r", str(self['refinement_multiplier'])]
        args += ["-e", self['experiment_name']]
        
        # Default seed or not
        if self['default_seed']:
            args += ["-d"]
        
        # Unified coefficient ordering or not
        if self['use_unified_coeff_order']:
            args += ["-u"]
        
        # OpenMP for CPU implementation
        if not self['use_gpu']:
            args += ["-m",str(self['n_omp_threads'])]
        
        # Call program
        if self.verbose:
            print(" ".join(args))
        call(args)
        
        # Detailed profiling stats
        if self['do_profile']:
            perfData = getPerfPercentages()
            print(perfData)
        
    def getStrainAndStress(self):
        return getStrainStressFromResultsFile(self['results_filename'])
        
    def getGigatermsComputationRate(self):
        # Fetch outputs    
        results = h5py.File(self['results_filename'], "r")
        elapsed = results.attrs['spectralPolycrystalSolveTime']
        
        if elapsed > 0:        
            # Overall profiling stats
            nFourierTermsComputed = results.attrs['nTimestepsTaken']*self['n_crystals']*self['n_terms']*results.attrs['nComponents']
            print("elapsed = ", elapsed)
            print("gigaterms computed =", nFourierTermsComputed/1.0e9)
            print("gigaterms/second =", (nFourierTermsComputed/elapsed)/1.0e9)
            
            # Implementation makes use of symmetries in real data, so the actual
            # number of terms computed at the hardware level is a little over half
            #print "gigaterms computed (hardware)=", nFourierTermsComputedHardware/1e9
            #print "gigaterms/second (hardware) =", (nFourierTermsComputedHardware/elapsed)/1.0e9
            print("terms/terms(hardware) =", float(nFourierTermsComputed)/results.attrs['nFourierTermsComputedHardware'])
            
            print("strain steps/second = ", results.attrs['nTimestepsTaken']/elapsed)
            
            return (nFourierTermsComputed/elapsed)/1.0e9
        else:
            return 0
    
    def getMaxMemUsedGB(self):
        results = h5py.File(self['results_filename'], "r")
        return results.attrs['maxMemUsedGB']
        
    def getPoleHistograms(self):
        histograms = {}
        results = h5py.File(self['results_filename'], "r")
        for key in results.keys():
            if key.startswith('poleHistogram_'):
                pole_name = key[key.find('_')+1:]
                histograms[pole_name] = results[key]
        return histograms

def expandRunsByListParameter(runs, paramName, paramValList):
    expanded_runs = []
    for run in runs:
        for val in paramValList:
            new_run = copy.deepcopy(run)
            new_run.params[paramName] = val
            # New run name
            name_format = "%s="
            if type(val) is str:
                name_format += "%s"
            elif 'int' in str(type(val)):
                name_format += "%d"
            elif 'float' in str(type(val)):
                name_format += "%g"
            else:
                print("type is", type(val))
                print("Treating", val, "as a string.")
                name_format += "%s"
                val = str(val)
            
            if new_run.params['name'] != '':
                new_run.params['name'] += "_"
            new_run.params['name'] += name_format % (paramName, val)
            
            # Add to expanded
            expanded_runs.append(new_run)
            
    # Return
    return expanded_runs

def expandRunsByAllListParameters(runs,exclude=[]):
    all_list_parameters = []
    allExpandedRuns = []
    for run in runs:
        # Get which parameters are lists
        expandParams = {}
        for paramName in run.params.keys():
            if paramName not in exclude:
                paramVal = run.params[paramName]
                if type(paramVal) is list or type(paramVal) is tuple:
                    expandParams[paramName] = paramVal
        
        # Expand this run through those lists
        expandedRuns = [run]
        for paramName, paramValList in expandParams.items():
            expandedRuns = expandRunsByListParameter(expandedRuns, paramName, paramValList)
            
        # Add the expanded runs to the new list
        for run in expandedRuns:
            allExpandedRuns.append(run)
    return allExpandedRuns
