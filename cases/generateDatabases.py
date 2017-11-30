import os
import sys
sys.path.append("../src")
from subprocess import call
from plotting import *
from numpy import *

setPlotDefaults('journal')

# Top-level control
do_generate_raw_database = True
do_convert_raw_database_to_spectral = True

do_plot_spectral_data = False

do_calculate_spectral_database_error = False
do_plot_spectral_database_error = False

# Run control
do_debug = False
do_memcheck = False
do_profile = False
debug_args = ["gdb","--args"]

# MPI
np = 8

# Database properties
spectral_dim = 64
refinement_multiplier = 1
raw_dim = spectral_dim*refinement_multiplier
use_unified_coeff_order = True

# Number of coefficients to store/use in spectral database generation and visualisation
n_coeffs = 65536
plot_coeffs_max = 65536

# Parameters for spectral error visualisation
spectral_error_out_filename = "spectralError%d.txt"%(spectral_dim)
spectral_error_axis_slice = [0, raw_dim/8, raw_dim/4, raw_dim/2]

# Executable paths
if do_debug or do_memcheck:
    build_dir = "../build-debug"
else:
    build_dir = "../build-release"
bin_dir = os.path.join(build_dir,"bin")
rawDatabaseExecutable = os.path.join(bin_dir,"generateRawDatabase")
spectralDatabaseExecutable = os.path.join(bin_dir,"generateSpectralDatabase")
spectralErrorExecutable = os.path.join(bin_dir,"evaluateCompressionError")

# File paths
db_dir = "databases"
if not os.path.exists(db_dir):
    os.makedirs(db_dir)
raw_database_filename = os.path.join(db_dir, "databaseRaw%d.hdf5" % (raw_dim))
if use_unified_coeff_order:
    spectral_database_ordered_basename = os.path.join(db_dir, "databaseSpectralOrderedUnified")
else:
    spectral_database_ordered_basename = os.path.join(db_dir, "databaseSpectralOrdered")
spectral_database_ordered_filename = spectral_database_ordered_basename+"%d.hdf5" % (spectral_dim)
spectral_error_database_filename = os.path.join(db_dir, "databaseSpectralError%d.hdf5" % (spectral_dim))

if do_generate_raw_database:
    args = []
    if do_profile:
        args += ["perf","record","-F997","--call-graph","dwarf"]
    if do_debug:
        args += debug_args
    else:
        args += ["mpirun","-np",str(np)]
    args += [rawDatabaseExecutable,"-o",raw_database_filename,"-n",str(raw_dim)]
    call(args)
    
if do_convert_raw_database_to_spectral:
    if do_debug:
        args = ["gdb","--args"]
    else:
        args = ["mpirun","-np",str(np)]
    args += [spectralDatabaseExecutable,"-i",raw_database_filename,"-o",spectral_database_ordered_filename,"-n",str(n_coeffs)]
    if use_unified_coeff_order:
        args += ["-u"]
    call(args)
    
if do_plot_spectral_data:
    # Data file
    h5file = h5py.File(spectral_database_ordered_filename, "r")
    
    # Gamma dot abs sum    
    clf()
    gammadot_abs_sum_fft_ordered = h5file['gammadot_abs_sum_vals'][()][1:]
    gammadot_abs_sum_fft_magnitudes = abs(gammadot_abs_sum_fft_ordered)/abs(gammadot_abs_sum_fft_ordered).max()
    loglog(abs(gammadot_abs_sum_fft_magnitudes[:plot_coeffs_max]))
    savefig("gammadot_abs_sum_coeffs_cpp.png",bbox_inches='tight')
    
    # Sigma prime
    clf()
    leg = []
    fft_max_coeffs = []
    fft_coeffs_all = h5file["sigma_prime_vals"][()][:,:,1:]
    for i in range(3):
        for j in range(3):
            fft_coeffs = fft_coeffs_all[i,j,:]
            fft_max_coeffs.append(abs(fft_coeffs).max())
    fft_max_coeff = max(fft_max_coeffs)
    for i in range(3):
        for j in range(3):
            doPlot = True
            if (i==0 and j==0) or (i==1 and j==1):
                linespec = "b-"
            elif i==0 and j==1:
                linespec = "r--"
            elif (i==0 and j==2) or (i==1 and j==2):
                linespec = "g--"
            else:
                doPlot = False
            if doPlot:
                fft_coeffs = fft_coeffs_all[i,j,:]
                fft_magnitudes = abs(fft_coeffs)/fft_max_coeff
                loglog(fft_magnitudes[:plot_coeffs_max], linespec)
                leg.append("$\sigma'_{%d%d}$"%(i+1,j+1))
    legend(leg)
    savefig("sigma_prime_coeffs_cpp.png",bbox_inches='tight')
    
    # W_p
    clf()
    leg = []
    fft_max_coeffs = []
    fft_coeffs_all = h5file["W_p_vals"][()][:,:,1:]
    for i in range(3):
        for j in range(3):
            fft_coeffs = fft_coeffs_all[i,j,:]
            fft_max_coeffs.append(abs(fft_coeffs).max())
    fft_max_coeff = max(fft_max_coeffs)
    for i in range(3):
        for j in range(3):
            doPlot = True
            if i==0 and j==1:
                linespec = "r--"
            elif (i==0 and j==2) or (i==1 and j==2):
                linespec = "g--"
            else:
                doPlot = False
            if doPlot:
                fft_coeffs = fft_coeffs_all[i,j,:]
                fft_magnitudes = abs(fft_coeffs)/fft_max_coeff
                loglog(fft_magnitudes[:plot_coeffs_max], linespec)
                leg.append("$W^p_{%d%d}$"%(i+1,j+1))
    legend(leg)
    savefig("W_p_coeffs_cpp.png",bbox_inches='tight')

if do_calculate_spectral_database_error:
    args = []
    if do_debug:
        args += debug_args
    else:
        args += ["mpirun","-np",str(np)]
    args += [spectralErrorExecutable,"-d",raw_database_filename,"-s",spectral_database_ordered_filename]
    args += ["-e", spectral_error_database_filename]
    args += ["-t", str(n_coeffs)]
    args += ["-r", str(refinement_multiplier)]
    if len(spectral_error_axis_slice) > 0:
        args += ["-a "+str(val) for val in spectral_error_axis_slice]
    args += ["-o", spectral_error_out_filename]
    if use_unified_coeff_order:
        args += ["-u"]
    print args
    call(args)

if do_plot_spectral_database_error:
    exec(open(spectral_error_out_filename))
    clf()
    raw = W_p_12_raw
    spectral = W_p_12_spectral
    nPoints = len(raw)
    nPointsSpectral = len(spectral)
    plot(linspace(0,2*pi,nPoints+1)[:-1], raw)
    plot(linspace(0,2*pi,nPointsSpectral+1)[:-1], spectral)
    legend(["Raw","Spectral"])
    savefig("spectralError.png")
