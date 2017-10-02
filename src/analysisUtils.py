from numpy.linalg import norm
from linearAlgebra import *
from crystal import *
from plotting import *
from runUtils import *
from numpy import polyfit, log, exp
import numpy as np

def gaussianFilterHistogramHistory(hists, sigma=12.0):
    filtered = zeros(hists.shape)
    for i in range(hists.shape[0]):
        filtered[i,:,:] = gaussian_filter(hists[i,:,:], sigma=sigma)
    return filtered

def doIterativeSpectralPlots(do_iterative_solve_plot, iterative_solve_runs, do_spectral_solve_plot, spectral_solve_runs):
    # Number of plots
    assert(do_iterative_solve_plot or do_spectral_solve_plot)
    if do_iterative_solve_plot:
        n_iterative_plots = len(iterative_solve_runs)
        n_plots = n_iterative_plots
    if do_spectral_solve_plot:
        n_spectral_plots = len(spectral_solve_runs)
        n_plots = n_spectral_plots
    if do_iterative_solve_plot and do_spectral_solve_plot:
        assert(n_iterative_plots == n_spectral_plots)    
    
    # Loop through plots
    for i_plot in range(n_plots):    
        # Make sure runs are matching up
        iterative_run = iterative_solve_runs[i_plot]
        spectral_run = spectral_solve_runs[i_plot]
        assert(iterative_run['name'] == spectral_run['name'])
        run_name = iterative_run['name']
        experiment_name = iterative_run['experiment_name']
        
        # Figure setup
        figure()
        leg = []    
        figname = run_name+"_ss"
        
        # Labels
        plane_normals_A = [[1.0,1.0,1.0],[1.0,1.0,0.0],[1.0,0.0,0.0]]
        pole_names_A = ['111','110','100']
        plane_normals_B = [[0.0,0.0,1.0],[0.0,1.0,1.0],[1.0,1.0,1.0]]
        pole_names_B = ['001','011','111']
        if experiment_name == 'kalidindi1992_simple_shear' or experiment_name == 'mihaila2014_simple_shear':    
            x_label = "$|\epsilon|$"
            y_label = "Stress (MPa)"
            plane_normals = plane_normals_A  
            pole_names = pole_names_A  
        elif experiment_name == 'kalidindi1992_simple_compression':
            x_label = "$|\epsilon|$"
            y_label = "$\sigma$ (MPa)"        
            plane_normals = plane_normals_B
            pole_names = pole_names_B  
        elif experiment_name == 'savage2015_plane_strain_compression':
            x_label = "$|\epsilon|$"
            y_label = "Stress (MPa)"        
            plane_normals = plane_normals_B
            pole_names = pole_names_B  
        else:
            plane_normals = plane_normals_A  
            pole_names = pole_names_A  
            x_label = "Strain"
            y_label = "Stress (MPa)"
        
        # Stress strain plots
        stress_max = 0.0
        if do_spectral_solve_plot:    
            # Load data
            true_strain_history, T_cauchy_history = spectral_run.getStrainAndStress()
            T_11 = array([T_cauchy[0,0] for T_cauchy in T_cauchy_history])
            T_12 = array([T_cauchy[0,1] for T_cauchy in T_cauchy_history])
            T_22 = array([T_cauchy[1,1] for T_cauchy in T_cauchy_history])
            T_33 = array([T_cauchy[2,2] for T_cauchy in T_cauchy_history])
            
            # Plotting
            if 'simple_shear' in experiment_name:
                spectral_stress = abs(T_12)
                leg.append("$\sigma_{12}$ (spectral)")
            elif experiment_name == 'kalidindi1992_simple_compression':
                sigma = T_33 - 0.5*(T_11+T_22)
                spectral_stress = abs(sigma)
                leg.append("spectral")
            elif 'plane_strain_compression' in experiment_name:
                spectral_stress = abs(T_33)
                leg.append("$\sigma_{33}$ (spectral)")
            
            stress_max = max(stress_max, np.max(spectral_stress))    
            plot(np.abs(true_strain_history), spectral_stress, 'k+')            
            figname += "_spectral"
        
        if do_iterative_solve_plot:
            # Load data
            true_strain_history, T_cauchy_history = iterative_run.getStrainAndStress()
            T_11 = array([T_cauchy[0,0] for T_cauchy in T_cauchy_history])
            T_12 = array([T_cauchy[0,1] for T_cauchy in T_cauchy_history])
            T_22 = array([T_cauchy[1,1] for T_cauchy in T_cauchy_history])
            T_33 = array([T_cauchy[2,2] for T_cauchy in T_cauchy_history])
            
            # Plotting
            if 'simple_shear' in experiment_name:
                iterative_stress = abs(T_12)
                leg.append("$\sigma_{12}$ (iterative)")
            elif experiment_name == 'kalidindi1992_simple_compression': 
                sigma = T_33 - 0.5*(T_11+T_22)
                iterative_stress = abs(sigma)
                leg.append("iterative")
            elif 'plane_strain_compression' in experiment_name:
                iterative_stress = abs(T_33)
                leg.append("$\sigma_{33}$ (iterative)")
            
            stress_max = max(stress_max, np.max(iterative_stress)) 
            plot(np.abs(true_strain_history), iterative_stress, 'k-')
            figname += "_iterative"
        
        # Common plotting
        ylim((0.0, stress_max*1.05)) 
        legend(leg, loc='best')
        xlabel(x_label)
        ylabel(y_label)        
    
        # Save
        figname += ".png"
        savefig(figname, bbox_inches='tight')
    
        # Which pole figures to plot
        do_spectral_pole_plots = do_spectral_solve_plot and spectral_run['do_plot_pole_figures']
        do_iterative_pole_plots = do_iterative_solve_plot and iterative_run['do_plot_pole_figures']
        do_pole_plots = do_spectral_pole_plots or do_iterative_pole_plots
        pole_figure_plot_timestep_list = spectral_run['pole_figure_timestep_selection']
        
        # Fetch pole data and set smoothing parameters
        if do_spectral_pole_plots:
            pole_histograms_spectral = spectral_run.getPoleHistograms()
            pole_data_spectral = {name:array(pole_histograms_spectral[name], dtype=numpy.float64) for name in pole_names}
            n_pixels_side = pole_histograms_spectral.values()[0].shape[1] 
            smoothing_per_pixel = spectral_run['histogram_smoothing_per_pixel']
        if do_iterative_pole_plots:
            pole_histograms_iterative = iterative_run.getPoleHistograms()
            pole_data_iterative = {name:array(pole_histograms_iterative[name], dtype=numpy.float64) for name in pole_names}
            n_pixels_side = pole_histograms_iterative.values()[0].shape[1]
            smoothing_per_pixel = iterative_run['histogram_smoothing_per_pixel']
        if do_pole_plots:
            smoothing_sigma = n_pixels_side*smoothing_per_pixel
        
        # Smooth data
        if do_spectral_pole_plots:   
            pole_data_spectral = {name: gaussianFilterHistogramHistory(pole_data_spectral[name], sigma=smoothing_sigma) for name in pole_names} 
        if do_iterative_pole_plots:
            pole_data_iterative = {name: gaussianFilterHistogramHistory(pole_data_iterative[name], sigma=smoothing_sigma) for name in pole_names} 
            
        # Plot pole figures        
        if do_spectral_pole_plots and do_iterative_pole_plots:
            pole_data_diffs = {name: pole_data_iterative[name]-pole_data_spectral[name] for name in pole_names}
            plotPoleHistogramsHistory(pole_data_diffs, pole_figure_plot_timestep_list, experiment_name+"_poles_spectral_iterative_diff")
        if do_spectral_pole_plots:
            plotPoleHistogramsHistory(pole_data_spectral, pole_figure_plot_timestep_list, experiment_name+"_poles_spectral")
        if do_iterative_pole_plots:
            plotPoleHistogramsHistory(pole_data_iterative, pole_figure_plot_timestep_list, experiment_name+"_poles_iterative")

def getLog2Ticks(x_variable_list):
    log2Ticks = [int(log(x)/log(2)) for x in x_variable_list]
    ticks = [2**a for a in log2Ticks]
    tickNames = ["$2^{%d}$"%(a) for a in log2Ticks]
    return ticks, tickNames

def doErrorStudy(reference_run, runs, x_variable_name, x_variable_plot_name=None):
    # Names for plotting
    if x_variable_plot_name == None:
        x_variable_plot_name = ""
        if x_variable_name == 'n_terms':
            x_variable_plot_name = "Number of terms in Fourier series"
        elif x_variable_name == 'refinement_multiplier':
            x_variable_plot_name = "Refinement multiplier"
        elif x_variable_name == 'db_dim':
            x_variable_plot_name = "Spectral database dimension"
    
    # Number of runs
    n_runs = len(runs)
    assert(n_runs > 1)
    
    # Number of terms range
    x_variable_list = [run[x_variable_name] for run in runs]
    x_variable_list = array(x_variable_list)
    
    # L2 errors
    l2_error_list = []
    reference_strains, reference_t_cauchys = reference_run.getStrainAndStress()
    l2_reference = L2NormFunctionValues(reference_strains, [norm(t_cauchy) for t_cauchy in reference_t_cauchys])  
    for i_run in range(n_runs):    
        strains, t_cauchys = runs[i_run].getStrainAndStress()
        if len(strains) != len(reference_strains):
            t_cauchys = resampleFunction(strains, t_cauchys, reference_strains)
        error_t_cauchys = [norm(reference_t_cauchys[i]-t_cauchys[i]) for i in range(len(t_cauchys))]        
        l2_error = L2NormFunctionValues(reference_strains, error_t_cauchys)/l2_reference
        l2_error_list.append(l2_error)    
    l2_error_list = array(l2_error_list)
    
    # Fit
    log_x = log(x_variable_list)
    log_l2 = log(l2_error_list)
    fit_parms = polyfit(log_x, log_l2, 1)
    slope = fit_parms[0]
    intercept = fit_parms[1]
    fit_log_l2 = slope*log_x + intercept
    fit_l2 = exp(fit_log_l2)    
        
    # Plot
    figure()
    leg = []
    loglog(x_variable_list, l2_error_list, 'k+')
    leg.append("Data")
    loglog(x_variable_list, fit_l2, 'k--')
    leg.append("Fit with slope %1.2f" % (slope))
    xlim(0.95*min(x_variable_list),1.05*max(x_variable_list))
    xlabel(x_variable_plot_name)
    ylabel("L2 error")
    legend(leg, loc='best')
    ticks, tickNames = getLog2Ticks(x_variable_list)
    xticks(ticks, tickNames)
    figname = reference_run['name']+"_error_vs_"+x_variable_name+".png"
    savefig(figname)
    
def doSpectralPerformanceStudy(runs, x_variable_name, x_variable_plot_name=None):
    # Names for plotting
    if x_variable_plot_name == None:
        x_variable_plot_name = ""
        if x_variable_name == 'n_terms':
            x_variable_plot_name = "Number of terms in Fourier series"
        elif x_variable_name == 'n_crystals':
            x_variable_plot_name = "Number of crystals"
        elif x_variable_name == 'refinement_multiplier':
            x_variable_plot_name = "Refinement multiplier"
            
    # Get values
    x_variable_list = [run[x_variable_name] for run in runs]
    gterms_rate_list = [run.getGigatermsComputationRate() for run in runs]
    
    # Plot
    figure()
    semilogx(x_variable_list, gterms_rate_list, 'k+')
    xlabel(x_variable_plot_name)
    ylabel("Throughput (Gigaterms/s)")
    ticks, tickNames = getLog2Ticks(x_variable_list)
    xticks(ticks, tickNames)
    figname = runs[-1]['name']+"_rate_vs_"+x_variable_name+".png"
    savefig(figname)
    
def doMemUsageStudy(runs, x_variable_name, x_variable_plot_name=None):
    # Names for plotting
    if x_variable_plot_name == None:
        x_variable_plot_name = ""
        if x_variable_name == 'n_crystals':
            x_variable_plot_name = "Number of crystals"
            y_variable_plot_name = "Bytes per crystal"
            
    # Get values
    x_variable_list = [run[x_variable_name] for run in runs]
    bytes_per_x_list = [run.getMaxMemUsedGB()*(1024*1024*1024)/float(run[x_variable_name]) for run in runs]
    
    # Plot
    figure()
    semilogx(x_variable_list, bytes_per_x_list, 'ko')
    xlabel(x_variable_plot_name)
    ylabel(y_variable_plot_name)
    log2Ticks = [int(log(x)/log(2)) for x in x_variable_list]
    ticks = [2**a for a in log2Ticks]
    tickNames = ["$2^{%d}$"%(a) for a in log2Ticks]
    xticks(ticks, tickNames)
    figname = runs[-1]['name']+"_mem_vs_"+x_variable_name+".png"
    savefig(figname)
