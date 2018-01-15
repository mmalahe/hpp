from plotting import *
from continuum import *
from math import *
from numpy import array, histogram2d, linspace, meshgrid, nan, ma
from numpy.linalg import norm
from scipy.ndimage.filters import gaussian_filter
from matplotlib.patches import Circle
import matplotlib.animation as manimation

def cartesianToSpherical(vec):
    """Uses "mathematics" convention.
    """
    
    # Magnitude
    r = norm(vec)
    unitVec = vec/r
    
    # Azimuthal
    theta = atan2(unitVec[1], unitVec[0])
    
    # Polar
    phi = acos(unitVec[2])    
    
    # Return
    return array([r,theta,phi])

def gaussFilterArrayWithNaNs(A, sigma):
    # Array with nans replaced by zero
    B = A.copy()
    B[A!=A] = 0
    BFiltered = gaussian_filter(B, sigma)
    
    # Array with data replaced by ones and nans replaced by zeros
    # After filtering, this represents how much has been lost by the
    # propagation of the zeros in place of nans in the previous
    # filtering
    C = ones(A.shape)
    C[A!=A] = 0
    CFiltered = gaussian_filter(C, sigma)
    
    # Make any remaining zeros safe for division
    CFiltered[CFiltered==0] = 1.0
    
    # Scale back in the data washed out by the nans replaced by zeros
    return BFiltered/CFiltered

def planeNormalToName(plane_normal):
    return "%d%d%d"%(plane_normal[0],plane_normal[1],plane_normal[2])

def planeNormalToPathFriendlyName(plane_normal):
    return "%d%d%d"%(plane_normal[0],plane_normal[1],plane_normal[2])

def projectionReferencePoints(projection_name, scale=None, centre=None):
    theta_range = linspace(0,2*pi,21)
    phi_range = linspace(0,pi/2,8)[1:]
    x_points = []
    y_points = []
    for theta in theta_range:
        for phi in phi_range:
            if projection_name == 'stereogrephic':
                R = tan(phi/2)
                maxR = tan(pi/4)
            elif projection_name == 'equal-area':
                R = 2*sin(phi/2)
                maxR = 2*sin(pi/4)
            if scale != None:
                R = (R/maxR)*scale
            x = R*cos(theta)
            y = R*sin(theta)
            if centre != None:
                x += centre[0]
                y += centre[1]
            x_points.append(x)
            y_points.append(y)
    return x_points, y_points

def getPoleHistograms(euler_angles, plane_normals=[array([1,1,1]),array([1,1,0]),array([1,0,0]),array([0,0,1]),array([0,1,1])], projection='equal-area', nBins=256):
    """Does a stereographic projection of the poles on the upper half sphere.
    """    
    pole_data = {}    
    for plane_normal in plane_normals:
        poles_azimuthal = []
        poles_polar = []
        poles_point_x = []
        poles_point_y = []
        for angle in euler_angles:            
            # Active rotation
            R = EulerZXZRotationMatrix(angle[0], angle[1], angle[2])
            pole = R.dot(plane_normal)
            
            # Spherical coordinates
            pole_spherical = cartesianToSpherical(pole)           
            theta = pole_spherical[1]
            phi = pole_spherical[2]
            
            # Projection from the northern hemisphere onto a plane through the equator
            if phi <= pi/2:
                if projection == 'stereographic':                    
                    R = tan(phi/2)
                    maxR = tan(pi/4)                         
                elif projection == 'equal-area':
                    R = 2*sin(phi/2)
                    maxR = 2*sin(pi/4)
                poles_point_x.append(R*cos(theta))
                poles_point_y.append(R*sin(theta))                
        
        # Histogram the points     
        xRange = [-maxR,maxR]
        yRange = [-maxR,maxR]
        hist, binsX, binsY = histogram2d(poles_point_x, poles_point_y, nBins, [xRange,yRange])
        
        # Store the data
        pole_data[planeNormalToName(plane_normal)] = hist
    
    # Return
    return pole_data

def plotPoleHistograms(pole_data, filename):    
    n_poles = len(pole_data)
    
    # Colorbar spacing
    cbar_main_frac = 0.13
    cbar_pad_frac = 0.07
    cbar_total_frac = cbar_main_frac+cbar_pad_frac
    
    # Set up subplots    
    fig, axes = subplots(nrows=1, ncols=n_poles, figsize=(8*n_poles, 8/(1-cbar_total_frac)))    
    subplots_adjust(wspace=0,hspace=0)
    i_subplot = 0
    
    for pole_name, pole_hist in pole_data.iteritems():
        # Set current axes
        ax = axes.flat[i_subplot]
        sca(ax)
        
        # Smooth data
        hist = gaussian_filter(pole_hist, sigma=12.0)
        
        # Histogram dimensions
        nBins = hist.shape[0]
        maxR = nBins/2.0
        histCentre = [nBins/2.0, nBins/2.0]
        xRange = [0.0,nBins]
        yRange = [0.0,nBins]
        
        # Sum
        hist_sum = sum(sum(hist))
        max_density = hist.max()
        
        # Mark invalid bins with nan
        for ix in range(nBins):
            for iy in range(nBins):
                x = ix - histCentre[0]
                y = iy - histCentre[1]
                if sqrt(x**2+y**2) > maxR:
                    hist[ix,iy] = nan        
        n_valid_bins = numpy.count_nonzero(hist!=nan)
        
        # Normalise to uniform density
        uniform_density = hist_sum/n_valid_bins
        hist /= uniform_density
        max_density /= uniform_density
        
        # Plot
        hist_masked = ma.masked_invalid(hist)              
        plot_ax = pcolormesh(hist_masked.T)
        plot_ax.set_clim(1.0, max_density)
        circ = Circle(histCentre, radius=maxR, fill=False)
        gca().add_patch(circ)
        removeBorder(gca())
        xlim(1.01*array(xRange))
        ylim(1.01*array(yRange))
        title(pole_name)
        if i_subplot == 0:
            xlabel("$\mathbf{e}_1$")
            ylabel("$\mathbf{e}_2$")
            
        # Projection reference points
        x_ref, y_ref = projectionReferencePoints(projection, scale=maxR, centre=histCentre)
        plot(x_ref, y_ref, '.', markersize=1, markerfacecolor=(0.5, 0.5, 0.5, 0.5), markeredgewidth=0.0)
        
        # Next subplot
        i_subplot += 1
    
    # Add colorbar
    cbar_ticks = linspace(1.0, max_density, 7)    
    cbar = fig.colorbar(plot_ax, ax=axes.ravel().tolist(), fraction=cbar_main_frac, 
    pad=cbar_pad_frac, ticks=cbar_ticks, orientation='horizontal', format='%1.1f', label='MRD')
    cbar.set_clim(1.0, max_density)
    
    # Save    
    savefig(filename, bbox_inches='tight')

def plotPoleHistogramsHistory(pole_history_data, timestep_selection, base_filename):    
    # This factor scales down the measured maximum density for the purposes of setting
    # the ranges of colour bars.
    # The intent is to allow lower density regions to be visible on the same plot.
    # One downside is that densities above this will all be represented by the same
    # colour, so there is an artificial grouping at the high end.
    # Set to 1.0, the plot is scaled normally.
    density_max_downscale = 1.0
    
    # Set up subplot dimensions
    n_poles = len(pole_history_data)
    poles_per_row = 3
    n_rows = int(ceil(float(n_poles)/poles_per_row))
    n_cols = ceil(n_poles/n_rows)
    
    # Colorbar spacing
    cbar_main_frac = 0.13/n_rows
    cbar_pad_frac = 0.07/n_rows
    cbar_total_frac = cbar_main_frac+cbar_pad_frac
    
    # Set up subplots
    fig_size = (8*n_cols, 8.0*(n_rows)/(1-cbar_total_frac))
    fig, axes = subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)    
    subplots_adjust(wspace=0)
    
    # Arrange poles
    pole_names_unordered = pole_history_data.keys()
    pole_names_unordered_strings_reversed = [name[::-1] for name in pole_names_unordered]
    pole_names_ordered_strings_reveresed = sorted(pole_names_unordered_strings_reversed)
    pole_names_ordered = [name[::-1] for name in pole_names_ordered_strings_reveresed]    
    
    # Get number of timesteps
    ntimesteps = list(pole_history_data.values())[0].shape[0]
    
    # Get ranges
    min_density = 1e10
    max_density = 0.0
    max_hist_name = None
    max_hist_it = None
    for pole_name in pole_names_ordered:
        for it in timestep_selection:
            hist = pole_history_data[pole_name][it,:,:].copy()
            local_min_density = hist.min()
            local_max_density = hist.max()         
            hist_sum = sum(sum(abs(hist)))
               
            # Mark invalid bins with nan
            nBins = hist.shape[0]
            maxR = nBins/2.0
            histCentre = [nBins/2.0, nBins/2.0]
            for ix in range(nBins):
                for iy in range(nBins):
                    x = ix - histCentre[0]
                    y = iy - histCentre[1]
                    if sqrt(x**2+y**2) > maxR:
                        hist[ix,iy] = nan        
            n_valid_bins = numpy.count_nonzero(hist!=nan)        
            
            # Normalise to uniform density
            uniform_density = hist_sum/n_valid_bins
            hist /= uniform_density
            local_max_density /= uniform_density
            local_min_density /= uniform_density
            
            # Keep track of the histogram with the highest density
            if local_max_density > max_density:
                max_density = local_max_density
                max_hist_name = pole_name
                max_hist_it = it
            if local_min_density < min_density:
                min_density = local_min_density
    max_density *= density_max_downscale
    min_density *= density_max_downscale
    
    # Initialisations
    pcolormeshes = {}
    i_subplot = 0
    for pole_name in pole_names_ordered:
        ax = axes.flat[i_subplot]
        sca(ax)
        hist = pole_history_data[pole_name][max_hist_it,:,:].copy()
        
        # Mark invalid bins with nan
        hist_sum = sum(sum(hist))
        nBins = hist.shape[0]
        maxR = nBins/2.0
        histCentre = [nBins/2.0, nBins/2.0]
        for ix in range(nBins):
            for iy in range(nBins):
                x = ix - histCentre[0]
                y = iy - histCentre[1]
                if sqrt(x**2+y**2) > maxR:
                    hist[ix,iy] = nan        
        n_valid_bins = numpy.count_nonzero(hist!=nan)  
        
        # Normalise to uniform density
        uniform_density = hist_sum/n_valid_bins
        hist = pole_history_data[pole_name][max_hist_it,:,:]/uniform_density
        hist *= density_max_downscale
        
        pcolormeshes[pole_name] = pcolormesh(hist.T)
        i_subplot += 1
    
    # Set lowest density to MRD=1.0
    min_density = 1.0
    
    cbar_ticks = linspace(min_density, max_density, 7)
    cbar = fig.colorbar(pcolormeshes[max_hist_name], ax=axes.ravel().tolist(), fraction=cbar_main_frac, 
        ticks=cbar_ticks, pad=cbar_pad_frac, orientation='horizontal', format='%1.1f', label='MRD')      
    cbar.set_clim(min_density, max_density)
    
    # Update function
    def animFunc(it):
        print("Plotting frame", it)
        i_subplot = 0
        for pole_name in pole_names_ordered:
            # Set current axes
            ax = axes.flat[i_subplot]
            sca(ax)
            gca().clear()
            
            # Get histogram
            hist = pole_history_data[pole_name][it,:,:]
            
            # Histogram dimensions
            nBins = hist.shape[0]
            maxR = nBins/2.0
            histCentre = [nBins/2.0, nBins/2.0]
            xRange = [0.0,nBins]
            yRange = [0.0,nBins]
            
            # Sum
            hist_sum = sum(sum(abs(hist)))
            local_max_density = hist.max()
            
            # Mark invalid bins with nan
            for ix in range(nBins):
                for iy in range(nBins):
                    x = ix - histCentre[0]
                    y = iy - histCentre[1]
                    if sqrt(x**2+y**2) > maxR:
                        hist[ix,iy] = nan        
            n_valid_bins = numpy.count_nonzero(hist!=nan)
            
            # Normalise to uniform density
            uniform_density = hist_sum/n_valid_bins
            hist /= uniform_density
            local_max_density /= uniform_density
            
            # Plot
            hist_masked = ma.masked_invalid(hist)
            plot_ax = pcolormesh(hist_masked.T)     
            circ = Circle(histCentre, radius=maxR, fill=False)
            gca().add_patch(circ)
            removeBorder(gca())
            xlim(1.01*array(xRange))
            ylim(1.01*array(yRange))
            title(pole_name)
            if i_subplot == 0:
                xlabel("$\mathbf{e}_1$")
                ylabel("$\mathbf{e}_2$")
            
            # Next subplot
            i_subplot += 1
        
        # Save
        savefig(base_filename+"%d.png"%(i), bbox_inches='tight')
    
    for i in timestep_selection:
        animFunc(i)
