""" A library for functions used to replicate the work in Mihaila2014.

Mihaila2014 refers to the the paper Bogdan Mihaila, 
Marko Knezevic and Andres Cardenas. Three orders of magnitude improved 
efficiency with high-performance spectral crystal plasticity on GPU 
platforms., (January):785--798, 2014

We also make reference to Kalidindi2006, which is the paper 
Surya R. Kalidindi, Hari K. Duvvuru and Marko Knezevic. Spectral 
calibration of crystal plasticity models. Acta Materialia, 
54:1795--1804, 2006
"""

from math import *
from continuum import *
from numpy import zeros, outer, array, max
import kalidindi1992
from recordclass import recordclass
from linearAlgebra import *

CrystalOrientationSpace = recordclass('CrystalOrientationSpace', ['phi_1','PHI','phi_2'])

def stretchingVelocityGradient(theta, e_dot):
    """Computes the stretching velocity gradient.
    
    This evaluates Equation 8 in Mihaila2014, which is given by
    \f{eqnarray}
    \dot{\epsilon} = |\mathbf{D}|& , & \mathbf{D}_0=\sum_{j=1}^3 
    \mathbf{D}_j \mathbf{e}_j^p \otimes \mathbf{e}_j^p,
    \f}
    where \f$\dot{\epsilon}\f$ is the strain rate, \f$\mathbf{e}_j^p\f$
    are the unit vectors in the principal frame of \f$\mathbf{D}\f$, and
    \f$\mathbf{D}\f$ is parametrised by
    \f{eqnarray}
    D_1 & = & \sqrt{\frac{2}{3}} \cos (\theta - \pi/3)\\
    D_2 & = & \sqrt{\frac{2}{3}} \cos (\theta + \pi/3)\\
    D_3 & = & - \sqrt{\frac{2}{3}} \cos (\theta),
    \f}
    where \f$\theta \in [0,2\pi)\f$ is an angular parameter for all possible
    traceless \f$3\times3\f$ diagonal matrices.
    
    The principal frame is taken to simply be the unrotated Cartesian frame.
    
    Args:
        theta: \f$\theta\f$
        e_dot: \f$\dot{\epsilon}\f$
    Returns:
        \f$\mathbf{D}\f$
    """
    D_comps = zeros((3))
    D_comps[0] = sqrt(2.0/3)*cos(theta-pi/3)
    D_comps[1] = sqrt(2.0/3)*cos(theta+pi/3)
    D_comps[2] = -sqrt(2.0/3)*cos(theta)
    D_0 = zeros((3,3))
    basis = [array([1,0,0],dtype='f8'),
             array([0,1,0],dtype='f8'),
             array([0,0,1],dtype='f8')]
    for i in range(3):
        e_i = basis[i]
        D_0 += dot(D_comps[i],numbaOuter3(e_i, e_i))
    D = e_dot*D_0
    return D

def velocityGradient(theta, e_dot):
    """Compute the velocity gradient.
    
    It is given by Equation 3 in Mihaila2014, which is
    \f[
        \mathbf{L} = \mathbf{D} + \mathbf{W}.
    \f]
    \f$\mathbf{D}\f$ is obtained from a call to mihaila2014.stretchingVelocityGradient.
    For \f$\mathbf{W}\f$, below Equation 19 in Kalidindi2006, "It is 
    implicitly assumed that the Fourier coefficients were obtained with 
    \f$\dot{\epsilon} = \dot{\epsilon}_0\f$ and \f$\mathbf{W}=\mathbf{0}\f$."
    
    Args:
        theta: \f$\theta\f$
        e_dot: \f$\dot{\epsilon}\f$
    Returns:
        \f$\mathbf{L}\f$    
    """
    D = stretchingVelocityGradient(theta, e_dot)
    W = zeros((3,3))
    L = D + W
    return L
    
def latticRotationTensor(F_e):
    """Determines the lattic rotation tensor from the elastic deformation gradient.
    
    Evaluates the statement at the top of journal page 3616 of Kalidindi2005, 
    which reads "At any point during the simulation of a given deformation 
    process, the lattice orientations in the current configuration can be 
    computed by using the rotation component \f$\mathbf{R}^*\f$ of \f$\mathbf{F}^*\f$ 
    (defined through the polar decomposition theorem)."
    
    Args:
        F_e: \f$\mathbf{F}^*\f$
    Returns:
        \f$\mathbf{R}^*\f$
    """
    R, U = polarDecomposition(F_e)
    return R

def getMainQuantities(mprops, T_0, s_0, F_p_0, g_p, theta, e_dot, dt_initial, min_strain):
    """Computes the three main quantities of interest from Mihaila2014.
    
    They are:
    - \f$\sigma'\f$: the deviatoric component of the Cauchy stress tensor 
    in the crystal
    - \f$\mathbf{W}^p\f$: the plastic spin tensor
    - \f$\sum_\beta \left|\dot{\gamma}^\beta\right|\f$: the sum of the magnitudes
    of the shear strain rates in the crystal.
    
    The quantities are computed for quasistatic loading conditions,
    as a function of the strain rate and direction. This is achieved by
    applying the strain is in the same direction, but rotating the
    crystal.    
    
    These quantities are computed using the methods in Kalidindi1992,
    by making a calls to functions in kalidindi1992.
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        T_0: the initial stress in the crystal, \f$\mathbf{T}\f$
        s_0: the initial slip system deformation resistance, \f$s_0\f$
        F_p_0: the initial plastic deformation gradient, \f$\mathbf{F}^p_0\f$
        g_p: \f$\mathbf{g}^p\f$, a mihaila2014.CrystalOrientationSpace instance defining the crystal orientation
        theta: the angular parameter for the strain, \f$\theta\f$, defined in mihaila2014.stretchingVelocityGradient
        e_dot: the strain rate, \f$\dot{\epsilon}\f$
        dt_initial: the initial timestep, \f$\Delta t\f$
        min_strain: the minimum strain \f$\epsilon\f$ before the simulation is stopped
    """
    # Rotate the crystal in the crystal orientation space
    rot_matrix = BungeEulerRotationMatrix(g_p.phi_1, g_p.PHI, g_p.phi_2)
    mprops_rotated = kalidindi1992.applyCrystalRotation(mprops, rot_matrix)
    
    # Construct the single crystal
    initial_crystals = [kalidindi1992.Crystal(mprops_rotated, T_0, s_0, F_p_0)]
    polycrystal = kalidindi1992.Polycrystal(initial_crystals)
    
    # Calculate the velocity gradient
    L = velocityGradient(theta, e_dot)
    
    # Set up the initial conditions
    dt = dt_initial
    F = eye(3)
    polycrystal.applyInitialConditions()
    t = 0
    
    # Set final time
    t_end = min_strain/e_dot
    t_end_tol = 1e-10
    
    # Simulate
    prevent_next_x_timestep_increases = 0
    while t<t_end and abs(t-t_end)/t_end > t_end_tol:
        # If timestep was decreased earlier, prevent subsequent increases
        if prevent_next_x_timestep_increases > 0:
            prevent_next_x_timestep_increases -= 1

        # Do the main solve
        try:
            step_good = False
            while not step_good:
                F_next = updateDeformationGradientFromVelocityGradient(L, F, dt)
                
                # Try step
                step_good, new_dt = polycrystal.step(F_next, dt)
                
                # If the step is not accepted, reduce the timestep
                if not step_good:
                    # Use the adjusted timestep now
                    dt = new_dt
                    prevent_next_x_timestep_increases = 10
        except numpy.linalg.linalg.LinAlgError as error:
            kalidindi1992.dumpSolverState(polycrystal, F_next, dt, "dump.txt")
            raise error
        
        # Update the current time
        t += dt
        #~ print t
        
        # Update deformation gradient
        F = F_next.copy()
        #~ print F
            
        # Set the adjusted timestep for the next step
        dt = new_dt 
    
    # Recover quantities of interest
    crystal = polycrystal.crystal_list[0]
    gammadot_alphas = kalidindi1992.ShearStrainRates(crystal._mprops, crystal._T, crystal._s_alphas)
    sigma_prime = deviatoricStressTensor(crystal._T_cauchy)
    m_alphas, n_alphas = crystal.getCurrentSlipSystems()
    W_p = kalidindi1992.PlasticSpinTensor(crystal._mprops, gammadot_alphas, m_alphas, n_alphas)
    gammadot_abs_sum = kalidindi1992.GammadotAbsSum(gammadot_alphas)
    
    # Return
    return sigma_prime, W_p, gammadot_abs_sum
