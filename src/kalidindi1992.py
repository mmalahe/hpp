""" A library for functions used to replicate the work in Kalidindi1992.

Kalidindi1992 refers to the paper Surya R. Kalidindi, Curt A. Bronkhorst 
and Lallit Anand. Crystallographic texture evolution in bulk deformation 
processing of FCC metals. Journal of the Mechanics and Physics of Solids
, 40(3):537--569, 1992.

We also make reference to Mihaila2014, which is the paper Bogdan Mihaila, 
Marko Knezevic and Andres Cardenas. Three orders of magnitude improved 
efficiency with high-performance spectral crystal plasticity on GPU 
platforms., (January):785--798, 2014

The unit system is:
- Time: seconds
- Mass: kilograms
- Length: metres
"""

import numpy
from numpy import array, zeros, sqrt, sign, tensordot, einsum, outer, eye, ones
from numpy import abs, min, max, sign, random, divide, log, maximum, argmax
from numpy.linalg import inv, det, cond, norm
from recordclass import recordclass
from linearAlgebra import *
from scipy.optimize import newton_krylov, anderson, root, fmin_cobyla
import copy
from numba import jit

## A recordclass for the material properties of a crystal
##
## A recordclass is a mutable named tuple
## Members:
## - **crystal_type**: the type of crystal. Currently only 'fcc' is supported.
## - **mu**: the elastic shear modulus \f$\mu\f$
## - **kappa**: the elastic bulk modulus \f$\kappa\f$
## - **m**: the rate sensitivty of slip, \f$m\f$
## - **gammadot_0**: the reference shearing rate \f$\dot{gamma}_0\f$
## - **h_0**, **s_s**, **a**: slip system hardening parameters. Journal page 548 in Kalidindi1992.
## - **q**: ratio of the latent hardening rate to the self-hardening rate
## - **n_alpha**: the number of slip systems
## - **m_0**: a list of the slip directions, \f$\mathbf{m}_0^\alpha\f$, for each slip system \f$\alpha\f$
## - **n_0**: a list of the slip plane normals, \f$\mathbf{n}_0^\alpha\f$ for each slip system \f$\alpha\f$
## - **S_0**: a list of the products \f$\mathbf{S}_0^\alpha \equiv \mathbf{m}_0^\alpha \otimes \mathbf{n}_0^\alpha\f$  for each slip system \f$\alpha\f$
## - **L**: the elasticity tensor \f$\mathcal{L}\f$ 
CrystalMaterialProperties = recordclass('CrystalMaterialProperties', ['crystal_type',
                                'mu','kappa','m','gammadot_0','h_0','s_s',
                                'a','q','n_alpha','m_0','n_0',
                                'S_0','L','Q'])

class Crystal():
    """This class handles a crystal grain and its evolution under
    external influences.
    
    The crystal is reponsible for its own evolution, including details
    of the numerical methods used to evaluate that evolution. However,
    it is only responsible for evaluating single timesteps. It should have no
    control over timesteps, except for suggesting them to external
    callers.
    """
    
    # Timestep control by shear rate
    ## Defined on journal page 546 of Kalidindi1992
    Dgamma_max = 1e-2
    ## Defined on journal page 546 of Kalidindi1992
    r_min = 0.8
    ## Defined on journal page 546 of Kalidindi1992
    r_max = 1.2

    # Timestep control by algebraic system convergence
    ## Defined on journal page 546 of Kalidindi1992
    algebraic_max_iter = 50
    ## Defined on journal page 547 of Kalidindi1992
    r_t = 0.75
    
    def __init__(self, mprops, T_init, s_0, F_p_0, volume=1.0):
        """Crystal constructor.
        
        Reads in and applies the crystal properties and initial conditions.
        
        Args:
            T_init: the initial stress tensor \f$\mathbf{T}\f$
            s_0: a single slip hardening rate \f$s_0\f$ applied to every slip system.
            F_p_0: the initial plastic deformation gradient \f$F^p_0\f$
            volume: the crystal volume
        """
        # Material properties
        self._mprops = mprops
        self._volume = volume
        
        # Convergence tolerances for the two-level iterative scheme
        ## Defined on journal page 545 of Kalidindi1992
        self._DT_tol = 1e-4*s_0
        ## Defined on journal page 545 of Kalidindi1992
        self._Ds_tol = 1e-3*s_0

        # Constraint on the Newton corrections for T
        ## Defined on journal page 545 of Kalidindi1992
        self._DT_max = (2.0/3.0)*s_0
        
        # Read in initial conditions
        self._T_init = T_init
        self._s_0 = s_0
        self._F_p_0 = F_p_0
        
        # Empty derived quantities
        self._m_alphas = zeros((mprops.n_alpha,3))
        self._n_alphas = zeros((mprops.n_alpha,3))
        
        # Apply initial conditions
        self.applyInitialConditions()
    
    def applyInitialConditions(self):
        """Reset the crystal to its initial conditions.
        """
        # Apply initial condititions
        self._T = self._T_init.copy()
        self._s_alphas = self._s_0*ones((self._mprops.n_alpha))
        self._F_p = self._F_p_0.copy()

        # The most recent step (the reset) has been accepted
        self._step_accepted = True
        self._step_rejected = False
        
        # The values at the next steps are no longer valid
        self._T_next = None
        self._s_alphas_next = None
        self._Dgamma_alphas_next = None
        self._F_p_next = None
        self._F_e_next = None

    def tryStep(self, F_next, dt):
        """Attempt to advance a timestep.
        
        This action doesn't actually apply the result of the attempted
        step. This must be done with kalidindi1992.Crystal.acceptStep,
        or if the results isn't satisfactory, it should be rejected
        with kalidindi1992.Crystal.rejectStep.
        
        Args:
            F_next: the deformation gradient in the crystal at time \f$t_{i+1}\f$
            dt: the timestep \f$\Delta t = t_{i+1}-t_{i}\f$
        Returns:
            'True' if the step produced a satisfactory result and
            'False' otherwise.
        """
        
        # The previous step should have been accepted xor rejected
        self.assertAcceptedOrRejectedStep()
        
        # Known quantities at the next timestep
        tensor_A = TensorA(self._F_p, F_next)
        
        # Take step
        try:
            step_good, self._T_next, self._s_alphas_next = updateTandS_Sequential(
            self._mprops, tensor_A, self._T, self._s_alphas, dt, self._DT_max, self._DT_tol, 
            self._Ds_tol, self. algebraic_max_iter)
        except numpy.linalg.linalg.LinAlgError as error:
            print "Linalg error, step not good."
            step_good = False
        self._step_good = step_good
        
        # Update derived quantities
        if self._step_good:
            self._Dgamma_alphas_next = ShearStrainIncrements(self._mprops, self._T_next, self._s_alphas_next, dt)
            self._F_p_next = PlasticDeformationGradientUpdate(self._mprops, self._F_p, self._Dgamma_alphas_next)
            self._F_e_next = F_next.dot(inv(self._F_p_next))
        
        # It is currently neither accepted or reject
        self._step_accepted = False
        self._step_rejected = False 
        
        # Report if step was successful
        return self._step_good
    
    def acceptStep(self):
        """Accept the attempted step.
        
        This takes the result of the attempted step (with kalidindi1992.Crystal.tryStep)
        and applied it to the current state of the crystal.
        """
        self._T = self._T_next.copy()
        self._s_alphas = self._s_alphas_next.copy()
        self._Dgamma_alphas = self._Dgamma_alphas_next.copy()
        self._F_p = self._F_p_next.copy()
        self._F_e = self._F_e_next.copy()
        self._step_rejected = False
        self._step_accepted = True
        
        # Update derived quantities
        self._T_cauchy = ((self._F_e.dot(self._T)).dot(self._F_e.T))/det(self._F_e)
    
    def getCurrentSlipSystems(self):
        F_e_T_inv = inv(self._F_e.T)
        for alpha in range(self._mprops.n_alpha):
            self._m_alphas[alpha,:] = self._F_e.dot(self._mprops.m_0[alpha,:])
            self._n_alphas[alpha,:] = F_e_T_inv.dot(self._mprops.n_0[alpha,:])
        return self._m_alphas, self._n_alphas
    
    def rejectStep(self):
        """Reject the attempted step.
        
        This puts the crystal back into the state it was in before the 
        step was attempted (with kalidindi1992.Crystal.tryStep).
        """
        self._T_next = None
        self._s_alphas_next = None
        self._Dgamma_alphas_next = None
        self._F_p_next = None
        self._F_e_next = None
        self._step_rejected = True
        self._step_accepted = False
        
    def assertAcceptedOrRejectedStep(self):
        """Check that the previous step was accepted or rejected.
        
        Post-processing and additional timesteps can only occur once
        the step attempted (with kalidindi1992.Crystal.tryStep) has
        been explicitly accepted or rejected. This function checks
        that condition.
        """
        if not (self._step_accepted or self._step_rejected):
            raise StandardError("Previous step must be accepted xor rejected before this operation.")
        elif (self._step_accepted and self._step_rejected):
            raise StandardError("Previous step must be accepted xor rejected before this operation.")
        
    def recommendNextTimestepSize(self, dt):
        """Given a current timestep size, recommend a new timestep size.
        
        If the step was accepted, this gives the new timestep based on
        maintaining the maximum shear rate defined by kalidindi1992.Crystal.Dgamma_max.
        If the step was rejected, gives a reduced timestep defined by
        kalidindi1992.Crystal.r_t.
        
        Args:
            dt: the current timestep
        Returns:
            The recommended timestep
        """       
        # New timestep recommendation
        if self._step_accepted:
            new_dt = SetTimestepByShearRate(dt, self._Dgamma_alphas, self.Dgamma_max, self.r_min, self.r_max)
        elif self._step_rejected:
            new_dt = self.r_t*dt
        else:
            new_dt = dt
        return new_dt
    
    def __repr__(self):
        """The unambiguous, potentially ugly print.
        
        This is called in an interactive prompt, or if a list containing
        instances of this class is printed. The goal is to have a string
        that would be sufficient for instantiating an object of this
        class in the same way that this instance was.
        """
        return "Crystal(%s,%s,%s,%s,%s)"%tuple([arg.__repr__() for arg in
        [self._mprops, self._T_init, self._s_0, self._F_p_0, self._volume]])
        
    def __str__(self):
        """The pretty, potentially ambiguous print.
        
        This is called when this class is printed. The goal is for it
        to be human readable.
        """
        return self.__repr__()
        
    
class Polycrystal():
    """This class handles a collection of crystal grains and their evolution 
    under external influences.
    
    It also handles calculating the averaged quantities in the aggregate.
    
    The polycrystal is reponsible for its own evolution, including details
    of the numerical methods used to evaluate that evolution. However,
    it is only responsible for evaluating single timesteps. It should have no
    control over timesteps, except for suggesting them to external
    callers.    
    """
    
    def __init__(self, crystal_list):
        """Polycrystal constructor
        
        Simply reads in the list of crystals.
        
        Args:
            crystal_list: a list of the kalidindi1992.Crystal instances
            that comprise the polycrystal
        """
        self.crystal_list = crystal_list
    
    def applyInitialConditions(self):
        for crystal in self.crystal_list:
            crystal.applyInitialConditions()
    
    def step(self, F_next, dt):
        """Attempt a timestep.
        
        Unlike kalidindi1992.Crystal.tryStep, the step is automatically
        accepted if the step is good in all of the crystals. If the step
        is not good in at least one crystal, it is rejected in all of the
        crystals. A new timestep is also recommended, based on the minimum
        timestep suggested by each of the crystals.
        
        Args:
            F_next: the deformation gradient in the crystal at time \f$t_{i+1}\f$
            dt: the timestep \f$\Delta t = t_{i+1}-t_{i}\f$
            
        Returns:
            The tuple (step_good, new_dt), where:
            - step_good is a boolean that indicates if the overall step was good
            - new_dt is a new recommended timestep
        """
        
        # Try stepping through all of the crystals
        step_good = True
        for crystal in self.crystal_list:
            step_good = crystal.tryStep(F_next, dt)
            if step_good == False:
                crystal.rejectStep()
                new_dt = crystal.recommendNextTimestepSize(dt)
                break        
        self._step_good = step_good
        
        # Recommend a new timestep
        if self._step_good:
            new_dts = []
            for crystal in self.crystal_list:
                crystal.acceptStep()
                new_dts.append(crystal.recommendNextTimestepSize(dt))
            new_dt = min(new_dts)
            self._updateDerivedQuantities()
        else:
            for crystal in self.crystal_list:
                crystal.rejectStep()
            
        # Return
        return self._step_good, new_dt
        
    def _updateDerivedQuantities(self):
        """Update the aggregate quantities of the polycrystal.
        
        This combines quantities in the consituent crystals into those
        in the polycrystal.
        """
        self._T_cauchy = 0.0
        volume = 0.0
        for crystal in self.crystal_list:
            self._T_cauchy += crystal._T_cauchy
            volume += crystal._volume
        self._T_cauchy /= volume

    def __repr__(self):
        """The unambiguous, potentially ugly print.
        
        This is called in an interactive prompt, or if a list containing
        instances of this class is printed. The goal is to have a string
        that would be sufficient for instantiating an object of this
        class in the same way that this instance was.
        """
        return "Polycrystal(%s)"%(self.crystal_list.__repr__())
        
    def __str__(self):
        """The pretty, potentially ambiguous print.
        
        This is called when this class is printed. The goal is for it
        to be human readable.
        """
        return self.__repr__()

def ElasticityTensor(mu, kappa):
    """Return the elasticity tensor \f$\mathcal{L}\f$ 
    
    Defined in Equation 39 of Kalidindi1992. The equation is:
    \f[
    \mathcal{L} \equiv \mu \mathcal{I} + [\kappa - (2/3) \mu] \mathbf{1}
    \otimes \mathbf{1}.
    \f]
    
    Args:
        mu: \f$\mu\f$
        kappa: \f$\kappa\f$
        
    Returns:
        \f$\mathcal{L}\f$
	"""
    L = 2*mu*fourthOrderIdentity(3) + (kappa-(2.0/3)*mu)*secondOrderOuterProduct3(eye(3),eye(3))
    return L

@jit('f8[:,:,:](i4,f8[:,:],f8[:,:])')
def getS_0_Internal(n_systems, m, n):
    """Gets the slip system outer products.
    
    This function evaluates
    \f[
    \mathbf{S}_0^\alpha = \mathbf{m}_0^\alpha \otimes \mathbf{n}_0^\alpha
    \f]
    for each slip system labeled by \f$\alpha\f$. This is based on 
    Equation 7 in Kalidindi1992. It provides a JIT-compiled inner
    loop for kalidindi1992.getS_0.
    
    Args:
        n_systems: the number of slip systems
        m: A list of \f$\mathbf{m}_0^\alpha\f$ for each slip system
        n: A list of \f$\mathbf{n}_0^\alpha\f$ for each slip system
    Returns:
        An array of the outer products for each slip system, with the 
        first dimension of the array indexing the slip system, and the 
        second and third dimensions indexing the components of the
        2nd order tensor.
    """
    S_0 = zeros((n_systems,3,3))
    for i in range(n_systems):
        S_0[i,:,:] = numbaOuter3(m[i,:], n[i,:])
    return S_0

def getS_0(m_0, n_0):
    """Gets the slip system outer products.
    
    This function evaluates
    \f[
    \mathbf{S}_0^\alpha = \mathbf{m}_0^\alpha \otimes \mathbf{n}_0^\alpha
    \f]
    for each slip system labeled by \f$\alpha\f$. This is based on 
    Equation 7 in Kalidindi1992. It simply infers the number of slip
    systems from the dimensions of \f$\mathbf{m}_0^\alpha\f$, and
    calls kalidindi1992.getS_0_Internal.
    
    Args:
        m: A list of \f$\mathbf{m}_0^\alpha\f$ for each slip system
        n: A list of \f$\mathbf{n}_0^\alpha\f$ for each slip system
    Returns:
        An array of the outer products for each slip system, with the 
        first dimension of the array indexing the slip system, and the 
        second and third dimensions indexing the components of the
        2nd order tensor.
    """
    n_systems = m_0.shape[0]
    S_0 = getS_0_Internal(n_systems, m_0, n_0)
    return S_0

def applyCrystalRotation(mprops, Q):
    """Apply a rotation to the material proprties of a crystal.
    
    Currently, this involves only rotating the slip systems of the
    crystal. The rotation matrix \f$\mathbf{Q}\f$ is applied to
    \f{eqnarray}
    \bar{\mathbf{m}}_0^\alpha & = & \mathbf{Q} \mathbf{m}_0^\alpha\\
    \bar{\mathbf{n}}_0^\alpha & = & \mathbf{Q} \mathbf{n}_0^\alpha,
    \f}
    followed by the update
    \f[
    \bar{\mathbf{S}}_0^\alpha = \bar{\mathbf{m}_0}^\alpha \otimes \bar{\mathbf{n}}_0^\alpha
    \f]
    
    Args:
        mprops: the kalidini1992.CrystalMaterialProperties instance that
        current describes the crystal before rotation.
        Q: the rotation matrix \f$\mathbf{Q}\f$
    Returns:
        A kalidini1992.CrystalMaterialProperties instance that describes
        the crystal after the rotation.
    """
    mprops_new = copy.copy(mprops)
    for alpha in range(mprops.n_alpha):
        mprops_new.m_0[alpha,:] = Q.dot(mprops.m_0[alpha,:])
        mprops_new.n_0[alpha,:] = Q.dot(mprops.n_0[alpha,:])
    mprops_new.S_0 = getS_0(mprops_new.m_0, mprops_new.n_0)
    return mprops_new

def randomizeCrystalOrientation(mprops):
    """Randomize the orientation of a crystal through its base properties.
    
    This is achieved by rotating all of the vector quantities that
    the define the crystal under the same random rotation matrix. This is
    currently just the slip plane normals and and slip directions. Makes
    use of kalidindi1992.applyCrystalRotation to apply the rotations.
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
    Returns:
        The modified material properties
    """
    Q = random_rot(3)
    mprops_new = applyCrystalRotation(mprops,Q)
    return mprops_new

def getSlipSystems(crystal_type):
    """Get the slip system parameters for a given crystal type.
    
    Currently only fcc crystals are supported. The slip systems for 
    these are defined in Table A1 of Kalidindi1992.
    
    Args:
        crystal_type: the name of the type of crystal, drawn from {'fcc'}
    Returns:
        The tuple (n_alpha, m_0, n_0, S_0), with:
        - **n_alpha**: the number of slip systems
        - **m_0**: a list of the slip directions, \f$\mathbf{m}_0^\alpha\f$, for each slip system \f$\alpha\f$
        - **n_0**: a list of the slip plane normals, \f$\mathbf{n}_0^\alpha\f$ for each slip system \f$\alpha\f$
        - **S_0**: a list of the products \f$\mathbf{S}_0^\alpha \equiv \mathbf{m}_0^\alpha \otimes \mathbf{n}_0^\alpha\f$  for each slip system \f$\alpha\f$
    """
    # Face-centred cubic crystal
    if crystal_type == 'fcc':
        # Number of systems
        n_alpha = 12
        
        # Slip directions
        m_0 = zeros((n_alpha,3))
        m_0[0,:] = array((1,-1,0))
        m_0[1,:] = array((-1,0,1))
        m_0[2,:] = array((0,1,-1))
        m_0[3,:] = array((1,0,1))
        m_0[4,:] = array((-1,-1,0))
        m_0[5,:] = array((0,1,-1))
        m_0[6,:] = array((-1,0,1))
        m_0[7,:] = array((0,-1,-1))
        m_0[8,:] = array((1,1,0))
        m_0[9,:] = array((-1,1,0))
        m_0[10,:] = array((1,0,1))
        m_0[11,:] = array((0,-1,-1))
        m_0 = m_0/sqrt(2.0)
        
        # Slip plane normals
        n_0 = zeros((n_alpha,3))
        for i in range(0,3):
            n_0[i,:] = array((1,1,1))
        for i in range(3,6):
            n_0[i,:] = array((-1,1,1))
        for i in range(6,9):
            n_0[i,:] = array((1,-1,1))
        for i in range(9,12):
            n_0[i,:] = array((-1,-1,1))
        n_0 = n_0/sqrt(3.0)
        
        # Slip systems
        S_0 = getS_0(m_0,n_0)
        
    else:
        raise LookupError("No slip system defined for \"%s\"" % (crystal_type))
    return n_alpha, m_0, n_0, S_0

def getDefaultCrystalMaterialProperties():
    """Generate the default crystal material properties in Kalidindi1992.
    
    Simply sets all the constants to those in Kalidindi1992.
    
    Returns:
        A kalidindi1992.CrystalMaterialProperties instance with the defaults in Kalidindi1992
    """
    
    # Elastic moduli (GPa)
    mu = 46.5e0
    kappa = 124.0e0

    # Power-law viscoplasticity parameters (dimensionless)
    m = 0.012

    # Power-law viscoplasticity parameters (s^{-1})
    gammadot_0 = 0.001

    # Slip hardening parameters (GPa)
    h_0 = 180.0e-3
    s_s = 148.0e-3

    # Slip hardening parameters (dimensionless)
    a = 2.25
    q = 1.4 #Bottom of journal page 549 in Kalidindi1992

    # Crystal type
    crystal_type = 'fcc'
    
    # Q tensor for FCC
    Q = TensorQ_fcc(q)

    # Number of slip systems, slip directions and slip plane normals for the crystal's slip systems     
    n_alpha, m_0, n_0, S_0 = getSlipSystems(crystal_type)
    
    L = ElasticityTensor(mu, kappa)

    mprops = CrystalMaterialProperties(crystal_type=crystal_type,
                                            mu=mu,
                                            kappa=kappa,
                                            m=m,
                                            gammadot_0=gammadot_0,
                                            h_0=h_0,
                                            s_s=s_s,
                                            a=a,
                                            q=q,
                                            n_alpha=n_alpha,
                                            m_0=m_0,
                                            n_0=n_0,
                                            S_0=S_0,
                                            L=L,
                                            Q=Q
                                            )
                                            
    return mprops

def TensorA(F_p, F_next):
	"""Return the tensor \f$\mathbf{A}\f$ defined in Equation 25 of Kalidindi1992.
    
    The equation is:
    \f[
    \mathbf{A} \equiv \mathbf{F}^{p^{-T}}(t_{i}) \mathbf{F}^T(t_{i+1})
    \mathbf{F}(t_{i+1}) \mathbf{F}^{p^{-1}}(t_{i}),
    \f]
    where \f$\mathbf{F}^{p}(t_{i})\f$ is the plastic deformation gradient
    at time \f$i\f$, and \f$\mathbf{F}(t_{i+1})\f$ is the full deformation
    gradient at time i.
    
    Args:
        F_p: \f$\mathbf{F}^{p}(t_{i})\f$
        F_next: \f$\mathbf{F}(t_{i+1})\f$
        
    Returns:
        \f$\mathbf{A}\f$
	"""
	F_p_inv = inv(F_p)
	tensor_A = (F_p_inv.T).dot(F_next.T).dot(F_next).dot(F_p_inv)
	return tensor_A

def TensorB_alpha(A, S_0_alpha):
    """Return the tensor \f$\mathbf{B}^\alpha\f$ defined in Equation 26 of Kalidindi1992.
    
    The equation is:
    \f[
    \mathbf{B}^\alpha \equiv \mathbf{A} \mathbf{S}_0^\alpha
    + \mathbf{S}_0^{\alpha^T}\mathbf{A} 
    \f]
    where \f$\mathbf{A}\f$ is the tensor returned by a call to kalidindi1992.TensorA,
    and \f$\mathbf{S}_0^\alpha\f$ is the quantity \f$\mathbf{S}_0\f$
    in kalidindi1992.CrystalMaterialProperties, evaluated for the
    slip system \f$\alpha\f$.
    
    Args:
        A: \f$\mathbf{A}\f$
        S_0_alpha: \f$\mathbf{S}_0^\alpha\f$
        
    Returns:
        \f$\mathbf{B}^\alpha\f$
	"""
    return AB_plusB_T_A3(A, S_0_alpha)
    
def TensorB_alpha_in_place(A, S_0_alpha, B_alpha):
    AB_plusB_T_A3_in_place(A, S_0_alpha, B_alpha)

@jit('f8[:,:,:](f8[:,:,:,:],f8[:,:],f8[:,:,:],i4)')
def TensorC_alphas(L, A, S_0, n_alpha):
    """Return the tensor \f$\mathbf{C}^\alpha\f$ defined in Equation 29 of Kalidindi1992.
    
    The equation is:
    \f[
    \mathbf{C}^\alpha \equiv \mathcal{L}\left[\frac{1}{2} \mathbf{B}^\alpha \right],
    \f]
    where \f$\mathcal{L}\f$ is a fourth order elasticity tensor, 
    and \f$\mathbf{B}^\alpha\f$ is the tensor returned by a call to 
    kalidindi1992.TensorB_alpha.
    
    Args:
        L: \f$\mathcal{L}\f$
        B_alpha: \f$\mathbf{B}^\alpha\f$
        
    Returns:
        \f$\mathbf{C}^\alpha\f$
	"""
    C_alphas = zeros((n_alpha, 3, 3))
    B_alpha = zeros((3,3))
    for alpha in range(n_alpha):
        AB_plusB_T_A3_in_place(A, S_0[alpha,:,:], B_alpha)
        tensordotKalidindi4_2_in_place(L, 0.5*B_alpha, C_alphas[alpha,:,:])
    return C_alphas

@jit('void(f8[:,:], i4, f8[:], f8[:,:,:])',nopython=True)
def TensorGInnerUpdate(G, n_alpha, Dgamma_alphas, C_alphas):
    """Computes the inner update loop of kalidindi1992.TensorG.
    
    This is provided to make the inner loop of kalidindi1992.TensorG 
    JIT compilable. See that function for the definitions.
    
    Args:
        G: the tensor \f$\mathbf{G}\f$
        n_alpha: the number of slip systems
        Dgamma_alphas: list of \f$\Delta \gamma^{\alpha} (\mathbf{T}^*_n(t_{i+1}), s_k^{\alpha}(t_{i+1}))\f$ for each \f$\alpha\f$
        C_alphas: list of \f$\mathbf{C}^\alpha\f$ for each \f$\alpha\f$
    Returns:
        None. The tensor \f$\mathbf{G}\f$ is updated in place.
    """
    for alpha in range(n_alpha):
        G += Dgamma_alphas[alpha]*C_alphas[alpha,:,:]

@jit('f8[:,:](f8[:,:,:,:],f8[:,:],f8[:,:],f8[:],f8[:,:,:])')
def TensorG(L, A, T_prev_iter, Dgamma_alphas, C_alphas):
    """Return the tensor \f$\mathbf{G}\f$ defined in Equation 32 of Kalidindi1992.
    
    The equation is:
    \f[
    \mathbf{G} \equiv \mathbf{T}_{n}^*(t_{i+1}) - \mathbf{T}^{*tr}(t_{i+1})
    + \sum_{\alpha} \Delta \gamma^{\alpha} (\mathbf{T}^*_n(t_{i+1}), s_k^{\alpha}(t_{i+1}))
    \mathbf{C}^\alpha
    \f]
    where \f$\mathbf{T}^*_n(t_{i+1})\f$ is the stress in the crystal at iteration 
    \f$n\f$ and time \f$i+1\f$, \f$s_k^{\alpha}(t_{i+1})\f$ is the 
    slip deformation resistance at iteration \f$ k \f$ and time \f$ i+1 \f$,
    \f$\mathbf{C}^\alpha\f$ is the tensor returned by a call to 
    kalidindi1992.TensorC_alpha, and 
    \f[
    \mathbf{T}^{*tr} \equiv \mathcal{L}\left[\frac{1}{2} (\mathbf{A}-\mathbf{1})\right],
    \f]
    where \f$\mathcal{L}\f$ is a fourth order elasticity tensor, 
    and \f$\mathbf{A}\f$ is the tensor returned by a call to 
    kalidindi1992.TensorA.
    
    Args:
        L: \f$\mathcal{L}\f$
        A: \f$\mathbf{A}\f$
        T_prev_iter: \f$\mathbf{T}_{n}^*(t_{i+1})\f$
        Dgamma_alphas: list of \f$\Delta \gamma^{\alpha} (\mathbf{T}^*_n(t_{i+1}), s_k^{\alpha}(t_{i+1}))\f$ for each \f$\alpha\f$
        C_alphas: list of \f$\mathbf{C}^\alpha\f$ for each \f$\alpha\f$
        
    Returns:
        \f$\mathbf{G}\f$
    """
    
    # Tensor T_tr: defined in Equation 28 of Kalidindi1992
    T_tr = tensordotKalidindi4_2(L,0.5*(A-eye(3)))
    
    # Evaluating Equation 32 of Kalidindi1992
    G = T_prev_iter - T_tr
    n_alpha = len(Dgamma_alphas)
    TensorGInnerUpdate(G, n_alpha, Dgamma_alphas, C_alphas)
        
    # Return
    return G

@jit('f8(f8,f8,f8,f8)',nopython=True)
def PlasticShearingRate(tau_alpha, s_alpha, gammadot_0, m):
    """Get The shearing rate \f$\dot{\gamma}^{\alpha}\f$.
    
    Defined in Equation 40 of Kalidindi 1992. The equation is:
    \f[
    \dot{\gamma}^\alpha = \dot{\gamma}_0 \left| \frac{\tau^\alpha}{s^\alpha}\right|^{1/m} 
    {\mathrm{sign}}(\tau^\alpha)
    \f]
    where \f$\dot{\gamma}_0\f$ is the reference shearing rate, \f$m\f$ is the
    rate of sensitivity of slip, both defined in kalidindi1992.CrystalMaterialProperties,
    \f$\tau^\alpha\f$ is the stress resolved on slip system \f$\alpha\f$,
    and \f$s^\alpha\f$ is the slip resistance on slip system \f$\alpha\f$.
    
    Args:
        tau_alpha: \f$\tau^\alpha\f$
        s_alpha: \f$s^\alpha\f$
        gammadot_0: \f$\dot{\gamma}_0\f$
        m: \f$m\f$
    Returns:
        \f$\dot{\gamma}^\alpha\f$
    """
    gammadot_alpha = gammadot_0*(abs(tau_alpha/s_alpha)**(1.0/m))*sign(tau_alpha)
    return gammadot_alpha

def PlasticShearingRateDigitsLost(mprops, T, s_alphas):
    """Get the number of decimal digits lost in calculating kalidindi1992.PlasticShearingRate
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        T: \f$\mathbf{T}^*(t_{i+1})\f$
        s_alphas: a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
    Returns
        The theoretical number of digits lost as a float.
    """
    n_digits_lost_max = 0.0
    for alpha in range(mprops.n_alpha):
        n = 1.0/mprops.m
        tau_alpha = tensorInner2_2(T, mprops.S_0[alpha,:,:])
        x = tau_alpha/s_alphas[alpha]
        if n*abs(x)**(n-1) > 0:
            n_digits_lost = log(n*abs(x)**(n-1))/log(10)
        else:
            n_digits_lost = 0
        n_digits_lost_max = maximum(n_digits_lost, n_digits_lost_max)
    return n_digits_lost_max

@jit('void(i4,f8[:,:,:],f8,f8,f8[:,:],f8[:],f8[:])',nopython=True)
def ShearStrainRatesInner(n_alpha, S_0, gammadot_0, m, T, s_alphas, gammadot_alphas):
    """Computes the inner update loop of kalidindi1992.ShearStrainRates.
    
    This is provided to make the inner loop of kalidindi1992.ShearStrainRates 
    JIT compilable. See that function and kalidindi1992.PlasticShearingRate
    for the definitions.
    
    Args:
        n_alpha: the number of slip systems
        S_0: \f$\mathbf{S}_0^\alpha\f$ for each slip system \f$\alpha\f$
        gammadot_0: \f$\dot{\gamma}_0\f$
        m: \f$m\f$
        T: \f$\mathbf{T}^*\f$
        s_alphas: a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
        gammadot_alphas: an array of \f$\dot{\gamma}^\alpha\f$ for each slip system \f$\alpha\f$ 
    Returns:
        None. The \f$\dot{\gamma}^\alpha\f$ values are updated in place.
    """
    for alpha in range(n_alpha):
        tau_alpha = tensorInnerKalidindi2_2(T, S_0[alpha,:,:])
        gammadot_alpha = PlasticShearingRate(tau_alpha, s_alphas[alpha], gammadot_0, m)
        gammadot_alphas[alpha] = gammadot_alpha

def ShearStrainRates(mprops, T, s_alphas):
    """Computes the shear strain rates.
    
    The functional form of the strain rates is defined in Equation 18 
    and Equation 19 of Kalidindi1992. This form is
    \f[
        \dot{\gamma}^\alpha = \dot{\gamma}^\alpha(\mathbf{T}^*,s^\alpha)
    \f]    
    In this case, the function specifically evaluates the form given in 
    Equation 40 of Kalidindi. The arguments are processed to be used in 
    a direct call to kalidindi1992.PlasticShearingRate, which evaluates
    that form.
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        T: \f$\mathbf{T}^*\f$
        s_alphas: s_alphas: a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
    Returns:
        An array of \f$\dot{\gamma}^\alpha\f$ for each slip system \f$\alpha\f$
    """
    gammadot_alphas = zeros((mprops.n_alpha))
    ShearStrainRatesInner(mprops.n_alpha, mprops.S_0, mprops.gammadot_0, mprops.m, T, s_alphas, gammadot_alphas)
    return gammadot_alphas

def ShearStrainIncrements(mprops, T, s_alphas, dt):
    """Get the shear strain increments .
    
    Defined in Equation 18 of  Kalidindi 1992. The equation is:
    \f[
    \Delta \gamma^\alpha = \dot{\gamma}^\alpha(\mathbf{T}^*(t_{i+1}):\mathbf{S}_0^\alpha),
    s^\alpha(t_{i+1})) \Delta t
    \f]
    where \f$\dot{\gamma}^\alpha(...)\f$ is the function kalidindi1992.PlasticShearingRate,
    \f$\mathbf{T}^*(t_{i+1})\f$ is the stress in the crystal at time
    \f$i+1\f$, \f$s^\alpha(t_{i+1})\f$ is the slip deformation resistance in 
    the slip system \f$\alpha\f$ and \f$\mathbf{S}_0^\alpha\f$ is the 
    quantity \f$\mathbf{S}_0\f$     in kalidindi1992.CrystalMaterialProperties, 
    evaluated for the slip system \f$\alpha\f$.
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        T: \f$\mathbf{T}^*(t_{i+1})\f$
        s_alphas: a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
        dt: \f$\Delta t\f$
    Returns:
        A list of \f$\Delta \gamma^\alpha\f$ for each \f$\alpha\f$
    """
    gammadot_alphas = ShearStrainRates(mprops, T, s_alphas)
    Dgamma_alphas = gammadot_alphas*dt        
    return Dgamma_alphas

@jit('f8[:,:](f8,f8,f8[:,:],f8[:,:],f8,f8)')
def PartialDGammaPartialT(m, gammadot_0, T, S_0_alpha, s_alpha, dt):
    """Evaluate \f$\frac{\partial}{\partial \mathbf{T}} \Delta \gamma^\alpha\f$
    
    This is the derivative of an entry of kalidindi1992.ShearStrainIncrements
    with respect to \f$\mathbf{T}\f$. This expression is used in
    kalidindi1992.TensorJ.
    
    Test coverage in kalidindi1992_test.TestDirectExpressions.test_PartialDGammaPartialT
    """
    tau_alpha = tensorInnerKalidindi2_2(T, S_0_alpha)
    return (1.0/m)*gammadot_0*(abs(s_alpha)**(-1.0/m))*(abs(tau_alpha)**(1.0/m-1.0))*dt*S_0_alpha

@jit('void(f8[:,:,:,:], i4, f8, f8, f8[:,:,:], f8[:,:,:], f8[:], f8[:,:], f8)',nopython=True)
def TensorJInnerUpdate(J, n_alpha, m, gammadot_0, S_0, C_alphas, s_alphas, T, dt):
    """Computes the inner update loop of kalidindi1992.TensorJ.
    
    This is provided to make the inner loop of kalidindi1992.TensorJ 
    JIT compilable. See that function for the definitions.
    
    Args:
        J: the tensor \f$\mathbf{J}\f$
        n_alpha: the number of slip systems
        m: \f$m\f$
        gammadot_0: \f$\dot{\gamma}_0\f$
        S_0: \f$\mathbf{S}_0^\alpha\f$ for each slip system \f$\alpha\f$
        C_alphas: list of \f$\mathbf{C}^\alpha\f$ for each \f$\alpha\f$
        s_alphas: a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
        T: \f$\mathbf{T}^*(t_{i+1})\f$
        dt: \f$\Delta t\f$
    Returns:
        None. \f$\mathbf{J}\f$ is updated in place.
    """
    for alpha in range(n_alpha):
        partial_Dgamma_partial_T = PartialDGammaPartialT(m, gammadot_0, T, S_0[alpha,:,:], s_alphas[alpha], dt)
        J += secondOrderOuterProduct3(C_alphas[alpha,:,:], partial_Dgamma_partial_T)

def TensorJ(mprops, C_alphas, s_alphas, T, dt):
    """Get the tensor J 
    
    Defined in Equation 33 of Kalidindi1992. The equation is:
    \f[
    \mathbf{J} \equiv \mathbf{I} + \sum_\alpha \mathbf{C}^\alpha \otimes
    \frac{\partial}{\partial \mathbf{T}_n^*(T_{i+1})} \Delta \gamma^\alpha
    (\mathbf{T}^*_n(t_{i+1}), s_k^\alpha(t_{i+1})).
    \f]
    This evaluates to 
    \f[
    \mathbf{J} = \mathbf{I} + \frac{1}{m} \sum_\alpha \dot{\gamma}_0 
    |\tau^\alpha|^{1/m-1} |s^\alpha|^{-1/m} \Delta t \mathbf{C}^\alpha 
    \otimes \mathbf{S_0^\alpha}. 
    \f]
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        C_alphas: list of \f$\mathbf{C}^\alpha\f$ for each \f$\alpha\f$
        s_alphas: a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
        T: \f$\mathbf{T}^*(t_{i+1})\f$
        dt: \f$\Delta t\f$
    Returns:
        The tensor \f$\mathbf{J}\f$
    """
    J = fourthOrderIdentity(3)
    TensorJInnerUpdate(J, mprops.n_alpha, mprops.m, mprops.gammadot_0, mprops.S_0, C_alphas, s_alphas, T, dt)     
    return J

def NewtonStressCorrection(mprops, tensor_A, C_alphas, s_alphas, T_old, DT_max, dt):
    """Get the Newton's method update of the stress in the crystal.

    This evaluates the final term (including the negative sign) in Equation
    31 of Kalidindi1992. The equation is:
    \f[
    \mathbf{T}^*_{n+1}(t_{i+1}) = \mathbf{T}^*_{n}(t_{i+1}) 
    - \mathcal{J}_n^{-1} [\mathbf{G}_n],
    \f]
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        tensor_A: the tensor \f$\mathbf{A}\f$ returned by a call to kalidindi1992.TensorA
        Dgamma_alphas: list of \f$\Delta \gamma^{\alpha} (\mathbf{T}^*_n(t_{i+1}), s_k^{\alpha}(t_{i+1}))\f$ for each \f$\alpha\f$
        C_alphas: list of \f$\mathbf{C}^\alpha\f$ for each \f$\alpha\f$
        s_alphas: a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
        T_old: \f$\mathbf{T}^*_n(t_{i+1})\f$
        DT_max: the maximum absolute value allowed for any component of DT
        dt: \f$\Delta t\f$
    Returns:
        The correction \f$\Delta \mathbf{T}^* \equiv - \mathcal{J}_n^{-1} [\mathbf{G}_n]\f$
    """
    # Shear strain increments
    Dgamma_alphas = ShearStrainIncrements(mprops, T_old, s_alphas, dt)
        
    # Tensor J
    J = TensorJ(mprops, C_alphas, s_alphas, T_old, dt)
    
    # Tensor G: defined in Equation 30 of Kalidindi1992
    G = TensorG(mprops.L, tensor_A, T_old, Dgamma_alphas, C_alphas)
    
    # Invert J
    J_inv = fourthOrderInverse3(J)
    
    # Calculate the correction
    DT = tensordotKalidindi4_2(-J_inv,G)

    # Constrain correction. Equation 35 in Kalidindi1992
    DT = constrainComponents3by3(DT, DT_max)

    # Return
    return DT 

@jit('f8(f8,f8,f8,f8)',nopython=True)
def SlipHardeningRate(h_0, s_s, a, s_beta):
    """
    Get the slip hardening rate from the slip deformation resistance in a single system.
    
    This evaluates Equation 43 in Kalidindi1992. The equation is:
    \f[
    h^{(\beta)} = h_0 \left(1-\frac{s^\beta}{s_s}\right)^a
    \f]
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        s_beta: \f$s^\beta\f$
    Returns:
        \f$h^{(\beta)}\f$
    """
    return h_0*((1.0-s_beta/s_s))**a

def TensorQ_fcc(q):
    """Get the tensor \mathbf{Q} defined in Equation 42 of Kalidindi1992
    
    The equation is:
    \f{equation}{
    \mathbf{Q}^{\alpha \beta} = 
    \begin{bmatrix}
    A & qA & qA & qA \\
    qA & A & qA & qA \\
    qA & qA & A & qA \\
    qA & qA & qA & A   
    \end{bmatrix},
    \f}
    
    where:
    \f{equation}{
    A = \begin{bmatrix}
    1 & 1 & 1\\
    1 & 1 & 1\\
    1 & 1 & 1
    \end{bmatrix}
    \f}
    
    Args:
        q: the slip hardening parameter \f$q\f$
    Returns:
        \f$\mathbf{Q}\f$
    """    
    Q = zeros((12,12))
    A = ones((3,3))
    for i in range(4):
        for j in range(4):
            if i==j:
                Q[i*3:(i+1)*3,j*3:(j+1)*3] = A
            else:
                Q[i*3:(i+1)*3,j*3:(j+1)*3] = q*A
    
    return Q

@jit('void(f8[:,:], i4, f8, f8, f8, f8[:], f8[:,:])',nopython=True)
def StrainHardeningRatesHUpdate(h, n_alpha, h_0, s_s, a, s_alphas, Q):
    """Computes the inner update loop of kalidindi1992.StrainHardeningRates.
    
    This is provided to make the inner loop of kalidindi1992.StrainHardeningRates 
    JIT compilable. See that function for the definitions.
    
    Args:
        h: \f$h\f$
        n_alpha: the number of slip systems
        h_0: \f$h_0\f$
        s_s: \f$s_s\f$
        a: \f$a\f$
        s_alphas: a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
        Q: \f$\mathbf{Q}\f$
    Returns:
        None. Each \f$h^{\alpha \beta}\f$ is updated in place.
    """
    for j in range(n_alpha):
        h_b = SlipHardeningRate(h_0, s_s, a, s_alphas[j])
        for i in range(n_alpha):
            h[i,j] = Q[i,j]*h_b

def StrainHardeningRates(mprops, s_alphas):
    """Calculate the strain hardening rates.
    
    Evaluates Equation 41 in Kalidindi1992. The equation is:
    \f[
    h^{\alpha \beta} = \mathbf{Q}^{\alpha \beta} h^{(\beta)},
    \f]
    where there is no sum on \f$\beta\f$.
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        s_alphas: a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
        Q: \f$\mathbf{Q}\f$
    Returns:
        \f$h^{\alpha \beta}\f$
    """       
    h = zeros((mprops.n_alpha,mprops.n_alpha))
    StrainHardeningRatesHUpdate(h, mprops.n_alpha, mprops.h_0, mprops.s_s, mprops.a, s_alphas, mprops.Q)
    return h

@jit('f8[:](i4,f8[:],f8[:,:],f8[:])')
def SlipDeformationResistanceUpdateInner(n, s_alphas_prev_time, h, Dgamma_alphas):
    """Computes the inner update loop of kalidindi1992.SlipDeformationResistanceUpdate.
    
    This is provided to make the inner loop of kalidindi1992.SlipDeformationResistanceUpdate
    JIT compilable. See that function for the definitions.
    
    Args:
        n: the number of slip systems
        s_alphas_prev_time: a list of \f$s^\alpha(t)\f$ for each slip system \f$\alpha\f$
        h: the strain hardening rates \f$h\f$
        Dgamma_alphas: list of \f$\Delta \gamma^{\alpha} (\mathbf{T}^*_n(t_{i+1}), s_k^{\alpha}(t_{i+1}))\f$ for each \f$\alpha\f$
    Returns:
        A list of \f$s^\alpha_{k+1}(t_{i+1})\f$ for each slip system \f$\alpha\f$
    """
    s_alphas = zeros((n))
    for a in range(n):
        s_alphas[a] = s_alphas_prev_time[a]
        for b in range(n):
            s_alphas[a] += h[a,b]*abs(Dgamma_alphas[b])
    return s_alphas

def SlipDeformationResistanceUpdate(mprops, T_next, s_alphas_prev_time, s_alphas_prev_iter, dt):
    """Get the updates of the deformation resistance in the slip systems.

    Evaluates Equation 36 and its predecessors in Kalidindi1992. The equation is:
    \f[
    s_{k+1}^\alpha(t_{i+1}) = s^\alpha(t) + \sum_\beta h^{\alpha\beta} (s_k^\beta(t_{i+1}))
    \left|\Delta \gamma^\beta(\mathbf{T}^*_{n+1} (t_{i+1}), s_k^\beta(t_{i+1}))\right|
    \f]

    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        T_next: \f$\mathbf{T}^*(t_{i+1})\f$
        s_alphas_prev_time: a list of \f$s^\alpha(t)\f$ for each slip system \f$\alpha\f$
        s_alphas_prev_iter: a list of \f$s^\alpha_k(t_{i+1})\f$ for each slip system \f$\alpha\f$
        dt: \f$\Delta t\f$
    Returns:
        A list of \f$s^\alpha_{k+1}(t_{i+1})\f$ for each slip system \f$\alpha\f$
    """
    # Evaluate shear strain increments
    Dgamma_alphas = ShearStrainIncrements(mprops, T_next, s_alphas_prev_iter, dt)
        
    # Evaluating Equation 41 in Kalidindi1992
    h = StrainHardeningRates(mprops, s_alphas_prev_iter)
    
    # Evaluating Equation 36 in Kalidindi 1992
    s_alphas = SlipDeformationResistanceUpdateInner(mprops.n_alpha, s_alphas_prev_time, h, Dgamma_alphas)

    # Return
    return s_alphas

def updateT(mprops, tensor_A, T_init, s_alphas_current_time, dt, DT_max, DT_tol, max_iter):
    """Update the stress in a crystal.

    Test coverage in kalidindi1992_test.TestIterativeSchemes.test_updateT.
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        tensor_A: the tensor \f$\mathbf{A}\f$ returned by a call to kalidindi1992.TensorA
        T_init: an initial stress \f$\mathbf{T}^*\f$ for the algebraic solver
        s_alphas_current_time: a list of \f$s^\alpha(t_i)\f$ for each slip system \f$\alpha\f$
        dt: the timestep \f$\Delta t\f$
        DT_max: the maximum incredment in \f$\mathbf{T}\f$, \f$\Delta T_{max}\f$
        DT_tol: the absolute change in \f$\mathbf{T}\f$ at which the iteration should be considered converged
        max_iter: the maximum number of algebraic iterations
    Returns:
        The tuple (step_accepted_for_T, T_iter), where:
        - **step_accepted_for_T**: is a boolean that's true if the step was good
        - **T_iter**: \f$\mathbf{T}\f$ at the final iteration
    """
    # Initial T for iterative scheme is T from the previous timestep
    T_iter = T_init.copy()
    
    # Tensors C_alpha
    C_alphas = TensorC_alphas(mprops.L, tensor_A, mprops.S_0, mprops.n_alpha)
    
    # Run the iteration
    step_accepted_for_T = False
    for n in range(max_iter):                
        # Evaluate the iteration step
        DT = NewtonStressCorrection(mprops, tensor_A, C_alphas, s_alphas_current_time, T_iter, DT_max, dt)
        T_iter = T_iter + DT
        
        # Convergence criterion in Kalidindi1992
        if max(abs(DT)) < DT_tol:
            step_accepted_for_T = True
            break
            
    return step_accepted_for_T, T_iter

def updateS(mprops, T_next, s_alphas_current_time, s_init, dt, Ds_tol, max_iter):
    """Updates the slip system deformation resistances in the crystal.
    
    Test coverage in kalidindi1992_test.TestIterativeSchemes.test_updateS.
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        T_next: the stress \f$\mathbf{T}^*(t_{i+1})\f$
        s_alphas_current_time: a list of \f$s^\alpha(t_i)\f$ for each slip system \f$\alpha\f$
        s_init: a list of initial \f$s^\alpha\f$ values for the algebraic solver
        dt: the timestep \f$\Delta t\f$
        Ds_tol: the absolute change in \f$\mathbf{T}\f$ at which the iteration should be considered converged
        max_iter: the maximum number of algebraic iterations
    Returns:
        The tuple (step_accepted_for_s, s_alphas_iter), where:
        - **step_accepted_for_s**: is a boolean that's true if the step was good
        - **s_alphas_iter**: \f$s^\alpha\f$ for each slip system \f$\alpha\f$ at the final iteration
    """
    # Initial s for iterative scheme is s from the previous timestep
    s_alphas_prev_iter = s_init.copy()
    
    # Second level of iterative scheme, for s
    step_accepted_for_s = False					
    for k in range(max_iter):
        
        # Evaluate the iteration step
        s_alphas_iter = SlipDeformationResistanceUpdate(mprops, T_next, s_alphas_current_time, s_alphas_prev_iter, dt)
        Ds = s_alphas_iter - s_alphas_prev_iter
        
        # Convergence criterion in Kalidindi1992
        if max(abs(Ds)) < Ds_tol:
            step_accepted_for_s = True
            break
        
        # This is will be the new "previous" iteration
        s_alphas_prev_iter = s_alphas_iter.copy()

    return step_accepted_for_s, s_alphas_iter

def updateTandS_Sequential(mprops, A, T_current_time, s_alphas_current_time, dt, DT_max, DT_tol, Ds_tol, max_iter):
    """Updates the stress and slip system deformation resistances in the crystal.
    
    Is simply a call to kalidindi1992.updateT followed by a call to
    kalidindi1992.updateS, ensuring that both updates converge.
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        A: the tensor \f$\mathbf{A}\f$ returned by a call to kalidindi1992.TensorA
        T_current_time: the stress \f$\mathbf{T}^*(t_{i})\f$
        s_alphas_current_time: a list of \f$s^\alpha(t_i)\f$ for each slip system \f$\alpha\f$
        dt: the timestep \f$\Delta t\f$
        DT_max: the maximum incredment in \f$\mathbf{T}\f$, \f$\Delta T_{max}\f$
        DT_tol: the absolute change in \f$\mathbf{T}\f$ at which the iteration should be considered converged
        Ds_tol: the absolute change in \f$\mathbf{T}\f$ at which the iteration should be considered converged
        max_iter: the maximum number of algebraic iterations
    Returns:
        The tuple (step_accepted, T_next, s_alphas_next), where:
        - **step_accepted**: is a boolean that's true if the step was good
        - **T_next**: the stress \f$\mathbf{T}^*(t_{i+1})\f$
        - **s_alphas_next**: \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
    """
    
    # T step
    step_accepted_for_T, T_next = updateT(mprops, A, T_current_time, s_alphas_current_time, dt, DT_max, DT_tol, max_iter)

    # S step
    step_accepted_for_s = False
    s_alphas_next = None
    if step_accepted_for_T:
        step_accepted_for_s, s_alphas_next = updateS(mprops, T_next, s_alphas_current_time, s_alphas_current_time, dt, Ds_tol, max_iter)
    step_accepted = step_accepted_for_T and step_accepted_for_s
    return step_accepted, T_next, s_alphas_next

def PlasticDeformationGradientInverseUpdate(mprops, F_p_inv_prev_time, Dgamma_alphas):
    """Update the plastic deformation gradient inverse
    
    Evaluates Equation 20 in Kalidindi1992. The equation is:
    \f[
    \mathbf{F}^{p^{-1}} (t_{i+1}) = \left( \mathbf{1}-\sum_\alpha \Delta \gamma^\alpha \mathbf{S}_0^\alpha \right) \mathbf{F}^{p^{-1}} (t_i).
    \f]
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        F_p_inv_prev_time: \f$\mathbf{F}^{p^{-1}} (t_i)\f$
        Dgamma_alphas: list of \f$\Delta \gamma^{\alpha} (\mathbf{T}^*_n(t_{i+1}), s_k^{\alpha}(t_{i+1}))\f$ for each \f$\alpha\f$  
    Returns:
        \f$\mathbf{F}^{p^{-1}} (t_{i+1})\f$
    """ 
    F_p_inv = eye(3)
    for alpha in range(mprops.n_alpha):
        F_p_inv -= Dgamma_alphas[alpha]*mprops.S_0[alpha,:,:]
    F_p_inv = F_p_inv.dot(F_p_inv_prev_time)

    # Force unit determinant
    det_F_p_inv = det(F_p_inv)
    F_p_inv /= det_F_p_inv**(1.0/3)

    # Return
    return F_p_inv

@jit('void(f8[:,:],f8[:],f8[:,:,:],i4)')
def PlasticDeformationGradientUpdateInner(F_p, Dgamma_alphas, S_0, n_alpha):
    for alpha in range(n_alpha):
        F_p += Dgamma_alphas[alpha]*S_0[alpha,:,:]

def PlasticDeformationGradientUpdate(mprops, F_p_prev_time, Dgamma_alphas):
    """Update the plastic deformation gradient
    
    Evaluates Equation 17 in Kalidindi1992. The equation is:
    \f[
    \mathbf{F}^p (t_{i+1}) = \left( \mathbf{1}+\sum_\alpha \Delta \gamma^\alpha \mathbf{S}_0^\alpha \right) \mathbf{F}^p (t_i).
    \f]
    It is further divided by $^3\sqrt{\det{\mathbf{F}^p}}$ to force a unit determinant.
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        F_p_prev_time: \mathbf{F}^p (t_i)
        Dgamma_alphas: list of \f$\Delta \gamma^{\alpha} (\mathbf{T}^*_n(t_{i+1}), s_k^{\alpha}(t_{i+1}))\f$ for each \f$\alpha\f$  
    Returns:
        \f$\mathbf{F}^p (t_{i+1})\f$
    """ 
    F_p = eye(3)
    PlasticDeformationGradientUpdateInner(F_p, Dgamma_alphas, mprops.S_0, mprops.n_alpha)
    F_p = F_p.dot(F_p_prev_time)

    # Force unit determinant
    scaleToUnitDeterminant3(F_p)

    # Return
    return F_p

def SetTimestepByShearRate(dt_old, DGamma_alphas, Dgamma_goal, r_min, r_max):
    """Set the new timestep based on a maximum shear rate.
    
    Based on Section 4.3 of Kalidindi 1992. Defining \f$R\f$ to be
    \f[
    R = \frac{\Delta \gamma_{\mathrm{max}}^\alpha}{\Delta \gamma_s}
    \f]
    The timestep is then set according to
    \f{eqnarray}{
    R < R_{\mathrm{min}} & \Rightarrow & \Delta t_{i+1} = \Delta t_i (R_{\mathrm{max}}-R_{\mathrm{min}})/2R_{\mathrm{min}}\\
    R_{\mathrm{max}} > R \geq R_{\mathrm{min}} & \Rightarrow & \Delta t_{i+1} = \Delta t_i (R_{\mathrm{max}}-R_{\mathrm{min}})/2R\\
    R \geq R_{\mathrm{max}} & \Rightarrow & \Delta t_{i+1} = \Delta t_i (R_{\mathrm{max}}-R_{\mathrm{min}})/2R
    \f}
    
    Args:
        dt_old: \f$\Delta t_i\f$
        Dgamma_alphas: list of \f$\Delta \gamma^{\alpha} (\mathbf{T}^*_n(t_{i}), s_k^{\alpha}(t_{i}))\f$ for each \f$\alpha\f$
        Dgamma_goal: \f$\Delta \gamma_s\f$
        r_min: \f$R_{\mathrm{min}}\f$
        r_max: \f$R_{\mathrm{max}}\f$
    """
    r = max(abs(DGamma_alphas))/Dgamma_goal
    #print "r = %1.3g"%(r)
    if r<r_min:
        dt = dt_old*(r_max+r_min)/(2*r_min)
    elif r >= r_min and r < r_max:
        dt = dt_old*(r_max+r_min)/(2*r)
    else:
        dt = dt_old*(r_max+r_min)/(2*r)
    return dt

@jit('void(f8[:,:],i4,f8[:],f8[:,:],f8[:,:])',nopython=True)
def PlasticSpinTensorInnerUpdate(W_p, n, gammadot_alphas, m_alphas, n_alphas):
    """Computes the inner update loop of kalidindi1992.PlasticSpinTensor.
    
    This is provided to make the inner loop of kalidindi1992.PlasticSpinTensor
    JIT compilable. See that function for the definitions.
    
    Args:
        W_p: the plastic spin tensor \f$\mathbf{W}^p\f$
        n: the number of slip systems
        gammadot_alphas: list of \f$\dot{\gamma}^{\alpha} (\mathbf{T}^*_n(t_{i+1}), s_k^{\alpha}(t_{i+1}))\f$ for each \f$\alpha\f$
        m_alphas: an array of \f$\mathbf{m}^\alpha\f$ for each slip system \f$\alpha\f$
        n_alphas: an array of \f$\mathbf{n}^\alpha\f$ for each slip system \f$\alpha\f$
    Returns:
        None. \f$\mathbf{W}^p\f$ is updated in place.
    """
    for alpha in range(n):
        gammadot_alpha = gammadot_alphas[alpha]
        m_alpha = m_alphas[alpha,:]
        n_alpha = n_alphas[alpha,:]
        W_p += gammadot_alpha*(numbaOuter3(m_alpha, n_alpha)-numbaOuter3(n_alpha, m_alpha))
    W_p *= 0.5

def PlasticSpinTensor(mprops, gammadot_alphas, m_alphas, n_alphas):
    """Calculate the plastic spin tensor.
    
    Evaluates Equation 14 in Mihaila2014, which is:
    \f[
    \mathbf{W}^p = \frac{1}{2} \sum_\alpha \dot{\gamma}^\alpha(\mathbf{m}^\alpha \otimes \mathbf{n}^\alpha - \mathbf{n}^\alpha \otimes \mathbf{m}^\alpha)
    \f]    
    This feeds into the calculation in Equation 17 of Mihaila2014.
    
    Args:
        mprops: the properties of the material defined in kalidindi1992.CrystalMaterialProperties
        gammdot_alphas: \f$\dot{\gamma}^\alpha\f$
        m_alphas: \f$\mathbf{m}^\alpha\f$ for each slip system \f$\alpha\f$
        n_alphas: \f$\mathbf{n}^\alpha\f$ for each slip system \f$\alpha\f$
    Returns:
        The plastic spin tensor \f$\mathbf{W}^p\f$
    """
    W_p = zeros((3,3))
    PlasticSpinTensorInnerUpdate(W_p, mprops.n_alpha, gammadot_alphas, m_alphas, n_alphas)
    return W_p

def GammadotAbsSum(gammadot_alphas):
    """Calculate the sum of absolute values of \f$\dot{\gamma}^\alpha\f$
    
    This function is trivial, but it establishes a name for a quantitiy
    that is used frequently. This feeds into the calculation in 
    Equation 18 of Mihaila2014.
    
    Args:
        gammdot_alphas: \f$\dot{\gamma}^\alpha\f$
    Returns:
        The sum of the absolute values.
    """
    return sum(abs(gammadot_alphas)) 

def simpleShearDeformationGradient(t,shear_rate):
    """The deformation gradient for linear shear in the x-y plane.
    
    Defined in section 6.1. of Kalidindi1992. The shear is given by
    \f[
    \mathbf{x} = \mathbf{X} + (\dot{\gamma} t) X_2 \mathbf{e}_1,
    \f]
    which gives a deformation gradient of
    \f{equation}{
    \mathbf{F} = 
    \begin{bmatrix}
    1 & \dot{\gamma} t & 0\\
    0 & 1 & 0\\
    0 & 0 & 1 
    \end{bmatrix}.
    \f}
    
    Args:
        t: the time in seconds
        shear_rate: \f$\dot{\gamma}\f$
    Returns:
        The deformation gradient \f$\mathbf{F}\f$
    """
    F = eye(3)
    F[0,1] = shear_rate*t
    return F
    
def dumpSolverState(polycrystal, F_next, dt, filename):
    """Dump the state of the solver.
    
    Intended to be used for debugging purposes. This allows one
    to run the state through
    """
    f = open(filename, 'w')
    f.write("dt = "+str(dt)+"\n")
    f.write("F_next = "+F_next.__repr__()+"\n")
    f.write("polycrystal = "+polycrystal.__repr__()+"\n")   
    
def replayStepFromDump(filename):
    """Replay the next step based on the dump from the solver.
    """
    # Read in state
    exec(open(filename,"r").read())

    # Replay step
    step_good, new_dt = polycrystal.step(F_next, dt)
    
def replicate():
    # MPI
    np = 8

    # Initial conditions
    T_0 = zeros((3,3))
    s_0 = 16.0e-3 # (Gpa)
    F_p_0 = eye(3)

    # Material parameters #
    #######################
    n_crystals = 1000

    # End time (seconds)
    t_end = 823.53

    # Main loop #
    #############

    # Important stored values
    times = []
    shears = []
    T_cauchys = []

    # Initial time quantities
    t = 0
    dt = 1e-3
    t_end_tol = 1e-10
    prevent_next_x_timestep_increases = 0

    # Loop iterations
    it_count = 0
    print_interval = 1

    if do_py_run:
        # Construct polycrystal
        initial_crystals = []
        for i_crystal in range(n_crystals):
            # Random crystal orientation
            mprops = kalidindi1992.getDefaultCrystalMaterialProperties()
            mprops = kalidindi1992.randomizeCrystalOrientation(mprops)

            # Initial crystal
            crystal = kalidindi1992.Crystal(mprops, T_0, s_0, F_p_0)
            initial_crystals.append(crystal)
        polycrystal = kalidindi1992.Polycrystal(initial_crystals)
        
        while t<t_end and abs(t-t_end)/t_end > t_end_tol:
            
            # Wall clock timing
            start_time = time.time()
            
            # Timestep banner
            if it_count % print_interval == 0:
                print "##########t=%1.2g start##########" % (t)

            # If timestep was decreased earlier, prevent subsequent increases
            if prevent_next_x_timestep_increases > 0:
                prevent_next_x_timestep_increases -= 1

            # Do the main solve
            step_good = False
            while not step_good:
                print "Current dt = %1.3g"%(dt)
                
                F_next = F_of_t(t + dt)
                
                # Try step
                step_good, new_dt = polycrystal.step(F_next, dt)
                print "new_dt/dt = %1.3g"%(new_dt/dt)
                
                # If the step is not accepted, reduce the timestep
                if not step_good:
                    # Use the adjusted timestep now
                    dt = new_dt
                    
                    # "Further, in order to improve on the efficiency of 
                    # the scheme, the next 10 time steps are taken with this time 
                    # step or with a smaller value as governed by the basic algorithm 
                    # discussed above." in Kalidindi1992
                    prevent_next_x_timestep_increases = 10                
            
            # Update the current time
            t += dt
                
            # Set the adjusted timestep for the next step
            if prevent_next_x_timestep_increases == 0:
                dt = new_dt    
            
            # Wall clock timing
            elapsed_time = time.time()-start_time
            approximate_remaining = (t_end-t)*elapsed_time/dt    
            #~ if it_count % print_interval == 0:
                #~ print "%1.2e seconds remaining"%(approximate_remaining)    
            it_count += 1
            
            # Store important quantities
            times.append(t)
            shears.append(shear_rate*t)
            T_cauchys.append(polycrystal._T_cauchy)

            # If we're near the end, adjust the timestep to exactly hit the end time 
            if t+dt > t_end:
                dt = t_end - t
