""" Tests for the kalidindi1992 library
"""

import unittest
from numpy.random import rand
from numpy import divide, ones, eye
from numpy.linalg import norm

import sys
sys.path.append("..")
from kalidindi1992 import *

class TestDirectExpressions(unittest.TestCase):
    """Tests of functions that evaluate a single expression.
    
    Expressions that are a single line or involve only simple arithmetic 
    are not covered.
    """
    
    def partialDgammaPartialTCentredDiffTheoreticalError(self,m,tau,h,nterms=1):
        """Leading order error of the centred difference approximation to kalidindi1992.PartialDGammaPartialT.
        """
        prefactor = 1.0
        rel_err = 0.0
        for i in range(nterms):
            for j in [i*2+1,i*2+2]:
                prefactor *= (1-j*m)
            order = 2*i+2
            rel_err += (prefactor/(m**order))*(abs(tau)**(-order))*(h**order)
        return rel_err
    
    def test_PartialDGammaPartialT(self):
        """Test of kalidindi1992.PartialDGammaPartialT
        
        This test compares the expected theoretical error in the numerical
        derivative withthe  actual error compared with the analytic 
        expression in kalidindi1992.PartialDGammaPartialT. If the actual
        error is less than the theoretical leading order, the test
        passes.
        """
        # Material properties
        m = 0.012
        S_0 = [1.1*ones((3,3))/9.0]
        mprops = MaterialProperties(crystal_type='None',
										mu=rand(),
										kappa=rand(),
										m=m,
										gammadot_0=rand(),
										h_0=rand(),
										s_s=rand(),
										a=rand(),
										q=rand(),
										n_alpha=1,
										m_0=None,
										n_0=None,
                                        S_0=S_0,
                                        L=rand(3,3,3,3)
                                        )
        
        # Timestep
        dt = 1e-3
        
        # Arbitrary tolerance
        tol = 1.0
        
        # Specific
        T = ones((3,3))
        s_alphas = [1.0]
        
        # Analytic derivative
        partial_Dgamma_partial_T_analytic = PartialDGammaPartialT(mprops.m, mprops.gammadot_0, T, mprops.S_0[0,:,:], s_alphas[0], dt)
        
        # Relative errors
        relErrs = []
        relErrsOverTheoreticalRelErrs = []
        for h in [10**(-i) for i in range(1,15)]:            
            # Approximate derivative
            partial_Dgamma_partial_T_approx = zeros((3,3))
            theoretical_relErr = zeros((3,3))
            for i in range(3):
                for j in range(3):
                    # Centered difference approximation
                    T_plus_h = T.copy()
                    T_minus_h = T.copy()
                    T_plus_h[i,j] += h
                    T_minus_h[i,j] -= h
                    partial_Dgamma_partial_T_approx[i,j] = (ShearStrainIncrements(mprops, T_plus_h, s_alphas, dt)[0]
                                                      -ShearStrainIncrements(mprops, T_minus_h, s_alphas, dt)[0])/(2*h)
                    
                    # Theoretical leading relative error term
                    tau = tensorInner2_2(T, mprops.S_0[0,:,:])
                    theoretical_relErr[i,j] = self.partialDgammaPartialTCentredDiffTheoreticalError(m,tau,h,nterms=3)
                           
                                                      
            # Errors
            relErr = divide(partial_Dgamma_partial_T_approx - partial_Dgamma_partial_T_analytic, partial_Dgamma_partial_T_analytic)
            relErrs.append(relErr)
            relErrsOverTheoreticalRelErrs.append(max(abs(relErr))/max(abs(theoretical_relErr)))
        
        # Find the point at which the relative errors stop decreasing
        minErr = 1e16
        minRelErrOverTheoreticalRelErr = 1e16
        for i in range(len(relErrs)):
            maxAbsRelErr = max(abs(relErrs[i]))
            if maxAbsRelErr < minErr:
                minErr = maxAbsRelErr
                minRelErrOverTheoreticalRelErr = relErrsOverTheoreticalRelErrs[i]
        
        # Check that the numerical relative error at this point is better than the theoretical bound
        self.assertTrue(minRelErrOverTheoreticalRelErr < tol)

class TestIterativeSchemes(unittest.TestCase):
    """Tests of functions responsible for the iterative algebraic solvers.
    """
    
    def test_updateT(self):
        """Test of kalidindi1992.updateT.
        
        The algebraic system that the iterative algebraic solver is supposed
        to solve is
        \f[
            \mathbf{T}^*(t_{i+1}) = \mathbf{T}^{*tr}(t_{i+1}) - 
            \sum_\alpha \Delta \gamma^{\alpha} (\mathbf{T}^*(t_{i+1}), s^{\alpha}(t_{i+1}))
            \mathbf{C}^\alpha.
        \f]        
        """
        # Material properties
        mprops = getDefaultMaterialProperties()
        mprops.m = 0.1
        
        # Error tolerance for test
        test_tol = 1e-10
                                        
        # Random configuration
        dt = 1.0
        T_init = ones((3,3))
        s_alphas = ones((12))
        
        # Simple deformation
        t = 1.0
        shear_rate = 1.0
        F = simpleShearDeformationGradient(t, shear_rate)
        F_p = eye(3)
        tensor_A = TensorA(F, F_p)
        
        # Do scipy solve
        step_accepted, T_scipy = solveForT_Scipy(mprops, tensor_A, T_init, s_alphas, dt)
        
        # Do the Kalidindi solve
        DT_max = 1.0
        DT_tol = 1e-16
        max_iter = 200
        step_accepted, T_kalidindi = updateT(mprops, tensor_A, T_init, s_alphas, dt, DT_max, DT_tol, max_iter)
        
        # Assert agreement of the two
        relErr = norm(T_scipy-T_kalidindi)/norm(T_scipy)
        self.assertTrue(relErr < test_tol)
    
    def test_updateS(self):
        """Test of kalidindi1992.updateS.
        
        The algebraic system that the iterative algebraic solver is supposed
        to solve is
        \f[
            s^\alpha(t_{i+1}) = s^\alpha(t_{i}) + \sum_\beta h^{\alpha\beta} (s^\beta(t_{i+1}))
            \left|\Delta \gamma^\beta(\mathbf{T}^* (t_{i+1}), s^\beta(t_{i+1}))\right|
        \f]        
        """
        # Material properties
        mprops = getDefaultMaterialProperties()
        mprops.m = 0.1
        
        # Error tolerance for test
        test_tol = 1e-10
                                        
        # Random configuration
        dt = 1.0
        T = ones((3,3))*16e-3
        s_alphas_init = ones((12))*16e-3
        s_alphas_prev = s_alphas_init.copy()
        
        # Do scipy solve
        step_accepted, s_scipy = solveForS_Scipy(mprops, T, s_alphas_prev, s_alphas_init, dt)
        
        # Do the Kalidindi solve
        Ds_tol = 1e-16
        max_iter = 200
        step_accepted, s_kalidindi = updateS(mprops, T, s_alphas_prev, s_alphas_init, dt, Ds_tol, max_iter)
        
        # Assert agreement of the two
        relErr = norm(s_scipy-s_kalidindi)/norm(s_scipy)
        self.assertTrue(relErr < test_tol)
if __name__ == '__main__':
    unittest.main()
