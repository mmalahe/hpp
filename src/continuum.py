from linearAlgebra import *
from numpy import dot, cos, sin
import numpy as np
from scipy.linalg import sqrtm

def deviatoricStressTensor(T):
    """The deviatoric component of the stress tensor T.
    
    This feeds into the calculation in Equation 16 of Mihaila2014.
    
    Args:
        T: the stress tensor
    Returns:
        The deviatoric component of the stress tensor.
    """
    return deviatoricComponent3(T)

def deformationGradientTimeDerivative(L, F):
    """Calculate the time derivative of the deformation gradient.
    
    Computed from the velocity gradient and current deformation gradient:
    \f[
    \dot{\mathbf{F}} = \mathbf{L} \mathbf{F}
    \f]
    
    Args:
        L: \f$\mathbf{L}\f$
        F: \f$\mathbf{F}\f$
    Returns:
        \f$\dot{\mathbf{F}}\f$
    """
    F_dot = L.dot(F)
    return F_dot

def updateDeformationGradientFromVelocityGradient(L, F_init, dt):
    """Get the deformation gradient after a small incremental update.
    
    The initial configuration is no deformation. This then evaluates the 
    approximation
    \f{eqnarray}
    \mathbf{F}(t+\Delta t) & \approx & \mathbf{F}(t) + \dot{\mathbf{F}}(t) \Delta t \\
    \mathbf{F}(t+\Delta t) & = & \mathbf{F}(t) + \mathbf{L}(t) \mathbf{F}(t) \Delta t.
    \f}
    
    \f$ \Delta t \f$ defaults to 1e-10.
    
    Args:
        L: \f$\mathbf{L}(t)\f$
        F_init: \f$\mathbf{F}(t)\f$
        dt: \f$\Delta t\f$
    Returns:
        \f$\mathbf{F}(t+\Delta t)\f$
    """
    F_dot = deformationGradientTimeDerivative(L, F_init)
    F_next = F_init + F_dot*dt
    return F_next
    
def polarDecomposition(F):
    """Compute the polar decomposition of a tensor.
    
    The polar decomposition is
    \f[
    \mathbf{F} = \mathbf{R} \mathbf{U},
    \f]
    where \f$\mathbf{R}\f$ is an orthogonal rotation tensor, and
    \f$\mathbf{U}\f$ is the s.p.d. right stretching tensor. The computation
    is
    \f{eqnarray}{
    & \mathbf{U}^2 = \mathbf{F}^T \mathbf{F}\\
    \Rightarrow & \mathbf{U} = \sqrt{\mathbf{F}^T \mathbf{F}},
    \f}
    followed by
    \f[
    \mathbf{R} = \mathbf{F} \mathbf{U}^{-1}
    """
    U = sqrtm((F.T).dot(F))
    R = F.dot(inv(U))
    return R, U

def EulerZXZRotationMatrix(alpha, beta, gamma):
    """Returns the active, intrinsic, right-handed EulerZXZ rotation matrix.
    
    The matrix is defined by
    \f{equation}
    R = Z_1 X_2 Z_3 =
    \begin{bmatrix}
    c_1 c_3 - c_2 s_1 s_3 & -c_1 s_3 - c_2 c_3 s_1 & s_1 s_2 \\
    c_3 s_1 + c_2 c_2 s_3 & c_1 c_2 c_3 - s_1 s_3 & -c_1 s_2 \\
    s_2 s_3 & c_3 s_2 & c2 
    \end{bmatrix},
    \f}
    where
    \f{eqnarray}
    c_1 & = & \cos(\alpha) \\
    c_2 & = & \cos(\beta) \\
    c_3 & = & \cos(\gamma) \\
    s_1 & = & \sin(\alpha) \\
    s_2 & = & \sin(\beta) \\
    s_3 & = & \sin(\gamma),
    \f}
    where \f$\alpha\f$, \f$\beta\f$, \f$\gamma\f$ are the proper Euler
    angles.
    
    Args:
        alpha: \f$\alpha\f$
        beta: \f$\beta\f$
        gamma: \f$\gamma\f$
    Returns:
        \f$R\f$
    """
    c1 = cos(alpha)
    c2 = cos(beta)
    c3 = cos(gamma)
    s1 = sin(alpha)
    s2 = sin(beta)
    s3 = sin(gamma)
    R = zeros((3,3))
    R[0,0] = c1*c3 - c2*s1*s3
    R[0,1] = -c1*s3 - c2*c3*s1
    R[0,2] = s1*s2
    R[1,0] = c3*s1 + c1*c2*s3
    R[1,1] = c1*c2*c3 - s1*s3
    R[1,2] = -c1*s2
    R[2,0] = s2*s3
    R[2,1] = c3*s2
    R[2,2] = c2
    return R

def BungeEulerRotationMatrix(phi_1, PHI, phi_2):
    """Bunge-Euler rotation matrix.
    
    This is the passive version of the Euler
    ZXZ matrix. The active version is obtained using continuum.EulerZXZRotationMatrix,
    and transposed to get the passive version.
    
    Args:
        phi_1: \f$\phi_1\f$
        PHI: \f$\Phi\f$
        phi_2: \f$\phi_2\f$
    Returns:
        The passive rotation matrix \f$R^T\f$
    """
    R = EulerZXZRotationMatrix(phi_1, PHI, phi_2)
    return R.T
    

