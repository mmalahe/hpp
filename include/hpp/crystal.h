/** @file crystal.h
* @author Michael Malahe
* @brief Header file for crystal classes
* @details
*
* Kalidindi1992 refers to the paper Surya R. Kalidindi, Curt A. Bronkhorst
and Lallit Anand. Crystallographic texture evolution in bulk deformation
processing of FCC metals. Journal of the Mechanics and Physics of Solids
, 40(3):537--569, 1992.
*
* We also make reference to Mihaila2014, which is the paper Bogdan Mihaila,
Marko Knezevic and Andres Cardenas. Three orders of magnitude improved
efficiency with high-performance spectral crystal plasticity on GPU
platforms., (January):785--798, 2014
*/

#ifndef HPP_CRYSTAL_H
#define HPP_CRYSTAL_H

#include <vector>
#include <functional>
#include <complex>
#include <stdlib.h>

#include "mpi.h"
#include <omp.h>

#include <hpp/config.h>
#include <hpp/tensor.h>
#include <hpp/continuum.h>
#include <hpp/mpiUtils.h>
#include <hpp/spectralUtils.h>
#include <hpp/profUtils.h>

namespace hpp
{

enum CrystalType {
    CRYSTAL_TYPE_FCC,
    CRYSTAL_TYPE_NONE
};    

enum CrystalDatasetIdx {   
    // Symmetric sigma in Voigt order
    // It is the deviatoric component, so sigma_22 = -sigma_11 - sigma_00
    // Therefore sigma_22 is excluded
    SIGMA00,
    SIGMA11,
    SIGMA12,
    SIGMA02,
    SIGMA01,
    // Anti-symmetric Wp
    WP01,
    WP12,
    WP02,
    // Gamma
    GAMMA
};

std::vector<SpectralDatasetID> defaultCrystalSpectralDatasetIDs();

class CrystalError: public std::runtime_error
{
public:
    explicit CrystalError (const std::string &val) : std::runtime_error::runtime_error(val) {}
};


// Forward declarations are necessarry for some operations
template <typename U>
class Crystal;
template <typename U>
class Polycrystal;

/*## A recordclass for the material properties of a crystal
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
## - **L**: the elasticity tensor \f$\mathcal{L}\f$ */

/**
 * @class CrystalProperties
 * @author Michael Malahe
 * @date 14/10/16
 * @file crystal.h
 * @brief
 */
template <typename U>
struct CrystalProperties {
    CrystalType crystal_type;
    unsigned int n_alpha;
    U mu;
    U kappa;
    U c11, c12, c44;
    U m;
    U gammadot_0;
    U h_0;
    U s_s;
    U a;
    U q;
    U volume = 1.0;
    std::vector<std::vector<U>> m_0;
    std::vector<std::vector<U>> n_0;
    std::vector<hpp::Tensor2<U>> S_0;
    hpp::Tensor4<U> L;
    hpp::Tensor2<U> Q;
};
template <typename U>
CrystalProperties<U> defaultCrystalProperties()
{
    CrystalProperties<U> props;

    // Crystal type
    props.crystal_type = CRYSTAL_TYPE_FCC;

    // Scalar parameters
    props.n_alpha = 12;
    
    // Elastic constants from Kalidindi1992
    props.mu = 46.5*GPA;
    props.kappa = 124.0*GPA;
    
    // Elastic constants from Anders and Thompson 1961
    // See Ledbetter and Naimon 1974, page 908 for the values actually tabulated
    props.c11 = 168.7*GPA;
    props.c12 = 121.7*GPA;
    props.c44 = 75.0*GPA;
    
    // From Kalidindi1992
    props.m = 0.012;
    props.gammadot_0 = 0.001;
    props.h_0 = 180.0*MPA;
    props.s_s = 148.0*MPA;
    props.a = 2.25;
    props.q = 1.4;

    // Tensor Q
    hpp::Tensor2<U> Q(12,12);
    for (unsigned int i=0; i<4; i++) {
        for (unsigned int j=0; j<4; j++) {
            U val;
            if (i==j) {
                val = props.q;
            } else {
                val = 1.0;
            }
            for (unsigned int k=0; k<3; k++) {
                for (unsigned int l=0; l<3; l++) {
                    Q(3*i+k,3*j+l) = val;
                }
            }
        }
    }
    props.Q = Q;

    // Slip directions
    std::vector<std::vector<U>> m_0(props.n_alpha);
    m_0[0] = {1, -1, 0};
    m_0[1] = {-1,0,1};
    m_0[2] = {0,1,-1};
    m_0[3] = {1,0,1};
    m_0[4] = {-1,-1,0};
    m_0[5] = {0,1,-1};
    m_0[6] = {-1,0,1};
    m_0[7] = {0,-1,-1};
    m_0[8] = {1,1,0};
    m_0[9] = {-1,1,0};
    m_0[10] = {1,0,1};
    m_0[11] = {0,-1,-1};
    m_0 = m_0/std::sqrt((U)2.0);
    props.m_0 = m_0;

    // Slip plane normals
    std::vector<std::vector<U>> n_0(props.n_alpha);
    for (int i=0; i<3; i++) {
        n_0[i] = {1,1,1};
    }
    for (int i=3; i<6; i++) {
        n_0[i] = {-1,1,1};
    }
    for (int i=6; i<9; i++) {
        n_0[i] = {1,-1,1};
    }
    for (int i=9; i<12; i++) {
        n_0[i] = {-1,-1,1};
    }
    n_0 = n_0/std::sqrt((U)3.0);
    props.n_0 = n_0;

    // Slip systems
    std::vector<hpp::Tensor2<U>> S_0(props.n_alpha);
    for (unsigned int i=0; i<props.n_alpha; i++) {
        S_0[i] = hpp::outer(m_0[i],n_0[i]);
    }
    props.S_0 = S_0;

    // Elasticity tensor
    props.L = cubeSymmetricElasticityTensor(props.c11, props.c12, props.c44);

    // Return
    return props;
};

template <typename U>
CrystalProperties<U> rotate(const CrystalProperties<U>& propsOld, hpp::Tensor2<U> rotTensor)
{
    CrystalProperties<U> propsNew = propsOld;
    for (unsigned int i=0; i<propsOld.n_alpha; i++) {
        propsNew.m_0[i] = rotTensor*propsOld.m_0[i];
        propsNew.n_0[i] = rotTensor*propsOld.n_0[i];
        propsNew.S_0[i] = hpp::outer(propsNew.m_0[i], propsNew.n_0[i]);
    }
    return propsNew;
}

template <typename U>
struct CrystalSolverConfig {
    /** @brief Timestep control by shear rate
    Defined on journal page 546 of Kalidindi1992*/
    U Dgamma_max;
    /** @brief Defined on journal page 546 of Kalidindi1992*/
    U r_min;
    /** @brief Defined on journal page 546 of Kalidindi1992*/
    U r_max;
    /** @brief Timestep control by algebraic system convergence
    Defined on journal page 546 of Kalidindi1992*/
    unsigned int algebraic_max_iter;
    /** @brief Defined on journal page 547 of Kalidindi1992*/
    U r_t;

    // Convergence tolerances for the two-level scheme
    // Defined on journal page 545 of Kalidindi1992
    U DT_tol_factor = 1e-4;
    // Defined on journal page 545 of Kalidindi1992
    U Ds_tol_factor = 1e-3;
    // Constraint on the Newton corrections for T
    // Defined on journal page 545 of Kalidindi1992
    U DT_max_factor = (2.0/3.0);

    // Verbosity
    bool verbose = true;
};
template <typename U>
CrystalSolverConfig<U> defaultCrystalSolverConfig()
{
    CrystalSolverConfig<U> config;
    config.Dgamma_max = 1e-2;
    config.r_min = 0.8;
    config.r_max = 1.25;
    config.algebraic_max_iter = 50;
    config.r_t = 0.75;
    config.verbose = true;
    return config;
};
template <typename U>
CrystalSolverConfig<U> defaultConservativeCrystalSolverConfig()
{
    CrystalSolverConfig<U> config;
    config.Dgamma_max = 2e-3;
    config.r_min = 0.8;
    config.r_max = 1.25;
    config.algebraic_max_iter = 50;
    config.r_t = 0.75;
    config.verbose = true;
    return config;
};

template <typename U>
struct CrystalInitialConditions {
    hpp::Tensor2<U> T_init;
    U s_0;
    hpp::Tensor2<U> F_p_0;
    hpp::Tensor2<U> crystalRotation;
};
template <typename U>
CrystalInitialConditions<U> defaultCrystalInitialConditions()
{
    CrystalInitialConditions<U> init;
    init.T_init = hpp::Tensor2<U>(3,3);
    init.s_0 = 16.0*MPA;
    init.F_p_0 = hpp::identityTensor2<U>(3);
    init.crystalRotation = hpp::identityTensor2<U>(3);
    return init;
};

/**
 * @brief Get the corresponding Euler angles for the rotation induced by this deformation.
 * @param F_star
 * @return
 */
template <typename T>
EulerAngles<T> getEulerAnglesFromDeformationGradient(const hpp::Tensor2<T>& F_star)
{
    PolarDecomposition<T> decomp = F_star.polarDecomposition();
    EulerAngles<T> EAngles = hpp::getEulerZXZAngles(decomp.R);
    return EAngles;
}

template <typename U>
class Crystal
{
    friend class Polycrystal<U>;

public:
    // Constructor
    Crystal();
    Crystal(const CrystalProperties<U>& props, const CrystalSolverConfig<U>& config,
            const CrystalInitialConditions<U>& init);

    // Stepping
    bool tryStep(const hpp::Tensor2<U>& F_next, U dt);
    void acceptStep();
    void rejectStep();
    U recommendNextTimestepSize(U dt);

    // Getters
    std::vector<std::vector<U>> getM_alphas() const;
    std::vector<std::vector<U>> getN_alphas() const;
    Tensor2<U> getTCauchy() const {
        return (F_e*T)*(F_e.trans())/F_e.det();
    }
    U getVolume() const {
        return props.volume;
    }
    const std::vector<U>& getSAlphas() const {
        return s_alphas;
    }
    EulerAngles<U> getEulerAngles() const; 

    // Getting derived properties
    std::vector<U> getShearStrainRates();
    Tensor2<U> getPlasticSpinTensor();

protected:
    // Initializing
    void applyInitialConditions();

private:
    // PROBLEM AND SOLVER SETTINGS //
    CrystalProperties<U> props;
    CrystalSolverConfig<U> config;
    CrystalInitialConditions<U> init;

    // Solver tolerances and constraintsbased on initial conditions
    /** @brief Defined on journal page 545 of Kalidindi1992 */
    U DT_max;
    /** @brief Defined on journal page 545 of Kalidindi1992 */
    U DT_tol;
    /** @brief Defined on journal page 545 of Kalidindi1992 */
    U Ds_tol;

    // CURRENT STATE //
    hpp::Tensor2<U> T;
    std::vector<U> s_alphas;
    hpp::Tensor2<U> F_p;
    bool step_accepted = false;
    bool step_rejected = false;

    // Derived quantities
    hpp::Tensor2<U> F_e;
    std::vector<U> Dgamma_alphas;

    // Stepping
    bool updateT(const hpp::Tensor2<U>& A, U dt);
    bool updateS(U dt);
    bool updateTandS(const hpp::Tensor2<U>& A, U dt);
    void assertAcceptedOrRejectedStep();

    // Slip systems
    std::vector<std::vector<U>> m_alphas;
    std::vector<std::vector<U>> n_alphas;

    // STATE AT NEXT STEP //
    hpp::Tensor2<U> T_next;
    std::vector<U> s_alphas_next;
    hpp::Tensor2<U> F_p_next;
    std::vector<U> Dgamma_alphas_next;
    hpp::Tensor2<U> F_e_next;

    // DUMMY VARIABLES
    std::vector<U> dumDgamma_alphas;
    std::vector<Tensor2<U>> dum2ndOrders;
    std::vector<Tensor4<U>> dum4thOrders;
    std::vector<hpp::Tensor2<U>> dumC_alphas;
};

template <typename U>
class Polycrystal
{
public:
    Polycrystal(const std::vector<Crystal<U>>& crystal_list, MPI_Comm comm);
    bool step(hpp::Tensor2<U> F_next, U dt);
    U recommendNextTimestepSize(U dt);
    void evolve(U t_start, U t_end, U dt_initial, std::function<hpp::Tensor2<U>(U t)> F_of_t);

    // Get crystal
    Crystal<U> getCrystal(int i) {
        return crystal_list.at(i);
    }

    // Write
    void writeResultHDF5(std::string filename);
    
    // Getters
    const std::vector<U>& getTHistory() const {return t_history;}
    
protected:

private:
    std::vector<Crystal<U>> crystal_list;

    // Derived quantities
    void updateDerivedQuantities();
    hpp::Tensor2<U> T_cauchy;

    // Initializing
    void applyInitialConditions();

    // History
    std::vector<U> t_history;
    std::vector<hpp::Tensor2<U>> T_cauchy_history;

    // MPI
    MPI_Comm comm;
    int comm_size;
    int comm_rank;
};


/////////////////////
// SPECTRAL SOLVER //
/////////////////////

template <typename U>
struct SpectralCrystalSolverConfig {
    unsigned int nTerms;
};

template <typename U>
class SpectralCrystal
{
    //friend class SpectralPolycrystal<U>;

public:
    // Constructor
    SpectralCrystal();
    SpectralCrystal(const CrystalProperties<U>& props, const SpectralCrystalSolverConfig<U>& config,
                    const CrystalInitialConditions<U>& init);

    // Stepping
    void step(const hpp::Tensor2<U>& F_next, const hpp::Tensor2<U>& L_next, const hpp::SpectralDatabase<U>& db, U dt);

    // Getting quantities
    Tensor2<U> getTCauchy() const {
        return TCauchy;
    }
    U getVolume() const {
        return props.volume;
    }

private:
    // CURRENT STATE //
    Tensor2<U> RStar;
    U s;
    Tensor2<U> TCauchy;

    // Initial conditions
    CrystalInitialConditions<U> init;

    // Material properties
    CrystalProperties<U> props;

    // Solver configuration
    SpectralCrystalSolverConfig<U> config;
};

template <typename U>
class SpectralPolycrystal
{
public:
    SpectralPolycrystal(const std::vector<SpectralCrystal<U>>& crystal_list, unsigned int nOmpThreads);
    void step(const hpp::Tensor2<U>& F_next, const hpp::Tensor2<U>& L_next, const hpp::SpectralDatabase<U>& db, U dt);
    void evolve(U t_start, U t_end, U dt, std::function<hpp::Tensor2<U>(U t)> F_of_t, std::function<hpp::Tensor2<U>(U t)> L_of_t, const hpp::SpectralDatabase<U>& db);

    // Get crystal
    SpectralCrystal<U> getCrystal(int i) {
        return crystal_list.at(i);
    }

    // Gather values
    Tensor2<U> getGlobalTCauchy();

    // Write
    void writeResultNumpy(std::string filename);
protected:

private:
    // List of crystal
    std::vector<SpectralCrystal<U>> crystal_list;

    // History
    std::vector<U> t_history;
    std::vector<hpp::Tensor2<U>> T_cauchy_history;

    // OpenMP configuration
    unsigned int nOmpThreads = 1;

    // Timing
    hpp::Timer solveTimer;
};

}//END NAMESPACE HPP


#endif /* HPP_CRYSTAL_H */
