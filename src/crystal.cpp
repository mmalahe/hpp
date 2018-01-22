#include <hpp/tensor.h>
#include <hpp/crystal.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <math.h>

namespace hpp {   

std::vector<SpectralDatasetID> defaultCrystalSpectralDatasetIDs() {
    std::vector<SpectralDatasetID> dsetIDs(9);
    std::vector<unsigned int> c00 = {0,0};
    std::vector<unsigned int> c11 = {1,1};
    std::vector<unsigned int> c12 = {1,2};
    std::vector<unsigned int> c02 = {0,2};
    std::vector<unsigned int> c01 = {0,1};
    dsetIDs[SIGMA00] = SpectralDatasetID("sigma_prime",c00);
    dsetIDs[SIGMA11] = SpectralDatasetID("sigma_prime",c11);
    dsetIDs[SIGMA12] = SpectralDatasetID("sigma_prime",c12);
    dsetIDs[SIGMA02] = SpectralDatasetID("sigma_prime",c02);
    dsetIDs[SIGMA01] = SpectralDatasetID("sigma_prime",c01);
    dsetIDs[WP01] = SpectralDatasetID("W_p",c01);
    dsetIDs[WP12] = SpectralDatasetID("W_p",c12);
    dsetIDs[WP02] = SpectralDatasetID("W_p",c02);
    dsetIDs[GAMMA] = SpectralDatasetID("gammadot_abs_sum");
    return dsetIDs;
}
    
/**
 * @brief See kalidindi1992.TensorA
 * @param F_p \f$\mathbf{F}^{p}(t_{i})\f$
 * @param F_next \f$\mathbf{F}(t_{i+1})\f$
 * @tparam U the scalar type
 * @return \f$\mathbf{A}\f$
 */
template <typename U>
hpp::Tensor2<U> tensorA(hpp::Tensor2<U> F_p, hpp::Tensor2<U> F_next) 
{
    hpp::Tensor2<U> F_p_inv = F_p.inv();
    hpp::Tensor2<U> tensor_A = (F_p_inv.trans())*(F_next.trans())*F_next*F_p_inv;
    return tensor_A;
}

/**
 * @brief See kalidindi1992.TensorC_alphas 
 * @param L \f$\mathcal{L}\f$
 * @param A \f$\mathbf{A}\f$
 * @param S_0 List of \f$\mathbf{S}_0^\alpha\f$
 * @param n_alpha the number of slip systems
 * @tparam U the scalar type
 * @return \f$\mathbf{C}^\alpha\f$
 */
template <typename U>
std::vector<hpp::Tensor2<U>> tensorC_alphas(const hpp::Tensor4<U>& L, const hpp::Tensor2<U>& A, 
const std::vector<hpp::Tensor2<U>>& S_0, const unsigned int n_alpha) 
{
    std::vector<hpp::Tensor2<U>> C_alphas(n_alpha);
    for (unsigned int i=0; i<n_alpha; i++) {
        hpp::Tensor2<U> B_alpha = A*S_0[i] + (S_0[i].trans())*A;
        C_alphas[i] = hpp::contract<U>(L, (U)0.5*B_alpha);
    }
    return C_alphas;
}

/**
 * @brief In-place version of hpp.tensorC_alphas
 * @details The input C_alphas do not need to be zeroed
 * @param L \f$\mathcal{L}\f$
 * @param A \f$\mathbf{A}\f$
 * @param S_0 List of \f$\mathbf{S}_0^\alpha\f$ for each \f$\alpha\f$
 * @param n_alpha the number of slip systems
 * @param C_alphas list of \f$\mathbf{C}^\alpha\f$ for each \f$\alpha\f$ to be returned
 * @tparam U the scalar type
 */
template <typename U>
void tensorC_alphasInPlace(const hpp::Tensor4<U>& L, const hpp::Tensor2<U>& A, 
const std::vector<hpp::Tensor2<U>>& S_0, const unsigned int n_alpha, hpp::Tensor2<U>& dumB_alpha, std::vector<hpp::Tensor2<U>>& C_alphas) 
{
    for (unsigned int i=0; i<n_alpha; i++) {
        hpp::ABPlusBTransposeAInPlace(A, S_0[i], dumB_alpha);
        dumB_alpha *= (U)0.5;
        hpp::contractInPlace(L, dumB_alpha, C_alphas[i]);
    }
}

/**
 * @brief See kalidindi1992.PlasticShearingRate
 * @param tau_alpha \f$\tau^\alpha\f$
 * @param s_alpha \f$s^\alpha\f$
 * @param gammadot_0 \f$\dot{\gamma}_0\f$
 * @param m \f$m\f$
 * @return \f$\dot{\gamma}^\alpha\f$
 */
template <typename U> 
U plasticShearingRate(U tau_alpha, U s_alpha, U gammadot_0, U m) 
{
    U t_over_s_abs = std::pow(std::abs(tau_alpha/s_alpha),(1.0/m));
    U t_over_s_signed = std::copysign(t_over_s_abs, tau_alpha);
    U gammadot_alpha = gammadot_0*t_over_s_signed;
    return gammadot_alpha;
}

/**
 * @brief See kalidindi1992.ShearStrainRates
 * @param props the properties of the material defined in hpp.CrystalProperties
 * @param T \f$\mathbf{T}^*\f$
 * @param s_alphas a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
 * @return An array of \f$\dot{\gamma}^\alpha\f$ for each slip system \f$\alpha\f$
 */
template <typename U>
std::vector<U> shearStrainRates(const CrystalProperties<U>& props, const hpp::Tensor2<U>& T, const std::vector<U>& s_alphas) 
{
    std::vector<U> gammadot_alphas(props.n_alpha);
    for (unsigned int i=0; i<props.n_alpha; i++) {
        U tau_alpha = hpp::contract<U>(T, props.S_0[i]);
        U gammadot_alpha = plasticShearingRate(tau_alpha, s_alphas[i], props.gammadot_0, props.m);
        gammadot_alphas[i] = gammadot_alpha;
    }
    return gammadot_alphas;
}


/**
 * @brief In-place version of hpp.shearStrainRates
 * @param props the properties of the material defined in hpp.CrystalProperties
 * @param T \f$\mathbf{T}^*\f$
 * @param s_alphas a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
 * @param gammadot_alphas An array of \f$\dot{\gamma}^\alpha\f$ for each slip system \f$\alpha\f$ that is returned
 */
template <typename U>
inline void shearStrainRatesInPlace(const CrystalProperties<U>& props, const hpp::Tensor2<U>& T, const std::vector<U>& s_alphas, std::vector<U>& gammadot_alphas) 
{
    for (unsigned int i=0; i<props.n_alpha; i++) {
        U tau_alpha = hpp::contract<U>(T, props.S_0[i]);
        U gammadot_alpha = plasticShearingRate(tau_alpha, s_alphas[i], props.gammadot_0, props.m);
        gammadot_alphas[i] = gammadot_alpha;
    }
}

/**
 * @brief See kalidindi1992.ShearStrainIncrements
 * @param props the properties of the material defined in hpp.CrystalProperties
 * @param T \f$\mathbf{T}^*\f$
 * @param s_alphas a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
 * @param dt \f$\Delta t\f$
 * @return A list of \f$\Delta \gamma^\alpha\f$ for each \f$\alpha\f$
 */
template <typename U>
inline std::vector<U> shearStrainIncrements(const CrystalProperties<U>& props, const hpp::Tensor2<U>& T, const std::vector<U>& s_alphas, const U dt) {
    std::vector<U> gammadot_alphas = shearStrainRates(props, T, s_alphas);
    std::vector<U> Dgamma_alphas = gammadot_alphas*dt;       
    return Dgamma_alphas;
}

/**
 * @brief In-place version of hpp.shearStrainIncrements
 * @param props the properties of the material defined in hpp.CrystalProperties
 * @param T \f$\mathbf{T}^*\f$
 * @param s_alphas a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
 * @param dt \f$\Delta t\f$
 * @param A list of \f$\Delta \gamma^\alpha\f$ for each \f$\alpha\f$ to return
 */
template <typename U>
inline void shearStrainIncrementsInPlace(const CrystalProperties<U>& props, const hpp::Tensor2<U>& T, const std::vector<U>& s_alphas, const U dt, std::vector<U>& Dgamma_alphas) {
    shearStrainRatesInPlace(props, T, s_alphas, Dgamma_alphas);
    Dgamma_alphas *= dt;
}

/**
 * @brief See kalidindi1992.PartialDGammaPartialT
 * @param m \f$m\f$
 * @param gammadot_0 \f$\dot{\gamma}_0\f$
 * @param T \f$\mathbf{T}^*(t_{i+1})\f$
 * @param S_0_alpha \f$\mathbf{S}_0^\alpha\f$
 * @param s_alpha \f$s^\alpha(t_{i+1})\f$
 * @param dt \f$\Delta t\f$
 * @return 
 */
template <typename U>
hpp::Tensor2<U> partialDGammaPartialT(U m, U gammadot_0, const hpp::Tensor2<U>& T, 
const hpp::Tensor2<U>& S_0_alpha, U s_alpha, U dt)
{
    U tau_alpha = hpp::contract<U>(T, S_0_alpha);
    U oom = (U)(1.0/m);
    return oom*gammadot_0*((std::pow(std::abs(s_alpha),-oom))*std::pow(std::abs(tau_alpha),oom-(U)1.0))*dt*S_0_alpha;
}

/**
 * @brief In-place version of hpp.partialDGammaPartialT
 * @param m \f$m\f$
 * @param gammadot_0 \f$\dot{\gamma}_0\f$
 * @param T \f$\mathbf{T}^*(t_{i+1})\f$
 * @param S_0_alpha \f$\mathbf{S}_0^\alpha\f$
 * @param s_alpha \f$s^\alpha(t_{i+1})\f$
 * @param dt \f$\Delta t\f$
 * @param pdgpt
 */
template <typename U>
void partialDGammaPartialTInPlace(U m, U gammadot_0, const hpp::Tensor2<U>& T, 
const hpp::Tensor2<U>& S_0_alpha, U s_alpha, U dt, hpp::Tensor2<U>& pdgpt)
{
    U tau_alpha = hpp::contract<U>(T, S_0_alpha);
    U oom = (U)(1.0/m);
    pdgpt = S_0_alpha;
    U fac = oom*gammadot_0*((std::pow(std::abs(s_alpha),-oom))*std::pow(std::abs(tau_alpha),oom-1.0))*dt;
    pdgpt *= fac;
}

/**
 * @brief See kalidindi1992.TensorJ
 * @param props the properties of the material defined in hpp.CrystalProperties
 * @param C_alphas list of \f$\mathbf{C}^\alpha\f$ for each \f$\alpha\f$
 * @param s_alphas a list of \f$s^\alpha(t_{i+1})\f$ for each slip system \f$\alpha\f$
 * @param T \f$\mathbf{T}^*(t_{i+1})\f$
 * @param dt \f$\Delta t\f$
 * @return \f$\mathbf{J}\f$
 */
template <typename U>
inline hpp::Tensor4<U> tensorJ(const CrystalProperties<U>& props, const std::vector<hpp::Tensor2<U>>& C_alphas, const std::vector<U>& s_alphas, const hpp::Tensor2<U>& T, const U dt) 
{
    hpp::Tensor4<U> J = hpp::identityTensor4<U>(3);
    for (unsigned int i=0; i<props.n_alpha; i++) {
        hpp::Tensor2<U> partial_Dgamma_partial_T = partialDGammaPartialT(props.m, props.gammadot_0, T, props.S_0[i], s_alphas[i], dt);
        J += hpp::outer<U>(C_alphas[i], partial_Dgamma_partial_T);
    }
    return J;
}

/**
 * @brief In-place version of hpp.tensorJ
 * @param dum2ndOrder a 3x3 dummy tensor
 * @param dum2ndOrder a 3x3x3x3 dummy tensor
 */
template <typename U>
inline void tensorJInPlace(const CrystalProperties<U>& props, const std::vector<hpp::Tensor2<U>>& C_alphas, 
const std::vector<U>& s_alphas, const hpp::Tensor2<U>& T, const U dt, hpp::Tensor2<U>& dum2ndOrder, hpp::Tensor4<U>& dum4thOrder, hpp::Tensor4<U>& J) 
{
    hpp::identityTensor4InPlace<U>(3, J);
    for (unsigned int i=0; i<props.n_alpha; i++) {
        partialDGammaPartialTInPlace(props.m, props.gammadot_0, T, props.S_0[i], s_alphas[i], dt, dum2ndOrder);    
        outerInPlace(C_alphas[i], dum2ndOrder, dum4thOrder);
        J += dum4thOrder;
    }
}

/**
 * @brief See kalidindi1992.TensorG
 */
template <typename U>
inline hpp::Tensor2<U> tensorG(const hpp::Tensor4<U>& L, const hpp::Tensor2<U>& A, const hpp::Tensor2<U>& T_prev_iter, 
const std::vector<U>& Dgamma_alphas, const std::vector<hpp::Tensor2<U>>& C_alphas) 
{
    // Tensor T_tr: defined in Equation 28 of Kalidindi1992
    hpp::Tensor2<U> T_tr = hpp::contract<U>(L, (U)0.5*(A-hpp::identityTensor2<U>(3)));
    
    // Evaluating Equation 32 of Kalidindi1992
    hpp::Tensor2<U> G = T_prev_iter - T_tr;
    unsigned int n_alpha = Dgamma_alphas.size();
    for (unsigned int i=0; i<n_alpha; i++) {
        G += Dgamma_alphas[i]*C_alphas[i];
    }
    
    // Return 
    return G;
}

/**
 * @brief In-place version of hpp.TensorG
 */
template <typename U>
inline void tensorGInPlace(const hpp::Tensor4<U>& L, const hpp::Tensor2<U>& A, const hpp::Tensor2<U>& T_prev_iter, 
const std::vector<U>& Dgamma_alphas, const std::vector<hpp::Tensor2<U>>& C_alphas, Tensor2<U>& dum2ndOrder, hpp::Tensor2<U>& dumT_tr, hpp::Tensor2<U>& G) 
{
    // Tensor T_tr: defined in Equation 28 of Kalidindi1992
    identityTensor2InPlace(3, dum2ndOrder);
    dum2ndOrder -= A;
    dum2ndOrder *= (U)(-0.5);
    hpp::contractInPlace(L, dum2ndOrder, dumT_tr);
    
    // Evaluating Equation 32 of Kalidindi1992
    G = T_prev_iter;
    G -= dumT_tr;
    unsigned int n_alpha = Dgamma_alphas.size();
    for (unsigned int i=0; i<n_alpha; i++) {
        G += Dgamma_alphas[i]*C_alphas[i];
    }
}

/**
 * @brief See kalidindi1992.NewtonStressCorrection
 */
template <typename U>
hpp::Tensor2<U> newtonStressCorrection(const CrystalProperties<U>& props, const hpp::Tensor2<U>& A, 
const std::vector<hpp::Tensor2<U>>& C_alphas, const std::vector<U>& s_alphas, const hpp::Tensor2<U>& T_old, const U DT_max, const U dt)
{
    // Shear strain increments
    std::vector<U> Dgamma_alphas = shearStrainIncrements(props, T_old, s_alphas, dt);    
    
    // Tensor J
    hpp::Tensor4<U> J = tensorJ(props, C_alphas, s_alphas, T_old, dt);
    
    // Tensor G: defined in Equation 30 of Kalidindi1992
    hpp::Tensor2<U> G = tensorG(props.L, A, T_old, Dgamma_alphas, C_alphas);
    
    // Invert J
    J.invInPlace();
    
    // Calculate the correction
    hpp::Tensor2<U> DT(3,3);
    hpp::contractInPlace<U>(J, G, DT);
    DT *= (U)(-1.0);

    // Constrain correction. Equation 35 in Kalidindi1992
    DT.constrainInPlace(-DT_max, DT_max);

    // Return
    return DT;
}

/**
 * @brief In-place version of hpp.newtonStressCorrection
 */
template <typename U>
void newtonStressCorrectionInPlace(const CrystalProperties<U>& props, const hpp::Tensor2<U>& A, 
const std::vector<hpp::Tensor2<U>>& C_alphas, const std::vector<U>& s_alphas, const hpp::Tensor2<U>& T_old, const U DT_max, const U dt, std::vector<U>& dumDgamma_alphas, hpp::Tensor4<U>& dumJ, hpp::Tensor2<U>& dumT_tr, hpp::Tensor2<U>& dumG, hpp::Tensor2<U>& dum2ndOrder, hpp::Tensor4<U>& dum4thOrder, hpp::Tensor2<U>& DT)
{
    // Shear strain increments
    shearStrainIncrementsInPlace(props, T_old, s_alphas, dt, dumDgamma_alphas);
    
    // Tensor J
    tensorJInPlace<U>(props, C_alphas, s_alphas, T_old, dt, dum2ndOrder, dum4thOrder, dumJ);
    
    // Tensor G: defined in Equation 30 of Kalidindi1992
    tensorGInPlace(props.L, A, T_old, dumDgamma_alphas, C_alphas, dum2ndOrder, dumT_tr, dumG);
    
    // Invert J
    dumJ.invInPlace();
    
    // Calculate the correction
    hpp::contractInPlace(dumJ, dumG, DT);
    DT *= (U)(-1.0);

    // Constrain correction. Equation 35 in Kalidindi1992
    DT.constrainInPlace(-DT_max, DT_max);
}


template <typename U>
void Crystal<U>::applyInitialConditions() 
{
    // Apply initial conditions
    T = init.T_init;
    s_alphas = init.s_0*hpp::ones<U>(props.n_alpha);
    F_p = init.F_p_0;
    
    // Derived initial conditions
    Tensor2<U> F = identityTensor2<U>(3);
    F_e = F*F_p.inv();

    // The most recent step (the reset) has been accepted
    step_accepted = true;
    step_rejected = false;
}    

// Default constructor
template <typename U>
Crystal<U>::Crystal() {
    ;
}

template <typename U>
Crystal<U>::Crystal(const CrystalProperties<U>& props, const CrystalSolverConfig<U>& config, 
                    const CrystalInitialConditions<U>& init, const CrystalOutputConfig& outputConfig)
{
    // Material properties
    this->props = props;
    
    // Solver configuration
    this->config = config;
    
    // Initial conditions
    this->init = init;
    
    // Output configuration
    this->outputConfig = outputConfig;
    
    // Convergence tolerances that depend on initial conditions
    DT_max = config.DT_max_factor*init.s_0;
    DT_tol = config.DT_tol_factor*init.s_0;
    Ds_tol = config.Ds_tol_factor*init.s_0;

    // Apply initial conditions
    this->applyInitialConditions();
    
    // Allocate for dummy variables
    dumDgamma_alphas = std::vector<U>(props.n_alpha);
    dumC_alphas = std::vector<hpp::Tensor2<U>>(props.n_alpha);
    for (auto&& C_alpha : dumC_alphas) {
        C_alpha = hpp::Tensor2<U>(3,3);
    }
    dum2ndOrders = std::vector<Tensor2<U>>(3);
    for (auto&& dum2ndOrder : dum2ndOrders) {
        dum2ndOrder = Tensor2<U>(3,3);
    }
    dum4thOrders = std::vector<Tensor4<U>>(2);
    for (auto&& dum4thOrder : dum4thOrders) {
        dum4thOrder = Tensor4<U>(3,3,3,3);
    }
}

template <typename U>
Crystal<U>::Crystal(const CrystalProperties<U>& props, const CrystalSolverConfig<U>& config, 
                    const CrystalInitialConditions<U>& init)
{
    CrystalOutputConfig outputConfig;
    *this = Crystal<U>(props, config, init, outputConfig);
}

template <typename U>
void Crystal<U>::assertAcceptedOrRejectedStep() 
{
    if (!(this->step_accepted || this->step_rejected)) {
        throw CrystalError("Previous step must be accepted xor rejected before this operation.");
    } else if (this->step_accepted && this->step_rejected) {
        throw CrystalError("Previous step must be accepted xor rejected before this operation.");
    }
}

template <typename U>
bool Crystal<U>::updateT(const hpp::Tensor2<U>& A, U dt) 
{
    // Initial state
    Tensor2<U> T_iter = T;
    
    // Tensors C_alpha
    tensorC_alphasInPlace(props.L, A, props.S_0, props.n_alpha, dum2ndOrders[0], dumC_alphas);
    
    // Run the iteration
    bool step_good = false;
    Tensor2<U> DT(3,3);
    for (unsigned int i=0; i<config.algebraic_max_iter; i++) {             
        // Evaluate the iteration step
        //hpp::Tensor2<U> DT = newtonStressCorrection(props, A, dumC_alphas, s_alphas, T_iter, DT_max, dt);
        newtonStressCorrectionInPlace(props, A, dumC_alphas, s_alphas, T_iter, DT_max, dt, dumDgamma_alphas, dum4thOrders[0], dum2ndOrders[0], dum2ndOrders[1], dum2ndOrders[2], dum4thOrders[1], DT);
        
        // Step
        T_iter += DT;
        
        // Convergence criterion in Kalidindi1992
        if ((DT.absmax() < DT_tol)) {
            step_good = true;
            break;
        }
    }    
    T_next = T_iter;
    
    // Return
    return step_good;
}

template <typename U>
U slipHardeningRate(U h_0, U s_s, U a, U s_beta) 
{
    return h_0*(std::pow((U)1.0-s_beta/s_s,a));
}

template <typename U>
hpp::Tensor2<U> strainHardeningRates(const CrystalProperties<U>& props, const std::vector<U>& s_alphas) 
{
    hpp::Tensor2<U> h(props.n_alpha, props.n_alpha);
    for (unsigned int j=0; j<props.n_alpha; j++) {
        U h_b = slipHardeningRate(props.h_0, props.s_s, props.a, s_alphas[j]);
        for (unsigned int i=0; i<props.n_alpha; i++) {
            h(i,j) = props.Q.getVal(i,j)*h_b;
        }
    }
    return h;
}

/**
 * @brief Voce hardening law
 * @detail identical for each slip system
 * @param props
 * @param s_alphas
 * @return 
 */
template <typename U>
U strainHardeningRateVoce(const CrystalProperties<U>& props, const std::vector<U>& s_alphas) 
{
    U h = slipHardeningRate(props.h_0, props.s_s, props.a, s_alphas[0]);
    return h;
}

template <typename U>
std::vector<U> slipDeformationResistanceUpdate(const CrystalProperties<U>& props, const hpp::Tensor2<U>& T_next, 
const std::vector<U>& s_alphas_current_time, const std::vector<U>& s_alphas_prev_iter, const U dt) 
{
    // Evaluate shear strain increments
    std::vector<U> Dgamma_alphas = shearStrainIncrements(props, T_next, s_alphas_prev_iter, dt);
    
    // Evaluating Equation 36 in Kalidindi 1992
    std::vector<U> s_alphas(props.n_alpha);    
    
    if (props.hardeningLaw == HARDENING_LAW_BROWN) {        
        // Evaluating Equation 41 in Kalidindi1992
        hpp::Tensor2<U> h;
        h = strainHardeningRates(props, s_alphas_prev_iter);
        
        // Continuing with evaluating Equation 36
        for (unsigned int a=0; a<props.n_alpha; a++) {
            s_alphas[a] = s_alphas_current_time[a];
            for (unsigned int b=0; b<props.n_alpha; b++) {
                s_alphas[a] += h(a,b)*std::abs(Dgamma_alphas[b]);
            }
        }
    }
    else if (props.hardeningLaw == HARDENING_LAW_VOCE) {
        // Evaluating Equation 41 in Kalidindi1992 for the case of
        // only one shared slip system deformation resistance
        U h = strainHardeningRateVoce(props, s_alphas_prev_iter);
        
        // Continuing with evaluating Equation 36 for the case of
        // only one shared slip system deformation resistance
        U Dgamma_abs_sum = 0.0;
        for (const auto& Dgamma_alpha : Dgamma_alphas) {
            Dgamma_abs_sum += std::abs(Dgamma_alpha);
        }
        U s_new = s_alphas_current_time[0] + h*Dgamma_abs_sum;
        for (auto&& s_alpha : s_alphas) {
            s_alpha = s_new;
        }
    }
    else {
        throw std::runtime_error("Did not recognise hardening law.");
    }

    // Return
    return s_alphas; 
}

/**
 * @brief From kalidindi2006 Equation 5
 * @param props
 * @param s_alpha
 * @param gammaSum
 * @param dt
 * @return 
 */
template <typename U>
U slipDeformationResistanceStepSpectralSolver(const CrystalProperties<U>& props, 
const U s_alpha, const U gammaSum, const U dt) 
{ 
    // Return
    U sDot = props.h_0*std::pow((U)1.0 - s_alpha/props.s_s, props.a)*gammaSum;
    return s_alpha + sDot*dt;
}

template <typename U>
bool Crystal<U>::updateS(U dt) 
{
    // Initial s for iterative scheme is s from the previous timestep
    std::vector<U> s_alphas_prev_iter = s_alphas;
    
    // Second level of iterative scheme, for s
    bool step_good = false;
    std::vector<U> s_alphas_iter;
    for (unsigned int i=0; i<config.algebraic_max_iter; i++) {
        // Evaluate the iteration step
        s_alphas_iter = slipDeformationResistanceUpdate(props, T_next, s_alphas, s_alphas_prev_iter, dt);
        std::vector<U> Ds = s_alphas_iter - s_alphas_prev_iter;
        
        // Convergence criterion in Kalidindi1992
        if (max(abs(Ds)) < Ds_tol) {
            step_good = true;
            break;
        }
        
        // This is will be the new "previous" iteration
        s_alphas_prev_iter = s_alphas_iter;
    }
    s_alphas_next = s_alphas_iter;
    
    // Return
    return step_good;
}

template <typename U>
bool Crystal<U>::updateTandS(const hpp::Tensor2<U>& A, U dt) 
{
    // T step
    bool step_good_for_T = this->updateT(A, dt);

    // S step
    bool step_good_for_s = false;
    if (step_good_for_T) {
        step_good_for_s = this->updateS(dt);
    }
    
    // Return
    bool step_good = (step_good_for_T && step_good_for_s);
    return step_good;    
}

template <typename U>
hpp::Tensor2<U> plasticDeformationGradientUpdate(const CrystalProperties<U>& props, const hpp::Tensor2<U>& F_p_prev_time,
const std::vector<U>& Dgamma_alphas)
{
    // Update
    hpp::Tensor2<U> F_p = hpp::identityTensor2<U>(3);
    for (unsigned int i=0; i<props.n_alpha; i++) {
        F_p += Dgamma_alphas[i]*props.S_0[i];
    }
    F_p = F_p*F_p_prev_time;
    
    // Scale to unit determinant
    hpp::Tensor2<U> F_p_scaled = F_p.scaledToUnitDeterminant();
    
    // Return
    return F_p_scaled;
}

/**
 * @brief Mihaila 2014 equation 14
 * @param mprops
 * @param gammadot_alphas
 * @return 
 */
template <typename U>
Tensor2<U> plasticSpinTensor(CrystalProperties<U> mprops, std::vector<U> gammadot_alphas,
                             std::vector<std::vector<U>> m_alphas, 
                             std::vector<std::vector<U>> n_alphas)
{
    // Calculate
    Tensor2<U> W_p(3,3);    
    for (unsigned int i=0; i<m_alphas.size(); i++) {
        W_p += gammadot_alphas[i]*(outer(m_alphas[i], n_alphas[i])-outer(n_alphas[i], m_alphas[i]));
    }
    W_p *= (U)0.5;
    
    // Return
    return W_p;
}

template <typename U>
bool strainRateLowEnough(const std::vector<U>& Dgamma_alphas, U Dgamma_goal) {
    U r = max(abs(Dgamma_alphas))/Dgamma_goal;
    if (r > 1.25) {
        return false;
    }
    else {
        return true;
    }
}

template <typename U>
bool Crystal<U>::tryStep(const hpp::Tensor2<U>& F_next, U dt) 
{
    // The previous step should have been accepted xor rejected
    this->assertAcceptedOrRejectedStep();
    
    // Known quantities at the next timestep
    hpp::Tensor2<U> tensor_A = tensorA<U>(F_p, F_next);
    
    // Take step
    bool step_good = false;
    try {
        step_good = updateTandS(tensor_A, dt);
    } catch (hpp::TensorError& error) {
        DEBUG_ONLY(std::cout << "Caught during step: " << error.what() << std::endl;)
        DEBUG_ONLY(std::cout << "Rejecting step." << std::endl;)
        step_good = false;
    }
    
    // Update derived quantities
    if (step_good) {
        Dgamma_alphas_next = shearStrainIncrements(props, T_next, s_alphas_next, dt);
        F_p_next = plasticDeformationGradientUpdate(props, F_p, Dgamma_alphas_next);
        F_e_next = F_next*F_p_next.inv();
    }
    
    // It is currently neither accepted nor rejected
    step_accepted = false;
    step_rejected = false;
    
    // Check if the strain rate is fine
    if (step_good) {
        step_good = strainRateLowEnough(Dgamma_alphas_next, config.Dgamma_max);
    }
    
    // Report if step was successful
    return step_good;
}

template <typename U>
void Crystal<U>::acceptStep() 
{
    // Update main quantities
    T = T_next;
    s_alphas = s_alphas_next;
    Dgamma_alphas = Dgamma_alphas_next;
    F_p = F_p_next;
    F_e = F_e_next;
    
    // Declare step accepted
    step_rejected = false;
    step_accepted = true;
}

template <typename U>
void Crystal<U>::rejectStep() 
{
    step_rejected = true;
    step_accepted = false;
}

/**
 * @brief The exact method in kalidindi1992.
 * @details Equation 38 in kalidindi1992
 * @param dt_old the old timestep
 * @param Dgamma_alphas the strain increments
 * @param Dgamma_goal the strain increment goal
 * @return The new timestep
 */
template <typename U>
U setTimestepByShearRate(U dt_old, const std::vector<U>& Dgamma_alphas, U Dgamma_goal)
{
    U r = max(abs(Dgamma_alphas))/Dgamma_goal;
    U dt;
    if (r < 0.8) {
        dt = 1.25*dt_old;
    } else if (r >= 0.8 && r <= 1.25) {
        dt = dt_old/r;
    } else {
        throw CrystalError("This step should have been rejected.");
    }        
    return dt;
}

template <typename U>
U Crystal<U>::recommendNextTimestepSize(U dt) 
{
    U new_dt;
    if (step_accepted) {
        new_dt = setTimestepByShearRate(dt, Dgamma_alphas, config.Dgamma_max);
    }
    else if (step_rejected) {
        new_dt = 0.75*dt;
    } else {
        throw CrystalError("The step should either be accepted or rejected.");
    }
    return new_dt;
}

/**
 * @brief Equation 13 in kalidindi1992
 */
template <typename U>
std::vector<std::vector<U>> Crystal<U>::getM_alphas() const
{
    std::vector<std::vector<U>> m_alphas(props.n_alpha);
    for (unsigned int i=0; i<props.n_alpha; i++) {
        m_alphas[i] = F_e*(props.m_0[i]);
    }
    return m_alphas;
}

/**
 * @brief Equation 14 in kalidindi1992
 */
template <typename U>
std::vector<std::vector<U>> Crystal<U>::getN_alphas() const
{
    hpp::Tensor2<U> F_e_T_inv = (F_e.trans()).inv();
    std::vector<std::vector<U>> n_alphas(props.n_alpha);
    for (unsigned int i=0; i<props.n_alpha; i++) {
        n_alphas[i] = F_e_T_inv*(props.n_0[i]);
    }
    return n_alphas;
}

template <typename U>
std::vector<U> Crystal<U>::getShearStrainRates() {
    return shearStrainRates(props, T, s_alphas); 
}

template <typename U>
Tensor2<U> Crystal<U>::getPlasticSpinTensor() {
    std::vector<std::vector<U>> m_alphas = this->getM_alphas();
    std::vector<std::vector<U>> n_alphas = this->getN_alphas();
    std::vector<U> gammadot_alphas = this->getShearStrainRates();
    return plasticSpinTensor(props, gammadot_alphas, m_alphas, n_alphas);
}

template <typename U>
EulerAngles<U> Crystal<U>::getEulerAngles() const {
    PolarDecomposition<U> F_e_decomp = F_e.polarDecomposition();
    return getEulerZXZAngles(F_e_decomp.R*init.crystalRotation);
}

// POLYCRYSTAL //
/////////////////

template <typename U>
Polycrystal<U>::Polycrystal(const std::vector<Crystal<U>>& crystal_list, MPI_Comm comm, const PolycrystalOutputConfig& outputConfig) {
    this->crystal_list = crystal_list;
    this->comm = comm;
    int csize, crank;
    MPI_Comm_size(comm, &csize);
    MPI_Comm_rank(comm, &crank);
    this->comm_size = csize;
    this->comm_rank = crank;
    this->outputConfig = outputConfig;
}

template <typename U>
Polycrystal<U>::Polycrystal(const std::vector<Crystal<U>>& crystal_list, MPI_Comm comm) {
    PolycrystalOutputConfig outputConfig;
    *this = Polycrystal(crystal_list, comm, outputConfig);
}

template <typename U>
void Polycrystal<U>::applyInitialConditions() 
{
    for (auto&& crystal : crystal_list) {
        crystal.applyInitialConditions();
    }
    this->updateDerivedQuantities();
}

template <typename U>
void Polycrystal<U>::updateDerivedQuantities() 
{
    // Local Cauchy stress and volume
    hpp::Tensor2<U> T_cauchy_local(3,3);
    U volume_local = 0.0;
    for (auto&& crystal : crystal_list) {
        T_cauchy_local += crystal.getTCauchy();
        volume_local += crystal.getVolume();
    }
    
    // Global volume
    U volume_global = MPISum(volume_local, comm);  
    
    // Global Cauchy stress
    hpp::Tensor2<U> T_cauchy_global = MPISum(T_cauchy_local, comm);
    
    // Average cauchy stress
    T_cauchy = T_cauchy_global/volume_global;
}

template <typename U>
bool Polycrystal<U>::step(hpp::Tensor2<U> F_next, U dt)
{ 
    // Try stepping through all of the crystals
    bool step_good = true;
    for (auto&& crystal : crystal_list) {
        step_good = crystal.tryStep(F_next, dt);
        if (step_good == false) {
            crystal.rejectStep();
            break;
        }     
    }
    bool step_good_global = MPIAllTrue(step_good, comm);
    
    // Accept or reject step
    if (step_good_global) {
        for (auto&& crystal : crystal_list) {
            crystal.acceptStep();
        }
        this->updateDerivedQuantities();
    } else {
        for (auto&& crystal : crystal_list) {
            crystal.rejectStep();
        }
    }
        
    // Return
    return step_good_global;      
}
    
template <typename U>
U Polycrystal<U>::recommendNextTimestepSize(U dt) {    
    U new_dt = std::numeric_limits<U>::max();
    for (auto&& crystal : crystal_list) {
        U recommended_dt = crystal.recommendNextTimestepSize(dt);
        if (recommended_dt < new_dt) new_dt = recommended_dt;
    }
    U new_dt_global = MPIMin(new_dt, comm);
    return new_dt_global;
}        

template <typename U>
void Polycrystal<U>::evolve(U t_start, U t_end, U dt_initial, std::function<hpp::Tensor2<U>(U t)> F_of_t) 
{
    // Initialize
    this->applyInitialConditions();
    U t = t_start;
    U dt = dt_initial;
    
    // Initial data
    t_history.push_back(t);
    T_cauchy_history.push_back(T_cauchy);
    
    // Manage texture writing interval
    U tNextTextureSave;
    if (outputConfig.writeTextureHistory) {
        tNextTextureSave = t+outputConfig.textureHistoryTimeInterval;
    }
    else {
        tNextTextureSave = t_end;
    }
    
    // Add initial texture
    if (outputConfig.writeTextureHistory) {
        this->addTextureToHistory();
    }
    
    // Evolve
    int prevent_next_x_timestep_increases = 0;
    while (t<t_end) {
        if (outputConfig.verbose) {
            if (comm_rank == 0) std::cout << "t = " << t << std::endl;
        }
        
        // If timestep was decreased earlier, prevent subsequent increases
        if (prevent_next_x_timestep_increases > 0) {
            prevent_next_x_timestep_increases -= 1;
        }
            
        // Do the main solve
        bool step_good = false;
        U new_dt;
        while (!step_good) {            
            // Next deformation gradient
            hpp::Tensor2<U> F_next = F_of_t(t + dt);
            
            // Try step
            step_good = this->step(F_next, dt);
            new_dt = this->recommendNextTimestepSize(dt);
            
            // If the step is not accepted, reduce the timestep
            if (!step_good) {
                // Use the adjusted timestep now
                dt = new_dt;
                
                // "Further, in order to improve on the efficiency of 
                // the scheme, the next 10 time steps are taken with this time 
                // step or with a smaller value as governed by the basic algorithm 
                // discussed above." in Kalidindi1992
                prevent_next_x_timestep_increases = 10;                
            }
        }
        
        // Update the current time
        t += dt;
            
        // Set the adjusted timestep for the next step
        if (prevent_next_x_timestep_increases == 0) {
            dt = new_dt;
        }

        // Store stress strain history
        t_history.push_back(t);
        T_cauchy_history.push_back(T_cauchy);
        
        // Store texture history
        if (t >= tNextTextureSave) {
            this->addTextureToHistory();
            tNextTextureSave += outputConfig.textureHistoryTimeInterval;
        }
    }
}

template <typename T>
std::vector<T> cartesianToSpherical(const std::vector<T>& cartVec) {
    // Magnitude
    T r = 0.0;
    for (const auto& comp : cartVec) {
        r += std::pow(comp, (T)2.0);
    }
    r = std::sqrt(r);
    std::vector<T> unitVec = cartVec/r;
    
    // Azimuthal component
    T theta = std::atan2(unitVec[1], unitVec[0]);
    
    // Polar
    T phi = std::acos(unitVec[2]);    
    
    // Return
    std::vector<T> sphereVec(3);
    sphereVec[0] = r;
    sphereVec[1] = theta;
    sphereVec[2] = phi;
    return sphereVec;
}

template<typename T>
Tensor2<T> histogramPoleEqualArea(const std::vector<EulerAngles<T>>& angles, const std::vector<T>& planeNormal) {
    // Maximum R value from northern hemisphere projection
    T maxR = (1.00001)*2*std::sin(M_PI/4);
    
    // Loop over crystals
    Tensor2<T> hist(HPP_POLE_FIG_HIST_DIM, HPP_POLE_FIG_HIST_DIM);
    for (const auto& angle : angles) {
        // Get orientation of the crystal
        Tensor2<T> ROrientation = EulerZXZRotationMatrix(angle);
        
        // Active rotation
        std::vector<T> pole = ROrientation*planeNormal;
        std::vector<T> poleSpherical = cartesianToSpherical(pole);
        T theta = poleSpherical[1];
        T phi = poleSpherical[2];
        
        // Equal-area projection
        T R = 2*std::sin(phi/2);
        T x, y;
        x = R*std::cos(theta);
        y = R*std::sin(theta);
        
        // Histogram index
        T xMin = -maxR;
        T xMax = maxR;
        T yMin = xMin;
        T yMax = xMax;
        T binwidthX = (xMax-xMin)/HPP_POLE_FIG_HIST_DIM;
        T binwidthY = (yMax-yMin)/HPP_POLE_FIG_HIST_DIM;
        int ix = (int) ((x-xMin)/binwidthX);
        int iy = (int) ((y-yMin)/binwidthY);
        
        // Add points to histogram
        if (ix >=0 && ix < HPP_POLE_FIG_HIST_DIM && iy>=0 && iy < HPP_POLE_FIG_HIST_DIM) {
            hist(ix, iy) += 1.0;
        }
    }
    
    // Return
    return hist;
}

template <typename U>
void Polycrystal<U>::addTextureToHistory() {
    // Calculate local orientations
    std::vector<U> alphasLocal(crystal_list.size());
    std::vector<U> betasLocal(crystal_list.size());
    std::vector<U> gammasLocal(crystal_list.size());
    for (unsigned int i=0; i<crystal_list.size(); i++) {
        EulerAngles<U> angles = crystal_list[i].getEulerAngles();
        alphasLocal[i] = angles.alpha;
        betasLocal[i] = angles.beta;
        gammasLocal[i] = angles.gamma;
    }
    
    // Gather on root
    std::vector<U> alphasGlobalRoot = MPIConcatOnRoot(alphasLocal, comm);
    std::vector<U> betasGlobalRoot = MPIConcatOnRoot(betasLocal, comm);
    std::vector<U> gammasGlobalRoot = MPIConcatOnRoot(gammasLocal, comm);
    std::vector<EulerAngles<U>> anglesGlobalRoot(alphasGlobalRoot.size());
    for (unsigned int i=0; i<anglesGlobalRoot.size(); i++) {
        anglesGlobalRoot[i].alpha = alphasGlobalRoot[i];
        anglesGlobalRoot[i].beta = betasGlobalRoot[i];
        anglesGlobalRoot[i].gamma = gammasGlobalRoot[i];
    }
    
    // Calculate pole figures and store on root
    if (comm_rank == 0) {
        this->poleHistogramHistory111.push_back(histogramPoleEqualArea(anglesGlobalRoot, std::vector<U>{1,1,1}));
        this->poleHistogramHistory110.push_back(histogramPoleEqualArea(anglesGlobalRoot, std::vector<U>{1,1,0}));
        this->poleHistogramHistory100.push_back(histogramPoleEqualArea(anglesGlobalRoot, std::vector<U>{1,0,0}));
        this->poleHistogramHistory001.push_back(histogramPoleEqualArea(anglesGlobalRoot, std::vector<U>{0,0,1}));
        this->poleHistogramHistory011.push_back(histogramPoleEqualArea(anglesGlobalRoot, std::vector<U>{0,1,1}));
    }
}

/**
 * @brief Writes out pole histograms to HDF5.
 * @detail
 * @param outfile the output file
 * @param poles the poles to plot
 */
template <typename T>
void writePoleHistogramHistoryHDF5(H5::H5File& outfile, const std::string& dsetBaseName, std::vector<Tensor2<T>>& history, const std::vector<T>& pole) {
    // Data dimensions
    const unsigned int nTimesteps = history.size();
    const unsigned int histDimX = history[0].getn1();
    const unsigned int histDimY = history[0].getn2();

    // Create dataset
    std::string dsetName = dsetBaseName + "_";        
    for (auto val : pole) {
        dsetName += std::to_string((int)val);
    }
    std::vector<hsize_t> dataDims = {nTimesteps, histDimX, histDimY};
    auto dset = createHDF5Dataset<T>(outfile, dsetName, dataDims);
    
    // Write to dataset
    for (unsigned int i=0; i<nTimesteps; i++) {
        std::vector<hsize_t> offset = {i};
        history[i].writeToExistingHDF5Dataset(dset, offset);
    }
}

template <typename U>
void Polycrystal<U>::writeResultHDF5(std::string filename) {    
    if (comm_rank == 0) {
        // Open output file
        H5::H5File outfile(filename.c_str(), H5F_ACC_TRUNC);
        
        // Pole figure histograms
        std::string poleHistBasename = "poleHistogram";
        writePoleHistogramHistoryHDF5(outfile, poleHistBasename, this->poleHistogramHistory111, std::vector<U>{1,1,1});
        writePoleHistogramHistoryHDF5(outfile, poleHistBasename, this->poleHistogramHistory110, std::vector<U>{1,1,0});
        writePoleHistogramHistoryHDF5(outfile, poleHistBasename, this->poleHistogramHistory100, std::vector<U>{1,0,0});
        writePoleHistogramHistoryHDF5(outfile, poleHistBasename, this->poleHistogramHistory001, std::vector<U>{0,0,1});
        writePoleHistogramHistoryHDF5(outfile, poleHistBasename, this->poleHistogramHistory011, std::vector<U>{0,1,1});
        
        // Stress history
        writeVectorToHDF5Array(outfile, "tHistory", this->t_history);    
        std::vector<hsize_t> timeDims = {this->T_cauchy_history.size()};
        std::vector<hsize_t> tensorDims = {3,3};
        H5::DataSet TCauchyDset = createHDF5GridOfArrays<U>(outfile, "TCauchyHistory", timeDims, tensorDims);
        for (unsigned int i=0; i<this->T_cauchy_history.size(); i++) {
            std::vector<hsize_t> offset = {i};
            this->T_cauchy_history[i].writeToExistingHDF5Dataset(TCauchyDset, offset);
        }
        
        // Close
        outfile.close();
    }
}

// Crystal and Polycrystal are restricted to these specific instantiations
template class Crystal<float>;
template class Crystal<double>;
template class Polycrystal<float>;
template class Polycrystal<double>;

//////////////////////////////
// SPECTRAL CRYSTAL SOLVERS //
//////////////////////////////

template <typename U>
SpectralCrystal<U>::SpectralCrystal() 
{
}

template <typename U>
SpectralCrystal<U>::SpectralCrystal(const CrystalProperties<U>& props, const SpectralCrystalSolverConfig<U>& config, const CrystalInitialConditions<U>& init) {
    // Material properties
    this->props = props;
    
    // Solver configuration
    this->config = config;
    
    // Initial conditions
    this->init = init;
    this->s = init.s_0;
    this->RStar = identityTensor2<U>(3);
    this->TCauchy = Tensor2<U>(3,3);
}

template <typename U>
void SpectralCrystal<U>::step(const hpp::Tensor2<U>& F_next, const hpp::Tensor2<U>& L_next, const hpp::SpectralDatabase<U>& db, U dt) {
    // Get database parameters to look up 
    StretchingTensorDecomposition<U> stretchingTensorDecomp = getStretchingTensorDecomposition(L_next); 
    U theta = stretchingTensorDecomp.theta;
    
    // The rotational component of the stretching tensor
    Tensor2<U> RStretchingTensor = stretchingTensorDecomp.evecs;
    
    // The rotation that transforms the template crystal to have the same orientation as this one
    // First, the rotation to get to the initial configuration: init.crystalRotation
    // Second, the further rotation caused by the deformation: RStar
    Tensor2<U> ROrientation = RStar*init.crystalRotation;   
    
    // Verify it's actually a rotation matrix
    if (!RStretchingTensor.isRotationMatrix()) {
        std::cerr << "RStretchingTensor is not a rotation matrix!" << std::endl;
    }    
    
    // Transform into the stretching tensor frame
    Tensor2<U> R = RStretchingTensor.trans()*ROrientation;
    
    // Euler angles
    EulerAngles<U> angle = hpp::getEulerZXZAngles(R);
    
    // Database coordinate
    std::vector<U> gridPos = {angle.alpha, angle.beta, angle.gamma, theta};
    std::vector<unsigned int> spatialCoord(4);
    auto gridStarts = db.getGridStarts();
    auto gridSteps = db.getGridSteps();
    for (unsigned int i=0; i<4; i++) {
        spatialCoord[i] = (unsigned int) ((gridPos[i] - gridStarts[i])/gridSteps[i]);
    }
    
    // Database scaling factors
    U eDot = stretchingTensorDecomp.DNorm;  
    
    // Variables to fetch
    Tensor2<U> sigmaPrimeNext(3,3);
    Tensor2<U> WpNext(3,3);
    U gammaNext;
    
    // Gamma
    U gammaScaling = eDot;
    std::vector<unsigned int> componentIdx; //scalar = empty component vector
    gammaNext = gammaScaling*db.getIDFTReal("gammadot_abs_sum", componentIdx, spatialCoord);  
    // Update slip deformation resistance
    s = slipDeformationResistanceStepSpectralSolver(props, s, gammaNext, dt);
    // Sigma
    U sigmaScaling = (this->s*std::pow(std::abs(eDot), this->props.m));
    // Only the upper triangular terms
    for (unsigned int i=0; i<3; i++) {
        for (unsigned int j=i; j<3; j++) {
            // Skip (2,2), will be set from fact that sigma_prime is the deviatoric component
            if (i==2 && j== 2) continue;
            
            // Get the component
            std::vector<unsigned int> componentIdx = {i,j};
            sigmaPrimeNext(i,j) = sigmaScaling*db.getIDFTReal("sigma_prime", componentIdx, spatialCoord);
            
            // Symmetric
            if (i != j) sigmaPrimeNext(j,i) = sigmaPrimeNext(i,j);                
        }
    }
    // Term due to deviatoric component
    sigmaPrimeNext(2,2) = -sigmaPrimeNext(1,1) - sigmaPrimeNext(0,0);
    
    
    // Wp
    U WpScaling = eDot;
    // Only the terms (0,1), (0,2) and (1,2)
    for (unsigned int i=0; i<2; i++) {
        for (unsigned int j=i+1; j<3; j++) {
            std::vector<unsigned int> componentIdx = {i,j};
            WpNext(i,j) = WpScaling*db.getIDFTReal("W_p", componentIdx, spatialCoord);
            
            // Anti-symmetric
            WpNext(j,i) = - WpNext(i,j);
        }
    }
    
    // Update lattice rotation tensor
    Tensor2<U> WNext = (U)0.5*(L_next-L_next.trans());
    Tensor2<U> WStarNext = WNext - WpNext;
    Tensor2<U> RStarNext = RStar + WStarNext*RStar*dt;

    // End step    
    RStar = RStarNext;

    // Calculate Cauchy stress
    // Transform out of the stretching tensor frame
    TCauchy = transformOutOfFrame(sigmaPrimeNext, RStretchingTensor);  
}

//////////////////////////
// SPECTRAL POLYCRYSTAL //
//////////////////////////
template <typename U>
SpectralPolycrystal<U>::SpectralPolycrystal(const std::vector<SpectralCrystal<U>>& crystal_list, unsigned int nOmpThreads) {
    this->crystal_list = crystal_list;
    this->nOmpThreads = nOmpThreads;
}

template <typename U>
void SpectralPolycrystal<U>::step(const hpp::Tensor2<U>& F_next, const hpp::Tensor2<U>& L_next, const hpp::SpectralDatabase<U>& db, U dt) {
    solveTimer.start();
    #pragma omp parallel for num_threads(nOmpThreads)
    for (unsigned int i=0; i<crystal_list.size(); i++) {
        crystal_list[i].step(F_next, L_next, db, dt);
    }
    solveTimer.stop();
}    

/**
 * @brief Volume average the cauchy stresses
 * @return 
 */
template <typename U>
Tensor2<U> SpectralPolycrystal<U>::getGlobalTCauchy() 
{
    // Local Cauchy stress and volume
    hpp::Tensor2<U> TCauchyGlobal(3,3);
    U volumeGlobal = 0.0;
    for (unsigned int i=0; i<crystal_list.size(); i++) {
        TCauchyGlobal += crystal_list[i].getTCauchy();
        volumeGlobal += crystal_list[i].getVolume();
    }

    // Average cauchy stress
    return TCauchyGlobal/volumeGlobal;
}

template <typename U>
void SpectralPolycrystal<U>::evolve(U tStart, U tEnd, U dt, std::function<hpp::Tensor2<U>(U t)> F_of_t, std::function<hpp::Tensor2<U>(U t)> L_of_t, const hpp::SpectralDatabase<U>& db) {
    // Initial data
    t_history.push_back(tStart);
    T_cauchy_history.push_back(this->getGlobalTCauchy());
    
    // Stepping
    unsigned int nsteps = (tEnd-tStart)/dt;    
    for (unsigned int i=0; i<nsteps; i++) {
        // Inputs for the next step
        U t = tStart + (i+1)*dt;
        std::cout << "t = " << t << std::endl;
        hpp::Tensor2<U> LNext = L_of_t(t);     
        hpp::Tensor2<U> FNext = F_of_t(t);
        
        // Step
        this->step(FNext, LNext, db, dt);
        
        // Get derived quantities
        Tensor2<U> TCauchy = this->getGlobalTCauchy();
        t_history.push_back(t);
        T_cauchy_history.push_back(TCauchy);        
    }
}

template <typename U>
void SpectralPolycrystal<U>::writeResultNumpy(std::string filename)
{
    std::ofstream outfile(filename.c_str());
    outfile << "t_history = " << this->t_history << std::endl;
    outfile << "T_cauchy_history = " << this->T_cauchy_history << std::endl;
    outfile << "spectral_polycrystal_solve_time = " << solveTimer.getDuration() << std::endl;
    outfile.close();
}

// SpectralCrystal and SpectralPolycrystal are restricted to these specific instantiations
template class SpectralCrystal<float>;
template class SpectralCrystal<double>;
template class SpectralPolycrystal<float>;
template class SpectralPolycrystal<double>;

} //END NAMESPACE HPP