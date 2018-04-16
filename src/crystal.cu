#include <hpp/config.h>
HPP_CHECK_CUDA_ENABLED_BUILD
#include <hpp/tensorCUDA.h>
#include <hpp/crystalCUDA.h>
#include <hpp/spectralUtilsCUDA.h>
#include <hpp/hdfUtilsCpp.h>
#include <hpp/cudaUtils.h>

namespace hpp
{

template <typename T, unsigned int N>
CrystalPropertiesCUDA<T,N>::CrystalPropertiesCUDA(const CrystalProperties<T>& in) {
    // Check compatability
    if (in.n_alpha != N) {
        throw CrystalError("Mismatch in number of slip systems.");
    }
    
    // Populate scalars
    n_alpha = in.n_alpha;
    mu = in.mu;
    kappa = in.kappa;
    m = in.m;
    gammadot_0 = in.gammadot_0;
    h_0 = in.h_0;
    s_s = in.s_s;
    a = in.a;
    q = in.q;
    volume = in.volume;
    
    // Populate vectors and tensors
    for (unsigned int i=0; i<n_alpha; i++) {
        m_0[i] = VecCUDA<T,3>(in.m_0[i]);
        n_0[i] = VecCUDA<T,3>(in.n_0[i]);
        S_0[i] = Tensor2CUDA<T,3,3>(in.S_0[i]);
    }
    L = Tensor4CUDA<T,3,3,3,3>(in.L);
    Q = Tensor2CUDA<T,N,N>(in.Q);
}

template <typename T>
SpectralCrystalListCUDA<T>::SpectralCrystalListCUDA(unsigned int nCrystals, const SpectralCrystalCUDA<T> *crystals) {
    // Assign the values
    std::vector<T> anglesAH(nCrystals);
    std::vector<T> anglesBH(nCrystals);
    std::vector<T> anglesCH(nCrystals);
    std::vector<T> sH(nCrystals);
    for (unsigned int iCrystal=0; iCrystal<nCrystals; iCrystal++) {
        anglesAH[iCrystal] = crystals[iCrystal].angles.alpha;
        anglesBH[iCrystal] = crystals[iCrystal].angles.beta;
        anglesCH[iCrystal] = crystals[iCrystal].angles.gamma;
        sH[iCrystal] = crystals[iCrystal].s;
    }
    
    // Make the device copies
    sharedPtrs.push_back(makeDeviceCopyVecSharedPtr(anglesAH));
    anglesA = sharedPtrs.back().get();
    
    sharedPtrs.push_back(makeDeviceCopyVecSharedPtr(anglesBH));
    anglesB = sharedPtrs.back().get();
    
    sharedPtrs.push_back(makeDeviceCopyVecSharedPtr(anglesCH));
    anglesC = sharedPtrs.back().get();
    
    sharedPtrs.push_back(makeDeviceCopyVecSharedPtr(sH));
    s = sharedPtrs.back().get();
}

template <typename T>
SpectralCrystalListCUDA<T>::SpectralCrystalListCUDA(const std::vector<SpectralCrystalCUDA<T>>& crystals) {
    *this = SpectralCrystalListCUDA(crystals.size(), crystals.data());
}

template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::doGPUSetup() {    
    // Find how many GPUs are avaiable
    int nDevices;
    CUDA_CHK(cudaGetDeviceCount(&nDevices));
    if (nDevices < 1) {
        throw std::runtime_error("No GPU to use.");
    }
    
    // Select the first GPU
    deviceID = 0;   
    CUDA_CHK(cudaGetDeviceProperties(&devProp, deviceID));
    std::cout << "Using " << devProp.name << std::endl;
    
    // Get ideal parallel layout for step kernel
    if (useUnifiedDB) {
        stepKernelCfg = getKernelConfigMaxOccupancy(devProp, (void*)SPECTRAL_POLYCRYSTAL_STEP_UNIFIED<T,N,9>, nCrystalPairs);
    }
    else {
        stepKernelCfg = getKernelConfigMaxOccupancy(devProp, (void*)SPECTRAL_POLYCRYSTAL_STEP<T,N>, nCrystals);
    }    
    unsigned int nBlocks = stepKernelCfg.dG.x;
    std::cout << "Step kernel:" << std::endl;
    std::cout << stepKernelCfg;
    
    // Get parallel layout for reduce kernel
    reduceKernelLevel0Cfg = getKernelConfigMaxOccupancy(devProp, (void*)BLOCK_REDUCE_KEPLER_TENSOR2<T,3,3>, nBlocks);
    std::cout << "Reduce kernel level 0:" << std::endl;
    std::cout << reduceKernelLevel0Cfg;
    
    // Check if we need a second level of reduction
    if (reduceKernelLevel0Cfg.dG.x > 1) {        
        reduceKernelLevel1Cfg = getKernelConfigMaxOccupancy(devProp, (void*)BLOCK_REDUCE_KEPLER_TENSOR2<T,3,3>, reduceKernelLevel0Cfg.dG.x);
        TCauchyLevel0Sums.resize(reduceKernelLevel0Cfg.dG.x);
        std::cout << "Reduce kernel level 1:" << std::endl;
        std::cout << reduceKernelLevel1Cfg;
    }
    else {
        reduceKernelLevel1Cfg.dG.x = 0;
        reduceKernelLevel1Cfg.dG.y = 0;
        reduceKernelLevel1Cfg.dG.z = 0;
        reduceKernelLevel1Cfg.dB.x = 0;
        reduceKernelLevel1Cfg.dB.y = 0;
        reduceKernelLevel1Cfg.dB.z = 0;
    }

    // Working memory for cauchy stress sum
    TCauchyPerBlockSums.resize(nBlocks);
    
    /////////
    // GSH //
    /////////
    
    // Layout for GSH calculation kernel
    gshKernelCfg = getKernelConfigMaxOccupancy(devProp, (void*)GET_GSH_FROM_ORIENTATIONS<T>, nCrystals);
    
    // Get parallel layout for GSH reduce kernel
    unsigned int nGSHBlocks = gshKernelCfg.dG.x;
    gshReduceKernelLevel0Cfg = getKernelConfigMaxOccupancy(devProp, (void*)BLOCK_REDUCE_KEPLER_GSH_COEFFS<T>, nGSHBlocks);
    std::cout << "GSH Reduce kernel level 0:" << std::endl;
    std::cout << gshReduceKernelLevel0Cfg;
    
    // Check if we need a second level of reduction
    if (gshReduceKernelLevel0Cfg.dG.x > 1) {        
        gshReduceKernelLevel1Cfg = getKernelConfigMaxOccupancy(devProp, (void*)BLOCK_REDUCE_KEPLER_GSH_COEFFS<T>, gshReduceKernelLevel0Cfg.dG.x);
        gshLevel0Sums.resize(gshReduceKernelLevel0Cfg.dG.x);
        std::cout << "GSH Reduce kernel level 1:" << std::endl;
        std::cout << gshReduceKernelLevel1Cfg;
    }
    else {
        gshReduceKernelLevel1Cfg.dG.x = 0;
        gshReduceKernelLevel1Cfg.dG.y = 0;
        gshReduceKernelLevel1Cfg.dG.z = 0;
        gshReduceKernelLevel1Cfg.dB.x = 0;
        gshReduceKernelLevel1Cfg.dB.y = 0;
        gshReduceKernelLevel1Cfg.dB.z = 0;
    }
    
    // Working memory for GSH coefficient calculation    
    gshPerBlockSums.resize(nGSHBlocks);
}

// WARNING: will modify the list of crystals to add padding crystals
template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::doSetup(std::vector<SpectralCrystalCUDA<T>>& crystals, const CrystalPropertiesCUDA<T, N>& crystalProps) {
    // Establish the CUDA context
    CUDA_CHK(cudaFree(0));
    
    // Direct device copies
    nCrystals = crystals.size();
    nCrystalPairs = nCrystals/2;
    if (nCrystals % 2 != 0) {
        nCrystalPairs++;
    }
    
    // Set up GPU configuration and working memory
    doGPUSetup();
    
    // Pad out the number of crystals to match the GPU configuration
    // This is for cases where the number of crystals doesn't fit neatly into the 
    // block size. The solver will operate on the additional crystals, but
    // not use them in subsequent calculations.
    int nPaddingCrystals = nCrystals%(2*stepKernelCfg.dB.x);
    for (int i=0; i<nPaddingCrystals; i++) {
        crystals.push_back(crystals[0]);
    }
    
    // Device copies of problem variables
    crystalsD = crystals;  
    crystalPropsD = makeDeviceCopySharedPtr(crystalProps);
    TCauchyGlobalD.resize(1);
    TCauchyGlobalD[0] = this->TCauchyGlobalH;
}

// WARNING: will modify the list of crystals to add padding crystals
template <typename T, unsigned int N>
SpectralPolycrystalCUDA<T,N>::SpectralPolycrystalCUDA(std::vector<SpectralCrystalCUDA<T>>& crystals, const CrystalPropertiesCUDA<T, N>& crystalProps, const SpectralDatabase<T>& dbIn){    
    // Not using unified database
    useUnifiedDB = false;
    
    // Do general setup
    this->doSetup(crystals, crystalProps);
    
    // Set up database
    std::vector<SpectralDatasetID> dsetIDs = defaultCrystalSpectralDatasetIDs();
    dbH = SpectralDatabaseCUDA<T,4>(dbIn, dsetIDs);    
    dbD = makeDeviceCopySharedPtr(this->dbH);
    
    // Get memory usage
    maxMemUsedGB = getUsedMemoryGB();
}

template <typename T, unsigned int N>
SpectralPolycrystalCUDA<T,N>::SpectralPolycrystalCUDA(std::vector<SpectralCrystalCUDA<T>>& crystals, const CrystalPropertiesCUDA<T, N>& crystalProps, const SpectralDatabaseUnified<T>& dbIn){
    // Using unified database
    useUnifiedDB = true;
    
    // Do general setup
    this->doSetup(crystals, crystalProps);
    
    // Set up database
    std::vector<SpectralDatasetID> dsetIDs = defaultCrystalSpectralDatasetIDs();    
    dbUnifiedH = SpectralDatabaseUnifiedCUDA<T,4,9>(dbIn, dsetIDs);    
    dbUnifiedD = makeDeviceCopySharedPtr(this->dbUnifiedH);
    
    // Get memory usage
    maxMemUsedGB = getUsedMemoryGB();
}

template <typename T>
__device__ Tensor2CUDA<T,3,3> getSigmaPrime(T sigmaScaling, T *dbVars) {
    Tensor2CUDA<T,3,3> sigmaPrime;
    
    // Only the upper triangular terms of sigmaPrime
    sigmaPrime(0,0) = sigmaScaling*dbVars[SIGMA00];
    sigmaPrime(1,1) = sigmaScaling*dbVars[SIGMA11];
    sigmaPrime(2,2) = -sigmaPrime(0,0) -sigmaPrime(1,1);//deviatoric component
    sigmaPrime(1,2) = sigmaScaling*dbVars[SIGMA12];
    sigmaPrime(0,2) = sigmaScaling*dbVars[SIGMA02];
    sigmaPrime(0,1) = sigmaScaling*dbVars[SIGMA01];    
    
    // Symmetric terms
    sigmaPrime(2,1) = sigmaScaling*sigmaPrime(1,2);
    sigmaPrime(2,0) = sigmaScaling*sigmaPrime(0,2);
    sigmaPrime(1,0) = sigmaScaling*sigmaPrime(0,1);
    
    // Return
    return sigmaPrime;
}

template <typename T>
__device__ Tensor2AsymmCUDA<T,3> getWp(T WpScaling, T*dbVars) {
    Tensor2AsymmCUDA<T,3> Wp;
    
    // Only the terms (0,1), (1,2) and (0,2)
    // Anti-symmetric terms are handled internally by Tensor2AsymmCUDA
    Wp.setVal(0,1,WpScaling*dbVars[WP01]);
    Wp.setVal(1,2,WpScaling*dbVars[WP12]);
    Wp.setVal(0,2,WpScaling*dbVars[WP02]);
    
    // Return
    return Wp;
}

// Step kernel
template<typename T, unsigned int N>
__global__ void SPECTRAL_POLYCRYSTAL_STEP(unsigned int nCrystals, SpectralCrystalCUDA<T>* crystals, CrystalPropertiesCUDA<T,N>* props, 
Tensor2CUDA<T,3,3> RStretchingTensor, Tensor2AsymmCUDA<T,3> WNext, T theta, T strainRate, T dt, SpectralDatabaseCUDA<T,4>* db, Tensor2CUDA<T,3,3> *TCauchyPerBlockSums) 
{
    // Get crystal index
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    // Shared memory for IDFT
    #ifdef DEBUG_BUILD // shared memory is limited for debugging
        const unsigned int nSharedSpectral = 128;
    #else
        const unsigned int nSharedSpectral = 1024;
    #endif  
    __shared__ SpectralCoordCUDA<4> sharedCoords[nSharedSpectral];
    __shared__ SpectralCoeffCUDA<T> sharedCoeffs[nSharedSpectral];
    
    // If out of bounds, go through motions of calculation, but don't update at end
    bool doUpdateCrystalState = true;
    if (idx > nCrystals-1) {
        idx = nCrystals-1;
        doUpdateCrystalState = false;
    }
    
    // Get copy of our crystal from global memory
    SpectralCrystalCUDA<T> crystal = crystals[idx];
   
    // The rotation that transforms the template crystal to have the same orientation as this one
    // First, the rotation to get to the initial configuration: init.crystalRotation
    // Second, the further rotation caused by the deformation: RStar
    Tensor2CUDA<T,3,3> R = EulerZXZRotationMatrixCUDA(crystal.angles);
    
    // Transform into the stretching tensor frame
    R = RStretchingTensor.trans()*R;
    
    // Euler angles
    EulerAngles<T> angles = getEulerZXZAngles(R);
    
    // There are possible branches in getEulerZXZAngles, so sync threads here
    // to head off divergence.
    __syncthreads();

    // Database coordinate
    T gridPos[4] = {angles.alpha, angles.beta, angles.gamma, theta};
    unsigned int spatialCoord[4];
    T *gridStarts = db->getGridStarts();
    T *gridSteps = db->getGridSteps();
    for (unsigned int i=0; i<4; i++) {
        spatialCoord[i] = (unsigned int) ((gridPos[i] - gridStarts[i])/gridSteps[i]);
    }
    
    // Variables to fetch
    Tensor2CUDA<T,3,3> sigmaPrimeNext;
    Tensor2AsymmCUDA<T,3> WpNext;
    T gammaNext;
    
    // Gamma
    T gammaScaling = strainRate;
    gammaNext = gammaScaling*db->getIDFTRealDShared(GAMMA, spatialCoord, nSharedSpectral, sharedCoords, sharedCoeffs);
    
    // Update slip deformation resistance
    crystal.s = slipDeformationResistanceStepSpectralSolver(props, crystal.s, gammaNext, dt);
    
    // Sigma
    T sigmaScaling = (crystal.s*powIntr(fabs(strainRate), props->m));
    
    // Only the upper triangular terms of sigmaPrime
    sigmaPrimeNext(0,0) = sigmaScaling*db->getIDFTRealDShared(SIGMA00, spatialCoord, nSharedSpectral, sharedCoords, sharedCoeffs);
    sigmaPrimeNext(1,1) = sigmaScaling*db->getIDFTRealDShared(SIGMA11, spatialCoord, nSharedSpectral, sharedCoords, sharedCoeffs);
    sigmaPrimeNext(2,2) = -sigmaPrimeNext(0,0) -sigmaPrimeNext(1,1);//deviatoric component
    sigmaPrimeNext(1,2) = sigmaScaling*db->getIDFTRealDShared(SIGMA12, spatialCoord, nSharedSpectral, sharedCoords, sharedCoeffs);
    sigmaPrimeNext(0,2) = sigmaScaling*db->getIDFTRealDShared(SIGMA02, spatialCoord, nSharedSpectral, sharedCoords, sharedCoeffs);
    sigmaPrimeNext(0,1) = sigmaScaling*db->getIDFTRealDShared(SIGMA01, spatialCoord, nSharedSpectral, sharedCoords, sharedCoeffs);    
    
    // Symmetric terms
    sigmaPrimeNext(2,1) = sigmaScaling*sigmaPrimeNext(1,2);
    sigmaPrimeNext(2,0) = sigmaScaling*sigmaPrimeNext(0,2);
    sigmaPrimeNext(1,0) = sigmaScaling*sigmaPrimeNext(0,1);    
 
    // Wp
    T WpScaling = strainRate;
    
    // Only the terms (0,1), (1,2) and (0,2)
    WpNext.setVal(0,1,WpScaling*db->getIDFTRealDShared(WP01, spatialCoord, nSharedSpectral, sharedCoords, sharedCoeffs));
    WpNext.setVal(1,2,WpScaling*db->getIDFTRealDShared(WP12, spatialCoord, nSharedSpectral, sharedCoords, sharedCoeffs));
    WpNext.setVal(0,2,WpScaling*db->getIDFTRealDShared(WP02, spatialCoord, nSharedSpectral, sharedCoords, sharedCoeffs));
    
    // Transform into lab frame
    Tensor2CUDA<T,3,3> TCauchy = transformOutOfFrame(sigmaPrimeNext, RStretchingTensor);
    Tensor2CUDA<T,3,3> WpNextLab = transformOutOfFrame(WpNext, RStretchingTensor);
    
    // Update lattice rotation tensor
    Tensor2CUDA<T,3,3> WStarNext = WNext - WpNextLab;
    Tensor2CUDA<T,3,3> RStar = EulerZXZRotationMatrixCUDA(crystal.angles);
    Tensor2CUDA<T,3,3> RStarNext = RStar + WStarNext*RStar*dt;
    
    // Update crystal rotations
    crystal.angles = getEulerZXZAngles(RStarNext);
    
    // There are possible branches in getEulerZXZAngles, so sync threads here
    // to head off divergence.
    __syncthreads();
    
    // Add up the Cauchy stresses for this block    
    Tensor2CUDA<T,3,3> TCauchyBlockSum;
    if (doUpdateCrystalState) {
        TCauchyBlockSum = TCauchy;
    }
    __syncthreads();
    TCauchyBlockSum = blockReduceSumTensor2(TCauchyBlockSum);
    if (threadIdx.x==0) {
        TCauchyPerBlockSums[blockIdx.x]=TCauchyBlockSum;
    }

    // Restore crystal to global memory
    if (doUpdateCrystalState) {
        crystals[idx] = crystal;
    }
}

// Step kernel
template<typename T, unsigned int N, unsigned int P>
__global__ void SPECTRAL_POLYCRYSTAL_STEP_UNIFIED(unsigned int nCrystalPairs, SpectralCrystalCUDA<T>* crystals, CrystalPropertiesCUDA<T,N>* props, 
Tensor2CUDA<T,3,3> RStretchingTensor, Tensor2AsymmCUDA<T,3> WNext, T theta, T strainRate, T dt, SpectralDatabaseUnifiedCUDA<T,4,P>* db, Tensor2CUDA<T,3,3> *TCauchyPerBlockSums) 
{
    // Get crystal index
    unsigned int pairIdx = blockDim.x*blockIdx.x + threadIdx.x;
    
    // Shared memory for IDFT
    #ifdef DEBUG_BUILD // shared memory is limited for debugging
        const unsigned int nSharedSpectral = (8/sizeof(T))*16;
    #else
        const unsigned int nSharedSpectral = (8/sizeof(T))*128;
    #endif        
    __shared__ SpectralDataUnifiedCUDA<T,4,P> sharedData[nSharedSpectral];
    
    // If out of bounds, go through motions of calculation, but don't update at end
    bool doUpdateCrystalStates = true;
    if (pairIdx > nCrystalPairs-1) {
        pairIdx = nCrystalPairs-1;
        doUpdateCrystalStates = false;
    }
    
    // Indices of each crystal in the pair
    unsigned int idx0 = 2*pairIdx;
    unsigned int idx1 = idx0+1;
    
    // Get copies of the crystals from global memory
    SpectralCrystalCUDA<T> crystal0 = crystals[idx0];
    SpectralCrystalCUDA<T> crystal1 = crystals[idx1];
    
    // Get the correct database coordinate based on the crystal orientations
    unsigned int dbCoord0[4];
    getSpectralCrystalDatabaseCoordinate(crystal0, db, RStretchingTensor, theta, &(dbCoord0[0]));
    unsigned int dbCoord1[4];
    getSpectralCrystalDatabaseCoordinate(crystal1, db, RStretchingTensor, theta, &(dbCoord1[0]));
    
    // There are possible branches in the coordinate fetch, so sync threads here
    // to head off divergence
    __syncthreads();
    
    // Fetch the variables
    T dbVars0[P];
    T dbVars1[P];
    db->getIDFTRealDSharedPair(dbCoord0, dbVars0, dbCoord1, dbVars1, nSharedSpectral, sharedData);
    
    // Gamma
    T gammaScaling = strainRate;
    T gammaNext0 = gammaScaling*dbVars0[GAMMA];
    T gammaNext1 = gammaScaling*dbVars1[GAMMA];
    
    // Update slip deformation resistance
    crystal0.s = slipDeformationResistanceStepSpectralSolver(props, crystal0.s, gammaNext0, dt);
    crystal1.s = slipDeformationResistanceStepSpectralSolver(props, crystal1.s, gammaNext1, dt);
    
    // Sigma
    T sigmaScaling0 = (crystal0.s*powIntr(fabs(strainRate), props->m));
    T sigmaScaling1 = (crystal1.s*powIntr(fabs(strainRate), props->m));
    Tensor2CUDA<T,3,3> sigmaPrimeNext0 = transformOutOfFrame(getSigmaPrime(sigmaScaling0, dbVars0), RStretchingTensor);
    Tensor2CUDA<T,3,3> sigmaPrimeNext1 = transformOutOfFrame(getSigmaPrime(sigmaScaling1, dbVars1), RStretchingTensor);
 
    // Wp
    T WpScaling = strainRate;
    Tensor2CUDA<T,3,3> WpNext0 = transformOutOfFrame(getWp(WpScaling, dbVars0), RStretchingTensor);
    Tensor2CUDA<T,3,3> WpNext1 = transformOutOfFrame(getWp(WpScaling, dbVars1), RStretchingTensor);
    
    // Fetch current rotational component of deformation
    Tensor2CUDA<T,3,3> RStar0 = EulerZXZRotationMatrixCUDA(crystal0.angles);
    Tensor2CUDA<T,3,3> RStar1 = EulerZXZRotationMatrixCUDA(crystal1.angles);
    
    // Update lattice rotation tensor
    Tensor2CUDA<T,3,3> WStarNext0 = WNext - WpNext0;
    Tensor2CUDA<T,3,3> WStarNext1 = WNext - WpNext1;
    Tensor2CUDA<T,3,3> RStarNext0 = RStar0 + WStarNext0*RStar0*dt;
    Tensor2CUDA<T,3,3> RStarNext1 = RStar1 + WStarNext1*RStar1*dt;
    
    // Update crystal rotations
    crystal0.angles = getEulerZXZAngles(RStarNext0);
    crystal1.angles = getEulerZXZAngles(RStarNext1);
    
    // There are possible branches in getEulerZXZAngles, so sync threads here
    // to head off divergence.
    __syncthreads();
    
    // Add up Cauchy stress for this thread
    Tensor2CUDA<T,3,3> pairTCauchySum;
    pairTCauchySum += sigmaPrimeNext0;
    pairTCauchySum += sigmaPrimeNext1;
    
    // Add up the Cauchy stresses for this block
    Tensor2CUDA<T,3,3> TCauchyBlockSum;
    if (doUpdateCrystalStates) {
        TCauchyBlockSum = pairTCauchySum;
    }
    __syncthreads();
    TCauchyBlockSum = blockReduceSumTensor2(TCauchyBlockSum);
    if (threadIdx.x==0) {
        TCauchyPerBlockSums[blockIdx.x]=TCauchyBlockSum;
    }

    // Restore crystal to global memory
    if (doUpdateCrystalStates) {
        crystals[idx0] = crystal0;
        crystals[idx1] = crystal1;
    }
}

// Average kernel
/**
 * @brief Get the average Cauchy stress.
 * @details Deprecated in favour of the fast reduction kernels.
 * @todo remove
 * @param F_next
 * @param L_next
 * @param dt
 */
template<typename T>
__global__ void GET_AVERAGE_TCAUCHY(unsigned int nCrystals, const SpectralCrystalCUDA<T>* crystals, Tensor2CUDA<T,3,3> *TCauchyGlobal) {
    // Get absolute thread index
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx > 1) return;
    
    // Zero the global memory
    for (unsigned int i=0; i<3; i++) {
        for (unsigned int j=0; j<3; j++) {
            (*TCauchyGlobal)(i,j) = (T)0.0;
        }
    }
    
    // Add it up
    for (unsigned int iCrystal=0; iCrystal<nCrystals; iCrystal++) {
        (*TCauchyGlobal) += crystals[iCrystal].TCauchy;
    }
    
    // Average
    *TCauchyGlobal /= (T)nCrystals;    
}

/**
 * @brief Get the orientation densities from Generalized Spherical Harmonic coefficients.
 * @detail Orientations are contained in crystals.
 * @param crystals
 * @param ncrystals
 * @param coeffsPerBlockSums
 */
template<typename T>
__global__ void GET_ODF_FROM_GSH(const GSHCoeffsCUDA<T>* coeffsPtr, const SpectralCrystalCUDA<T>* crystals, unsigned int ncrystals, T* densities) {
    // Get absolute crystal/thread index
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    bool doAddContribution = true;
    if (idx > ncrystals-1) {
        idx = ncrystals-1;
        doAddContribution = false;
    }
    
    // Crystal orientation
    EulerAngles<T> angles = crystals[idx].angles;
    T phi1 = angles.alpha;
    T Phi = angles.beta;
    T phi2 = angles.gamma;
    typename cuTypes<T>::complex density = make_cuComplex((T)0.0,(T)0.0);
    
    // Read in coeffs
    GSHCoeffsCUDA<T> coeffs = *coeffsPtr;
    
    // Useful constants
    typename cuTypes<T>::complex one = make_cuComplex((T)1.0,(T)0.0);
    typename cuTypes<T>::complex Iim = make_cuComplex((T)0.0,(T)1.0);
    
    // Dummy variables
    int l;
    T normFactor;
    int m, n;
    typename cuTypes<T>::complex expMult;
    typename cuTypes<T>::complex P;    
    
    /////////
    // l=0 //
    /////////
    l = 0;

    m=0, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)1.00000000000000;
    density += P*expMult*coeffs.get(l, m, n);

    /////////
    // l=1 //
    /////////
    l = 1;

    m=-1, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.5*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.707106781186548*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.5*powFull(-cosIntr(Phi) + (T)1, (T)1.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.707106781186548*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)1.0*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.707106781186548*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1);
    density += P*expMult*coeffs.get(l, m, n);

    /////////
    // l=2 //
    /////////
    l = 2;

    m=-2, n=-2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.25*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.5*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.612372435695794*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.5*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.25*powFull(-cosIntr(Phi) + (T)1, (T)2.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=-2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.5*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*((T)1.0*cosIntr(Phi) - (T)0.5)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)1.22474487139159*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*-powFull(-cosIntr(Phi) + (T)1, (T)1.0)*((T)1.0*cosIntr(Phi) + (T)0.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.5*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=-2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.612372435695794*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)1.22474487139159*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)1.5*powFull(cosIntr(Phi), (T)2) - (T)0.5;
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)1.22474487139159*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.612372435695794*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    density += P*expMult*coeffs.get(l, m, n);

    /////////
    // l=3 //
    /////////
    l = 3;

    m=-3, n=-3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.125*powFull(cosIntr(Phi) + (T)1, (T)3.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=-2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.306186217847897*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.484122918275927*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.559016994374947*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.484122918275927*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.306186217847897*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.125*powFull(-cosIntr(Phi) + (T)1, (T)3.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=-3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.306186217847897*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=-2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*((T)0.75*cosIntr(Phi) - (T)0.5)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*((T)1.18585412256314*cosIntr(Phi) - (T)0.395284707521047);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-1.36930639376292*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*-Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)1.18585412256314*cosIntr(Phi) + (T)0.395284707521047);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*((T)0.75*cosIntr(Phi) + (T)0.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.306186217847897*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=-3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.484122918275927*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=-2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*((T)1.18585412256314*cosIntr(Phi) - (T)0.395284707521047);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)1.875*powFull(cosIntr(Phi), (T)2) - (T)1.25*cosIntr(Phi) - (T)0.125);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)2.1650635094611*powFull(cosIntr(Phi), (T)2) - (T)0.433012701892219);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*((T)-1.875*powFull(cosIntr(Phi), (T)2) - (T)1.25*cosIntr(Phi) + (T)0.125);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*-Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)1.18585412256314*cosIntr(Phi) + (T)0.395284707521047);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.484122918275927*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=-3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.559016994374947*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=-2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-1.36930639376292*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)2.1650635094611*powFull(cosIntr(Phi), (T)2) - (T)0.433012701892219);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*((T)2.5*powFull(cosIntr(Phi), (T)2) - (T)1.5)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)2.1650635094611*powFull(cosIntr(Phi), (T)2) - (T)0.433012701892219);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-1.36930639376292*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.559016994374947*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    density += P*expMult*coeffs.get(l, m, n);

    /////////
    // l=4 //
    /////////
    l = 4;

    m=-4, n=-4;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.0625*powFull(cosIntr(Phi) + (T)1, (T)4.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-4, n=-3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.176776695296637*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)3.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-4, n=-2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.330718913883074*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)3.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-4, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.467707173346743*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-4, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.522912516583797*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-4, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.467707173346743*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-4, n=2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.330718913883074*powFull(-cosIntr(Phi) + (T)1, (T)3.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-4, n=3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.176776695296637*Iim*powFull(-cosIntr(Phi) + (T)1, (T)3.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    density += P*expMult*coeffs.get(l, m, n);

    m=-4, n=4;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.0625*powFull(-cosIntr(Phi) + (T)1, (T)4.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=-4;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.176776695296637*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)3.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=-3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*((T)0.5*cosIntr(Phi) - (T)0.375)*powFull(cosIntr(Phi) + (T)1, (T)3.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=-2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*((T)0.935414346693485*cosIntr(Phi) - (T)0.467707173346743)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*((T)-1.3228756555323*cosIntr(Phi) + (T)0.330718913883074)*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-1.4790199457749*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)1.3228756555323*cosIntr(Phi) + (T)0.330718913883074);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*((T)0.935414346693485*cosIntr(Phi) + (T)0.467707173346743)*sqrtIntr(cosIntr(Phi) + (T)1);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*-powFull(-cosIntr(Phi) + (T)1, (T)3.0)*((T)0.5*cosIntr(Phi) + (T)0.375);
    density += P*expMult*coeffs.get(l, m, n);

    m=-3, n=4;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.176776695296637*Iim*powFull(-cosIntr(Phi) + (T)1, (T)3.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=-4;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.330718913883074*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)3.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=-3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*((T)0.935414346693485*cosIntr(Phi) - (T)0.467707173346743)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=-2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*powFull(cosIntr(Phi) + (T)1, (T)2.0)*((T)1.75*powFull(cosIntr(Phi), (T)2) - (T)1.75*cosIntr(Phi) + (T)0.25);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*((T)2.47487373415292*powFull(cosIntr(Phi), (T)2) - (T)1.23743686707646*cosIntr(Phi) - (T)0.176776695296637);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)-2.76699295264733*powFull(cosIntr(Phi), (T)2) + (T)0.395284707521047);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)-2.47487373415292*powFull(cosIntr(Phi), (T)2) - (T)1.23743686707646*cosIntr(Phi) + (T)0.176776695296637);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*((T)1.75*powFull(cosIntr(Phi), (T)2) + (T)1.75*cosIntr(Phi) + (T)0.25);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*((T)0.935414346693485*cosIntr(Phi) + (T)0.467707173346743)*sqrtIntr(cosIntr(Phi) + (T)1);
    density += P*expMult*coeffs.get(l, m, n);

    m=-2, n=4;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.330718913883074*powFull(-cosIntr(Phi) + (T)1, (T)3.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=-4;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-0.467707173346743*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=-3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*((T)-1.3228756555323*cosIntr(Phi) + (T)0.330718913883074)*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=-2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*((T)2.47487373415292*powFull(cosIntr(Phi), (T)2) - (T)1.23743686707646*cosIntr(Phi) - (T)0.176776695296637);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)3.5*powFull(cosIntr(Phi), (T)3) - (T)2.625*powFull(cosIntr(Phi), (T)2) - (T)0.75*cosIntr(Phi) + (T)0.375);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)3.91311896062463*powFull(cosIntr(Phi), (T)2) - (T)1.67705098312484)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*((T)-3.5*powFull(cosIntr(Phi), (T)3) - (T)2.625*powFull(cosIntr(Phi), (T)2) + (T)0.75*cosIntr(Phi) + (T)0.375);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)-2.47487373415292*powFull(cosIntr(Phi), (T)2) - (T)1.23743686707646*cosIntr(Phi) + (T)0.176776695296637);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)1.3228756555323*cosIntr(Phi) + (T)0.330718913883074);
    density += P*expMult*coeffs.get(l, m, n);

    m=-1, n=4;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.467707173346743*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=-4;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.522912516583797*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=-3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-1.4790199457749*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=-2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)-2.76699295264733*powFull(cosIntr(Phi), (T)2) + (T)0.395284707521047);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=-1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)3.91311896062463*powFull(cosIntr(Phi), (T)2) - (T)1.67705098312484)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=0;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)4.375*powFull(cosIntr(Phi), (T)4) - (T)3.75*powFull(cosIntr(Phi), (T)2) + (T)0.375;
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=1;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)3.91311896062463*powFull(cosIntr(Phi), (T)2) - (T)1.67705098312484)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=2;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)-2.76699295264733*powFull(cosIntr(Phi), (T)2) + (T)0.395284707521047);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=3;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)-1.4790199457749*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*cosIntr(Phi);
    density += P*expMult*coeffs.get(l, m, n);

    m=0, n=4;
    expMult = expIntr(Iim*(n*phi1+m*phi2));
    P = one*(T)0.522912516583797*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    density += P*expMult*coeffs.get(l, m, n);
    
    // Scale and store density
    __syncthreads();
    if (doAddContribution) {
        densities[idx] = density.x/ncrystals;
    }
}   

/**
 * @brief Get the generalized spherical harmonic coefficients from crystal orientations
 * @param crystals Active extrinsic ZXZ Euler Angles
 * @param coeffs The GSH coefficients
 */
template<typename T>
__global__ void GET_GSH_FROM_ORIENTATIONS(const SpectralCrystalCUDA<T>* crystals, unsigned int ncrystals, GSHCoeffsCUDA<T>* coeffsPerBlockSums) {
    // Get absolute crystal/thread index
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    bool doAddContribution = true;
    if (idx > ncrystals-1) {
        idx = ncrystals-1;
        doAddContribution = false;
    }
    
    // Crystal orientation
    EulerAngles<T> angles = crystals[idx].angles;
    T phi1 = angles.alpha;
    T Phi = angles.beta;
    T phi2 = angles.gamma;    
    
    // Calculate the coefficients
    GSHCoeffsCUDA<T> coeffs;
    typename cuTypes<T>::complex one = make_cuComplex((T)1.0,(T)0.0);
    typename cuTypes<T>::complex Iim = make_cuComplex((T)0.0,(T)1.0);
    
    // l=0
    coeffs.set(0, 0, 0, make_cuComplex((T)1.0, (T)0.0));
    
    /////////
    // l=1 //
    /////////
    int l=1; 
    T normFactor = (T)2.*l+(T)1.;
    
    int m=-1, n=-1;   
    typename cuTypes<T>::complex expMult = expIntr(make_cuComplex((T)0., -n*phi1-m*phi2));
    typename cuTypes<T>::complex P = make_cuComplex((T)0.5*((T)1.+cosIntr(Phi)), (T)0.);
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    m=-1, n=0;   
    expMult = expIntr(make_cuComplex((T)0., -n*phi1-m*phi2));
    P = make_cuComplex((T)0., ((T)1./sqrtIntr((T)2.))*sinIntr(Phi));
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    m=-1, n=1;   
    expMult = expIntr(make_cuComplex((T)0., -n*phi1-m*phi2));
    P = make_cuComplex((T)0.5*(cosIntr(Phi)-1), (T)0.);
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    m=0, n=-1;   
    expMult = expIntr(make_cuComplex((T)0., -n*phi1-m*phi2));
    P = make_cuComplex((T)0., ((T)1./sqrtIntr((T)2.))*sinIntr(Phi));
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    m=0, n=0;   
    expMult = expIntr(make_cuComplex((T)0., -n*phi1-m*phi2));
    P = make_cuComplex(cosIntr(Phi), (T)0.);
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    /////////
    // l=2 //
    /////////
    l=2; 
    normFactor = (T)2.*l+(T)1.;
    
    m=-2, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.25*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.5*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.612372435695794*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.5*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.25*powFull(-cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.5*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*((T)1.0*cosIntr(Phi) - (T)0.5)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)1.22474487139159*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*-powFull(-cosIntr(Phi) + (T)1, (T)1.0)*((T)1.0*cosIntr(Phi) + (T)0.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.5*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.612372435695794*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)1.22474487139159*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)1.5*powFull(cosIntr(Phi), (T)2) - (T)0.5;
    coeffs.set(l, m, n, normFactor*P*expMult);

    /////////
    // l=3 //
    /////////
    l=3; 
    normFactor = (T)2.*l+(T)1.;
    
    m=-3, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.125*powFull(cosIntr(Phi) + (T)1, (T)3.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.306186217847897*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.484122918275927*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.559016994374947*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.484122918275927*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.306186217847897*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.125*powFull(-cosIntr(Phi) + (T)1, (T)3.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.306186217847897*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*((T)0.75*cosIntr(Phi) - (T)0.5)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*((T)1.18585412256314*cosIntr(Phi) - (T)0.395284707521047);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-1.36930639376292*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*-Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)1.18585412256314*cosIntr(Phi) + (T)0.395284707521047);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*((T)0.75*cosIntr(Phi) + (T)0.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.306186217847897*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.484122918275927*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*((T)1.18585412256314*cosIntr(Phi) - (T)0.395284707521047);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)1.875*powFull(cosIntr(Phi), (T)2) - (T)1.25*cosIntr(Phi) - (T)0.125);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)2.1650635094611*powFull(cosIntr(Phi), (T)2) - (T)0.433012701892219);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*((T)-1.875*powFull(cosIntr(Phi), (T)2) - (T)1.25*cosIntr(Phi) + (T)0.125);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*-Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)1.18585412256314*cosIntr(Phi) + (T)0.395284707521047);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.484122918275927*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.559016994374947*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-1.36930639376292*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)2.1650635094611*powFull(cosIntr(Phi), (T)2) - (T)0.433012701892219);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*((T)2.5*powFull(cosIntr(Phi), (T)2) - (T)1.5)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    /////////
    // l=4 //
    /////////
    l=4; 
    normFactor = (T)2.*l+(T)1.;
    
    m=-4, n=-4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.0625*powFull(cosIntr(Phi) + (T)1, (T)4.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.176776695296637*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)3.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.330718913883074*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)3.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.467707173346743*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.522912516583797*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.467707173346743*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.330718913883074*powFull(-cosIntr(Phi) + (T)1, (T)3.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.176776695296637*Iim*powFull(-cosIntr(Phi) + (T)1, (T)3.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.0625*powFull(-cosIntr(Phi) + (T)1, (T)4.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=-4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.176776695296637*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)3.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*((T)0.5*cosIntr(Phi) - (T)0.375)*powFull(cosIntr(Phi) + (T)1, (T)3.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*((T)0.935414346693485*cosIntr(Phi) - (T)0.467707173346743)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*((T)-1.3228756555323*cosIntr(Phi) + (T)0.330718913883074)*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-1.4790199457749*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)1.3228756555323*cosIntr(Phi) + (T)0.330718913883074);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*((T)0.935414346693485*cosIntr(Phi) + (T)0.467707173346743)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*-powFull(-cosIntr(Phi) + (T)1, (T)3.0)*((T)0.5*cosIntr(Phi) + (T)0.375);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.176776695296637*Iim*powFull(-cosIntr(Phi) + (T)1, (T)3.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.330718913883074*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)3.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*((T)0.935414346693485*cosIntr(Phi) - (T)0.467707173346743)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(cosIntr(Phi) + (T)1, (T)2.0)*((T)1.75*powFull(cosIntr(Phi), (T)2) - (T)1.75*cosIntr(Phi) + (T)0.25);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*((T)2.47487373415292*powFull(cosIntr(Phi), (T)2) - (T)1.23743686707646*cosIntr(Phi) - (T)0.176776695296637);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)-2.76699295264733*powFull(cosIntr(Phi), (T)2) + (T)0.395284707521047);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)-2.47487373415292*powFull(cosIntr(Phi), (T)2) - (T)1.23743686707646*cosIntr(Phi) + (T)0.176776695296637);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*((T)1.75*powFull(cosIntr(Phi), (T)2) + (T)1.75*cosIntr(Phi) + (T)0.25);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*((T)0.935414346693485*cosIntr(Phi) + (T)0.467707173346743)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.330718913883074*powFull(-cosIntr(Phi) + (T)1, (T)3.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.467707173346743*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*((T)-1.3228756555323*cosIntr(Phi) + (T)0.330718913883074)*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*((T)2.47487373415292*powFull(cosIntr(Phi), (T)2) - (T)1.23743686707646*cosIntr(Phi) - (T)0.176776695296637);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)3.5*powFull(cosIntr(Phi), (T)3) - (T)2.625*powFull(cosIntr(Phi), (T)2) - (T)0.75*cosIntr(Phi) + (T)0.375);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)3.91311896062463*powFull(cosIntr(Phi), (T)2) - (T)1.67705098312484)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*((T)-3.5*powFull(cosIntr(Phi), (T)3) - (T)2.625*powFull(cosIntr(Phi), (T)2) + (T)0.75*cosIntr(Phi) + (T)0.375);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)-2.47487373415292*powFull(cosIntr(Phi), (T)2) - (T)1.23743686707646*cosIntr(Phi) + (T)0.176776695296637);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)1.3228756555323*cosIntr(Phi) + (T)0.330718913883074);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.467707173346743*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.522912516583797*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-1.4790199457749*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)-2.76699295264733*powFull(cosIntr(Phi), (T)2) + (T)0.395284707521047);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)3.91311896062463*powFull(cosIntr(Phi), (T)2) - (T)1.67705098312484)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)4.375*powFull(cosIntr(Phi), (T)4) - (T)3.75*powFull(cosIntr(Phi), (T)2) + (T)0.375;
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    // Add up coefficients
    __syncthreads();
    GSHCoeffsCUDA<T> coeffsBlockSum;
    if (doAddContribution) {
        coeffsBlockSum = coeffs;
    }
    coeffsBlockSum = blockReduceSumGSHCoeffs(coeffsBlockSum);
    if (threadIdx.x==0) {
        coeffsPerBlockSums[blockIdx.x] = coeffsBlockSum;
    }
}

/**
 * @brief Get the generalized spherical harmonic coefficients from crystal orientations
 * @detail Weighted with some crystal densities. That is, the density associated
 * with a crystal represents the probability of finding a crystal in a
 * neighbourhood of that orientation. For this collection of crystals and
 * densities to represent an orientation distribution function, the crystal
 * orientations must be uniformly distributed in orientation space, and the
 * densities must sum to one.
 * @param crystals Active extrinsic ZXZ Euler Angles
 * @param coeffs The GSH coefficients
 */
template<typename T>
__global__ void GET_GSH_FROM_ORIENTATIONS_AND_DENSITIES(const SpectralCrystalCUDA<T>* crystals, unsigned int ncrystals, const T* densities, GSHCoeffsCUDA<T>* coeffsPerBlockSums) {
    // Get absolute crystal/thread index
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    bool doAddContribution = true;
    if (idx > ncrystals-1) {
        idx = ncrystals-1;
        doAddContribution = false;
    }
    
    // Crystal orientation
    EulerAngles<T> angles = crystals[idx].angles;
    T phi1 = angles.alpha;
    T Phi = angles.beta;
    T phi2 = angles.gamma;
    
    // Calculate the coefficients
    GSHCoeffsCUDA<T> coeffs;
    typename cuTypes<T>::complex one = make_cuComplex((T)1.0,(T)0.0);
    typename cuTypes<T>::complex Iim = make_cuComplex((T)0.0,(T)1.0);
    
    // l=0
    coeffs.set(0, 0, 0, make_cuComplex((T)1.0, (T)0.0));
    
    /////////
    // l=1 //
    /////////
    int l=1; 
    T normFactor = (T)2.*l+(T)1.;
    
    int m=-1, n=-1;   
    typename cuTypes<T>::complex expMult = expIntr(make_cuComplex((T)0., -n*phi1-m*phi2));
    typename cuTypes<T>::complex P = make_cuComplex((T)0.5*((T)1.+cosIntr(Phi)), (T)0.);
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    m=-1, n=0;   
    expMult = expIntr(make_cuComplex((T)0., -n*phi1-m*phi2));
    P = make_cuComplex((T)0., ((T)1./sqrtIntr((T)2.))*sinIntr(Phi));
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    m=-1, n=1;   
    expMult = expIntr(make_cuComplex((T)0., -n*phi1-m*phi2));
    P = make_cuComplex((T)0.5*(cosIntr(Phi)-1), (T)0.);
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    m=0, n=-1;   
    expMult = expIntr(make_cuComplex((T)0., -n*phi1-m*phi2));
    P = make_cuComplex((T)0., ((T)1./sqrtIntr((T)2.))*sinIntr(Phi));
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    m=0, n=0;   
    expMult = expIntr(make_cuComplex((T)0., -n*phi1-m*phi2));
    P = make_cuComplex(cosIntr(Phi), (T)0.);
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    /////////
    // l=2 //
    /////////
    l=2; 
    normFactor = (T)2.*l+(T)1.;
    
    m=-2, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.25*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.5*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.612372435695794*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.5*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.25*powFull(-cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.5*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*((T)1.0*cosIntr(Phi) - (T)0.5)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)1.22474487139159*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*-powFull(-cosIntr(Phi) + (T)1, (T)1.0)*((T)1.0*cosIntr(Phi) + (T)0.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.5*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.612372435695794*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)1.22474487139159*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)1.5*powFull(cosIntr(Phi), (T)2) - (T)0.5;
    coeffs.set(l, m, n, normFactor*P*expMult);

    /////////
    // l=3 //
    /////////
    l=3; 
    normFactor = (T)2.*l+(T)1.;
    
    m=-3, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.125*powFull(cosIntr(Phi) + (T)1, (T)3.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.306186217847897*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.484122918275927*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.559016994374947*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.484122918275927*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.306186217847897*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.125*powFull(-cosIntr(Phi) + (T)1, (T)3.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.306186217847897*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*((T)0.75*cosIntr(Phi) - (T)0.5)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*((T)1.18585412256314*cosIntr(Phi) - (T)0.395284707521047);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-1.36930639376292*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*-Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)1.18585412256314*cosIntr(Phi) + (T)0.395284707521047);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*((T)0.75*cosIntr(Phi) + (T)0.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.306186217847897*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.484122918275927*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*((T)1.18585412256314*cosIntr(Phi) - (T)0.395284707521047);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)1.875*powFull(cosIntr(Phi), (T)2) - (T)1.25*cosIntr(Phi) - (T)0.125);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)2.1650635094611*powFull(cosIntr(Phi), (T)2) - (T)0.433012701892219);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*((T)-1.875*powFull(cosIntr(Phi), (T)2) - (T)1.25*cosIntr(Phi) + (T)0.125);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*-Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)1.18585412256314*cosIntr(Phi) + (T)0.395284707521047);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.484122918275927*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.559016994374947*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-1.36930639376292*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)2.1650635094611*powFull(cosIntr(Phi), (T)2) - (T)0.433012701892219);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*((T)2.5*powFull(cosIntr(Phi), (T)2) - (T)1.5)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    /////////
    // l=4 //
    /////////
    l=4; 
    normFactor = (T)2.*l+(T)1.;
    
    m=-4, n=-4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.0625*powFull(cosIntr(Phi) + (T)1, (T)4.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.176776695296637*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)3.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.330718913883074*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)3.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.467707173346743*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.522912516583797*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.467707173346743*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.330718913883074*powFull(-cosIntr(Phi) + (T)1, (T)3.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.176776695296637*Iim*powFull(-cosIntr(Phi) + (T)1, (T)3.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-4, n=4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.0625*powFull(-cosIntr(Phi) + (T)1, (T)4.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=-4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.176776695296637*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)3.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*((T)0.5*cosIntr(Phi) - (T)0.375)*powFull(cosIntr(Phi) + (T)1, (T)3.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*((T)0.935414346693485*cosIntr(Phi) - (T)0.467707173346743)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*((T)-1.3228756555323*cosIntr(Phi) + (T)0.330718913883074)*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-1.4790199457749*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)1.3228756555323*cosIntr(Phi) + (T)0.330718913883074);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*((T)0.935414346693485*cosIntr(Phi) + (T)0.467707173346743)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*-powFull(-cosIntr(Phi) + (T)1, (T)3.0)*((T)0.5*cosIntr(Phi) + (T)0.375);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-3, n=4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.176776695296637*Iim*powFull(-cosIntr(Phi) + (T)1, (T)3.5)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.330718913883074*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)3.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*((T)0.935414346693485*cosIntr(Phi) - (T)0.467707173346743)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(cosIntr(Phi) + (T)1, (T)2.0)*((T)1.75*powFull(cosIntr(Phi), (T)2) - (T)1.75*cosIntr(Phi) + (T)0.25);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*((T)2.47487373415292*powFull(cosIntr(Phi), (T)2) - (T)1.23743686707646*cosIntr(Phi) - (T)0.176776695296637);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)-2.76699295264733*powFull(cosIntr(Phi), (T)2) + (T)0.395284707521047);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)-2.47487373415292*powFull(cosIntr(Phi), (T)2) - (T)1.23743686707646*cosIntr(Phi) + (T)0.176776695296637);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*((T)1.75*powFull(cosIntr(Phi), (T)2) + (T)1.75*cosIntr(Phi) + (T)0.25);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*((T)0.935414346693485*cosIntr(Phi) + (T)0.467707173346743)*sqrtIntr(cosIntr(Phi) + (T)1);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-2, n=4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.330718913883074*powFull(-cosIntr(Phi) + (T)1, (T)3.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-0.467707173346743*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)2.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*((T)-1.3228756555323*cosIntr(Phi) + (T)0.330718913883074)*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*((T)2.47487373415292*powFull(cosIntr(Phi), (T)2) - (T)1.23743686707646*cosIntr(Phi) - (T)0.176776695296637);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)3.5*powFull(cosIntr(Phi), (T)3) - (T)2.625*powFull(cosIntr(Phi), (T)2) - (T)0.75*cosIntr(Phi) + (T)0.375);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)3.91311896062463*powFull(cosIntr(Phi), (T)2) - (T)1.67705098312484)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*((T)-3.5*powFull(cosIntr(Phi), (T)3) - (T)2.625*powFull(cosIntr(Phi), (T)2) + (T)0.75*cosIntr(Phi) + (T)0.375);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)-2.47487373415292*powFull(cosIntr(Phi), (T)2) - (T)1.23743686707646*cosIntr(Phi) + (T)0.176776695296637);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)1.3228756555323*cosIntr(Phi) + (T)0.330718913883074);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=-1, n=4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.467707173346743*Iim*powFull(-cosIntr(Phi) + (T)1, (T)2.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-4;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)0.522912516583797*powFull(-cosIntr(Phi) + (T)1, (T)2.0)*powFull(cosIntr(Phi) + (T)1, (T)2.0);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-3;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)-1.4790199457749*Iim*powFull(-cosIntr(Phi) + (T)1, (T)1.5)*powFull(cosIntr(Phi) + (T)1, (T)1.5)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-2;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*powFull(-cosIntr(Phi) + (T)1, (T)1.0)*powFull(cosIntr(Phi) + (T)1, (T)1.0)*((T)-2.76699295264733*powFull(cosIntr(Phi), (T)2) + (T)0.395284707521047);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=-1;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*Iim*sqrtIntr(-cosIntr(Phi) + (T)1)*sqrtIntr(cosIntr(Phi) + (T)1)*((T)3.91311896062463*powFull(cosIntr(Phi), (T)2) - (T)1.67705098312484)*cosIntr(Phi);
    coeffs.set(l, m, n, normFactor*P*expMult);

    m=0, n=0;
    expMult = expIntr(-Iim*(n*phi1+m*phi2));
    P = one*(T)4.375*powFull(cosIntr(Phi), (T)4) - (T)3.75*powFull(cosIntr(Phi), (T)2) + (T)0.375;
    coeffs.set(l, m, n, normFactor*P*expMult);
    
    // Add up coefficients
    __syncthreads();
    GSHCoeffsCUDA<T> coeffsBlockSum;
    if (doAddContribution) {
        coeffsBlockSum = densities[idx]*coeffs;
    }
    coeffsBlockSum = blockReduceSumGSHCoeffs(coeffsBlockSum);
    if (threadIdx.x==0) {
        coeffsPerBlockSums[blockIdx.x] = coeffsBlockSum;
    }
}

template<typename T, unsigned int N>
__global__ void HISTOGRAM_POLES_EQUAL_AREA(unsigned int nCrystals, const SpectralCrystalCUDA<T>* crystals, VecCUDA<T,3>* planeNormalG, Tensor2CUDA<T,N,N>* histG) {
    // Get absolute thread index
    unsigned int baseIdx = blockDim.x*blockIdx.x + threadIdx.x;
    
    // Read in normal from global memory
    VecCUDA<T,3> planeNormal = *planeNormalG;

    // Maximum R value from northern hemisphere projection
    T maxR = (1.00001)*2*sinIntr(M_PI/4);
    
    // Grid stride loop over crystals
    for (unsigned int idx=baseIdx; idx<nCrystals; idx+=blockDim.x*gridDim.x) {
        // Get orientation of the crystal
        SpectralCrystalCUDA<T> crystal = crystals[idx];
        Tensor2CUDA<T,3,3> ROrientation = EulerZXZRotationMatrixCUDA(crystal.angles);
        
        // Active rotation
        VecCUDA<T,3> pole = ROrientation*planeNormal;
        VecCUDA<T,3> poleSpherical = cartesianToSpherical(pole);
        T theta = poleSpherical(1);
        T phi = poleSpherical(2);
        
        // Equal-area projection
        T R = 2*sinIntr(phi/2);
        T x, y;
        sincosIntr(theta, &y, &x);
        x *= R;
        y *= R;
        
        // Histogram index
        T xMin = -maxR;
        T xMax = maxR;
        T yMin = xMin;
        T yMax = xMax;
        T binwidthX = (xMax-xMin)/N;
        T binwidthY = (yMax-yMin)/N;
        int ix = (int) ((x-xMin)/binwidthX);
        int iy = (int) ((y-yMin)/binwidthY);
        
        // Add points to histogram
        if (ix >=0 && ix < N && iy>=0 && iy < N) {
            atomicAdd(&((*histG)(ix,iy)), 1.0);
        }
        __syncthreads();
    }    
} 

// Reset host function
template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::resetRandomOrientations(T init_s, unsigned long int seed) {
    // Create reset crystals on host
    std::vector<SpectralCrystalCUDA<T>> crystalList(this->nCrystals);
    Tensor2<T> R(3,3);
    std::mt19937 randgen(seed);    
    for (auto&& crystal : crystalList) {        
        // Initial slip-system deformation resistance and angles
        crystal.s = init_s;
        randomRotationTensorArvo1992<T>(randgen, R);
        crystal.angles = getEulerZXZAngles(R);
    }
    
    // Copy to device
    crystalsD = crystalList;

    // Reset other dependent quantities
    this->resetHistories();
}

template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::resetGivenOrientations(T init_s, const std::vector<EulerAngles<T>>& angleList) {
    // Check sizes
    if (angleList.size() != this->nCrystals) {
        std::cerr << "Number of angles supplied = " << angleList.size() << std::endl;
        std::cerr << "Number of crystals = " << this->nCrystals << std::endl;
        throw std::runtime_error("Mismatch in angles supplied vs. number of crystals.");
    }
    
    // Set angles
    std::vector<SpectralCrystalCUDA<T>> crystalList(this->nCrystals);
    for (unsigned int i=0; i<this->nCrystals; i++) {       
        crystalList[i].s = init_s;
        crystalList[i].angles = angleList[i];
    }
    
    // Copy to device
    crystalsD = crystalList;
    
    // Reset other dependent quantities
    this->resetHistories();
}

template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::resetHistories() {
    tHistory.clear();
    TCauchyHistory.clear();
    poleHistogramHistory111.clear();
    poleHistogramHistory110.clear();
    poleHistogramHistory100.clear();
    poleHistogramHistory001.clear();
    poleHistogramHistory011.clear();    
}

// Step host function
template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::step(const hpp::Tensor2<T>& F_next, const hpp::Tensor2<T>& L_next, T dt) 
{   
    this->step(L_next, dt);
}

// Step host function
template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::step(const hpp::Tensor2<T>& L_next, T dt) 
{        
    // Get stretching tensor decomposition
    StretchingTensorDecomposition<T> stretchingTensorDecomp = getStretchingTensorDecomposition(L_next); 
    T theta = stretchingTensorDecomp.theta;
    T strainRate = stretchingTensorDecomp.DNorm;
    
    // DEBUG: report on recomposed velocity gradient
//    std::cout << "L in = " << L_next << std::endl;
//    auto stretchingTensor = stretchingVelocityGradient(stretchingTensorDecomp.theta, stretchingTensorDecomp.DNorm);
//    auto LReconstructed = transformOutOfFrame(stretchingTensor, stretchingTensorDecomp.evecs);
//    std::cout << "LReconstructed = " << LReconstructed << std::endl;
    
    // The rotational component of the stretching tensor
    Tensor2CUDA<T,3,3> RStretchingTensor(stretchingTensorDecomp.evecs);
    
    // The overall spin tensor
    Tensor2AsymmCUDA<T,3> WNext = (T)0.5*(L_next-L_next.trans());
    
    // Compute the next step
    dim3 dG = stepKernelCfg.dG;
    dim3 dB = stepKernelCfg.dB;
    
    // Start solve timer
    solveTimer.start();
    
    if (useUnifiedDB) {
        SPECTRAL_POLYCRYSTAL_STEP_UNIFIED<<<dG,dB>>>(nCrystalPairs, crystalsD.data().get(), crystalPropsD.get(), RStretchingTensor, WNext, theta, strainRate, dt, dbUnifiedD.get(), TCauchyPerBlockSums.data().get());
    }
    else {
        SPECTRAL_POLYCRYSTAL_STEP<<<dG,dB>>>(nCrystals, crystalsD.data().get(), crystalPropsD.get(), RStretchingTensor, WNext, theta, strainRate, dt, dbD.get(), TCauchyPerBlockSums.data().get());    
    }
    
    // Stop solve timer
    CUDA_CHK(cudaDeviceSynchronize());
    solveTimer.stop();
    
    // Sum up the per-block stress
    // TCauchyGlobalD will now contain the global sum of the stress
    unsigned int nBlocks = stepKernelCfg.dG.x;
    
    // Single level reduction
    if (reduceKernelLevel0Cfg.dG.x <= 1) {
        BLOCK_REDUCE_KEPLER_TENSOR2<<<reduceKernelLevel0Cfg.dG, reduceKernelLevel0Cfg.dB>>>(TCauchyPerBlockSums.data().get(), TCauchyGlobalD.data().get(), nBlocks);
    }
    // Two level reduction
    else{        
        BLOCK_REDUCE_KEPLER_TENSOR2<<<reduceKernelLevel0Cfg.dG, reduceKernelLevel0Cfg.dB>>>(TCauchyPerBlockSums.data().get(), TCauchyLevel0Sums.data().get(), nBlocks);
        BLOCK_REDUCE_KEPLER_TENSOR2<<<reduceKernelLevel1Cfg.dG, reduceKernelLevel1Cfg.dB>>>(TCauchyLevel0Sums.data().get(), TCauchyGlobalD.data().get(), reduceKernelLevel0Cfg.dG.x);
    }
    
    // Sync device and host before copying across memory
    CUDA_CHK(cudaDeviceSynchronize());
    
    // Move required quantities to host
    TCauchyGlobalH = TCauchyGlobalD[0];
    TCauchyGlobalH /= (T)nCrystals;
}

template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::evolve(T tStart, T tEnd, T dt, std::function<hpp::Tensor2<T>(T t)> F_of_t, std::function<hpp::Tensor2<T>(T t)> L_of_t) {
    // Initial data
    tHistory.push_back(tStart);
    TCauchyHistory.push_back(TCauchyGlobalH);
    
    // Stepping
    unsigned int nsteps = (tEnd-tStart)/dt;
    poleHistogramHistory111.resize(nsteps+1);
    poleHistogramHistory110.resize(nsteps+1);
    poleHistogramHistory100.resize(nsteps+1);
    poleHistogramHistory001.resize(nsteps+1);
    poleHistogramHistory011.resize(nsteps+1);
    getPoleHistogram(this->poleHistogramHistory111[0], VecCUDA<T,3>{1,1,1});
    getPoleHistogram(this->poleHistogramHistory110[0], VecCUDA<T,3>{1,1,0});
    getPoleHistogram(this->poleHistogramHistory100[0], VecCUDA<T,3>{1,0,0});
    getPoleHistogram(this->poleHistogramHistory001[0], VecCUDA<T,3>{0,0,1});
    getPoleHistogram(this->poleHistogramHistory011[0], VecCUDA<T,3>{0,1,1});
    for (unsigned int i=0; i<nsteps; i++) {
        // Inputs for the next step
        T t = tStart + (i+1)*dt;
        std::cout << "t = " << t << std::endl;
        hpp::Tensor2<T> LNext = L_of_t(t);     
        hpp::Tensor2<T> FNext = F_of_t(t);
        
        // Step
        this->step(FNext, LNext, dt);
        
        // Store quantities
        tHistory.push_back(t);
        TCauchyHistory.push_back(TCauchyGlobalH);
        getPoleHistogram(this->poleHistogramHistory111[i+1], VecCUDA<T,3>{1,1,1});
        getPoleHistogram(this->poleHistogramHistory110[i+1], VecCUDA<T,3>{1,1,0});
        getPoleHistogram(this->poleHistogramHistory100[i+1], VecCUDA<T,3>{1,0,0});
        getPoleHistogram(this->poleHistogramHistory001[i+1], VecCUDA<T,3>{0,0,1});
        getPoleHistogram(this->poleHistogramHistory011[i+1], VecCUDA<T,3>{0,1,1});
    }
}

template <typename T, unsigned int N>
std::shared_ptr<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> SpectralPolycrystalCUDA<T,N>::getPoleHistogram(const VecCUDA<T,3>& pole) {
    // Histogram configuration
    const unsigned int histDim = HPP_POLE_FIG_HIST_DIM;
    CudaKernelConfig histKernelCfg = getKernelConfigMaxOccupancy(devProp, (void*)HISTOGRAM_POLES_EQUAL_AREA<T, histDim>, nCrystals);
    dim3 dG = histKernelCfg.dG;
    dim3 dB = histKernelCfg.dB;
      
    // Generate histogram
    std::shared_ptr<VecCUDA<T,3>> poleD = makeDeviceCopySharedPtr(pole);    
    std::shared_ptr<Tensor2CUDA<T,histDim,histDim>> histHSharedPtr(new Tensor2CUDA<T,histDim,histDim>);
    std::shared_ptr<Tensor2CUDA<T,histDim,histDim>> histD = makeDeviceCopySharedPtrFromPtr(histHSharedPtr.get());        
    CUDA_CHK(cudaDeviceSynchronize());
    HISTOGRAM_POLES_EQUAL_AREA<T, histDim><<<dG,dB>>>(nCrystals, crystalsD.data().get(), poleD.get(), histD.get());
    CUDA_CHK(cudaDeviceSynchronize());
    copyToHost(histD, histHSharedPtr.get());
    
    // Return
    return histHSharedPtr;
}

template <typename T, unsigned int N>
Tensor2<T> SpectralPolycrystalCUDA<T,N>::getPoleHistogram(int p0, int p1, int p2) {
    VecCUDA<T,3> pole;
    pole(0) = (T)p0;
    pole(1) = (T)p1;
    pole(2) = (T)p2;
    auto histHSharedPtr = this->getPoleHistogram(pole);
    Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM> *histHPtr = histHSharedPtr.get();
    Tensor2<T> hist(HPP_POLE_FIG_HIST_DIM, HPP_POLE_FIG_HIST_DIM);
    for (unsigned int i=0; i<HPP_POLE_FIG_HIST_DIM; i++) {
        for (unsigned int j=0; j<HPP_POLE_FIG_HIST_DIM; j++) {
            hist.setVal(i, j, histHPtr->getVal(i,j));
        }
    }
    return hist;
}

/**
 * @brief Get the Euler angles of the crystals
 */
template <typename T, unsigned int N>
std::vector<EulerAngles<T>> SpectralPolycrystalCUDA<T,N>::getEulerAnglesZXZActive() {
    thrust::host_vector<SpectralCrystalCUDA<T>> crystalsH = this->crystalsD;
    std::vector<EulerAngles<T>> anglesVec(crystalsH.size());
    for (unsigned int i=0; i<crystalsH.size(); i++) {
        anglesVec[i] = crystalsH[i].angles;
    }
    return anglesVec;
}

template <typename T, unsigned int N>
GSHCoeffsCUDA<T> SpectralPolycrystalCUDA<T,N>::getGSHCoeffs() { 
    GSHCoeffsCUDA<T> coeffsH;
    auto coeffsSumD = makeDeviceCopySharedPtr(coeffsH);
    
    // Compute coefficients
    dim3 dG = gshKernelCfg.dG;
    dim3 dB = gshKernelCfg.dB;
    GET_GSH_FROM_ORIENTATIONS<<<dG,dB>>>(crystalsD.data().get(), nCrystals, gshPerBlockSums.data().get());
    
    // Single level reduction
    unsigned int nBlocksGSH = gshKernelCfg.dG.x;
    if (gshReduceKernelLevel0Cfg.dG.x <= 1) {
        BLOCK_REDUCE_KEPLER_GSH_COEFFS<<<gshReduceKernelLevel0Cfg.dG, gshReduceKernelLevel0Cfg.dB>>>(gshPerBlockSums.data().get(), coeffsSumD.get(), nBlocksGSH);
    }
    // Two level reduction
    else{        
        BLOCK_REDUCE_KEPLER_GSH_COEFFS<<<gshReduceKernelLevel0Cfg.dG, gshReduceKernelLevel0Cfg.dB>>>(gshPerBlockSums.data().get(), gshLevel0Sums.data().get(), nBlocksGSH);
        BLOCK_REDUCE_KEPLER_GSH_COEFFS<<<gshReduceKernelLevel1Cfg.dG, gshReduceKernelLevel1Cfg.dB>>>(gshLevel0Sums.data().get(), coeffsSumD.get(), gshReduceKernelLevel0Cfg.dG.x);
    }
    
    // Sync before transfers
    cudaDeviceSynchronize();
    // Sum->integral
    coeffsH = getHostValue(coeffsSumD)/(T)nCrystals;
    
    // Return
    return coeffsH;
}

template <typename T, unsigned int N>
GSHCoeffsCUDA<T> SpectralPolycrystalCUDA<T,N>::getDensityWeightedGSH(const std::vector<T>& densities) { 
    // Check densities
    if (densities.size() != nCrystals) {
        std::cerr << "nDensities = " << densities.size() << std::endl;
        std::cerr << "nCrystals = " << nCrystals << std::endl;
        throw std::runtime_error("Mismatch between number of densities provided and number of crystals.");
    }
    T densitySum = 0.0;
    for (const auto& density : densities) {
        densitySum += density;
    }
    if (std::abs(densitySum-1.0) > 1000.0*std::numeric_limits<T>::epsilon()) {
        std::cerr << "Density sum = " << densitySum << std::endl;
        throw std::runtime_error("Densities don't sum to 1.0.");
    }
    
    // Device and host memory
    auto coeffsSumD = allocDeviceMemorySharedPtr<GSHCoeffsCUDA<T>>();
    auto densitiesD = makeDeviceCopyVecSharedPtr(densities);
    
    // Compute coefficients
    dim3 dG = gshKernelCfg.dG;
    dim3 dB = gshKernelCfg.dB;
    GET_GSH_FROM_ORIENTATIONS_AND_DENSITIES<<<dG,dB>>>(crystalsD.data().get(), nCrystals, densitiesD.get(), gshPerBlockSums.data().get());
    
    // Single level reduction
    unsigned int nBlocksGSH = gshKernelCfg.dG.x;
    if (gshReduceKernelLevel0Cfg.dG.x <= 1) {
        BLOCK_REDUCE_KEPLER_GSH_COEFFS<<<gshReduceKernelLevel0Cfg.dG, gshReduceKernelLevel0Cfg.dB>>>(gshPerBlockSums.data().get(), coeffsSumD.get(), nBlocksGSH);
    }
    // Two level reduction
    else{        
        BLOCK_REDUCE_KEPLER_GSH_COEFFS<<<gshReduceKernelLevel0Cfg.dG, gshReduceKernelLevel0Cfg.dB>>>(gshPerBlockSums.data().get(), gshLevel0Sums.data().get(), nBlocksGSH);
        BLOCK_REDUCE_KEPLER_GSH_COEFFS<<<gshReduceKernelLevel1Cfg.dG, gshReduceKernelLevel1Cfg.dB>>>(gshLevel0Sums.data().get(), coeffsSumD.get(), gshReduceKernelLevel0Cfg.dG.x);
    }
    
    // Sync before transfers
    cudaDeviceSynchronize();

    // Return
    auto coeffsH = getHostValue(coeffsSumD);
    return coeffsH;
}

template <typename T, unsigned int N>
std::vector<T> SpectralPolycrystalCUDA<T,N>::getDensitiesFromGSH(const GSHCoeffsCUDA<T> coeffsH) {
    auto coeffsD = makeDeviceCopySharedPtr(coeffsH);
    auto densitiesD = allocDeviceMemorySharedPtr<T>(nCrystals);    
    
    // Compute coefficients
    dim3 dG = gshKernelCfg.dG;
    dim3 dB = gshKernelCfg.dB;
    GET_ODF_FROM_GSH<<<dG,dB>>>(coeffsD.get(), crystalsD.data().get(), nCrystals, densitiesD.get());
    
    // Sync before transfers
    cudaDeviceSynchronize();
    
    // Copy back and return
    auto densitiesH = makeHostVecFromSharedPtr<T>(densitiesD, nCrystals);
    return densitiesH;
}

/**
 * @brief Generate a pole histogram
 * @param hist
 */
template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::getPoleHistogram(Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>& hist, const VecCUDA<T,3>& pole) {
    auto histHSharedPtr = this->getPoleHistogram(pole);
    hist = *(histHSharedPtr.get());
}

/**
 * @brief Writes out pole histograms to HDF5.
 * @details
 * @param outfile the output file
 * @param poles the poles to plot
 */
template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::writePoleHistogramHistoryHDF5(H5::H5File& outfile, std::string dsetBaseName, std::vector<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>>& history, const VecCUDA<T,3>& pole) {
    // Data dimensions
    const unsigned int nTimesteps = history.size();
    const unsigned int histDim = HPP_POLE_FIG_HIST_DIM;

    // Create dataset
    std::string dsetName = dsetBaseName + "_";        
    for (auto val : pole) {
        dsetName += std::to_string((int)val);
    }
    std::vector<hsize_t> dataDims = {nTimesteps, histDim, histDim};
    auto dset = createHDF5Dataset<T>(outfile, dsetName, dataDims);
    
    // Write to dataset
    for (unsigned int i=0; i<nTimesteps; i++) {
        std::vector<hsize_t> offset = {i};
        history[i].writeToExistingHDF5Dataset(dset, offset);
    }
}

template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::writeResultHDF5(std::string filename)
{
    H5::H5File outfile(filename.c_str(), H5F_ACC_TRUNC);
    
    // Stress history
    writeVectorToHDF5Array(outfile, "tHistory", this->tHistory);    
    std::vector<hsize_t> timeDims = {this->TCauchyHistory.size()};
    std::vector<hsize_t> tensorDims = {3,3};
    H5::DataSet TCauchyDset = createHDF5GridOfArrays<T>(outfile, "TCauchyHistory", timeDims, tensorDims);
    for (unsigned int i=0; i<this->TCauchyHistory.size(); i++) {
        std::vector<hsize_t> offset = {i};
        this->TCauchyHistory[i].writeToExistingHDF5Dataset(TCauchyDset, offset);
    }
    
    // Pole figure histograms
    std::string poleHistBasename = "poleHistogram";
    this->writePoleHistogramHistoryHDF5(outfile, poleHistBasename, this->poleHistogramHistory111, VecCUDA<T,3>{1,1,1});   
    this->writePoleHistogramHistoryHDF5(outfile, poleHistBasename, this->poleHistogramHistory110, VecCUDA<T,3>{1,1,0});
    this->writePoleHistogramHistoryHDF5(outfile, poleHistBasename, this->poleHistogramHistory100, VecCUDA<T,3>{1,0,0});
    this->writePoleHistogramHistoryHDF5(outfile, poleHistBasename, this->poleHistogramHistory001, VecCUDA<T,3>{0,0,1});
    this->writePoleHistogramHistoryHDF5(outfile, poleHistBasename, this->poleHistogramHistory011, VecCUDA<T,3>{0,1,1});
    
    // Scalar attributes
    addAttribute(outfile, "spectralPolycrystalSolveTime", solveTimer.getDuration());
    addAttribute(outfile, "nTimestepsTaken", this->getNTimestepsTaken());
    addAttribute(outfile, "nComponents", this->getNComponents());
    addAttribute(outfile, "nFourierTermsComputedHardware", this->getNTermsComputedHardware());
    addAttribute(outfile, "maxMemUsedGB", maxMemUsedGB);
 
    // Close
    outfile.close();
}

template <typename T, unsigned int N>
unsigned int SpectralPolycrystalCUDA<T,N>::getNTimestepsTaken() {
    return tHistory.size()-1;
}

template <typename T, unsigned int N>
unsigned int SpectralPolycrystalCUDA<T,N>::getNComponents() {
    if (useUnifiedDB) {
        return dbUnifiedH.getNDsets();
    }
    else {
        return dbH.getNDsets();
    }
}

template <typename T, unsigned int N>
unsigned long long int SpectralPolycrystalCUDA<T,N>::getNTermsComputedHardware() {
    if (useUnifiedDB) {
        unsigned long long int nTerms = 1;
        nTerms *= this->getNTimestepsTaken();
        nTerms *= nCrystals;  
        nTerms *= this->getNComponents();
        nTerms *= dbUnifiedH.getNTerms();
        return nTerms;
    }
    else {
        unsigned long long int nTerms = 1;
        nTerms *= this->getNTimestepsTaken();
        nTerms *= nCrystals;  
        nTerms *= this->getNComponents();
        nTerms *= dbH.getNDsets();
        nTerms *= dbH.getNTermsTypical();
        return nTerms;
    }
}

// Explicit instantiations
template class SpectralCrystalListCUDA<float>;
template class SpectralCrystalListCUDA<double>;
template class CrystalPropertiesCUDA<float,12>;
template class CrystalPropertiesCUDA<double,12>;
template class SpectralPolycrystalCUDA<float,12>;
template class SpectralPolycrystalCUDA<double,12>;
template class SpectralPolycrystalGSHCUDA<float, CRYSTAL_TYPE_FCC>;
template class SpectralPolycrystalGSHCUDA<double, CRYSTAL_TYPE_FCC>;

}//END NAMESPACE HPP