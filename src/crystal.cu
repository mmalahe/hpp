#include <hpp/tensorCUDA.h>
#include <hpp/crystalCUDA.h>

#include <hpp/spectralUtilsCUDA.h>

#include <hpp/hdfUtilsCpp.h>

namespace hpp
{
#ifdef HPP_USE_CUDA

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
void SpectralPolycrystalCUDA<T,N>::doCrystalSetup(const std::vector<SpectralCrystalCUDA<T>>& crystals, const CrystalPropertiesCUDA<T, N>& crystalProps) {
    // Establish the CUDA context
    CUDA_CHK(cudaFree(0));
    
    // Direct device copies
    nCrystals = crystals.size();
    if (nCrystals % 2 != 0) {
        std::cerr << "Warning: paired crystal implementation does not account for odd total number of crystals" << std::endl;
        ///@todo account for this
    }
    nCrystalPairs = nCrystals/2;
    crystalsD = makeDeviceCopyVecSharedPtr(crystals);
    
    // Crystal properties    
    crystalPropsD = makeDeviceCopySharedPtr(crystalProps);
    
    // Initialise global cauchy stress
    TCauchyGlobalD = makeDeviceCopySharedPtr(this->TCauchyGlobalH);
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
    
    // Get parallel layout for step kernel
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
        TCauchyLevel0Sums = allocDeviceMemorySharedPtr<Tensor2CUDA<T,3,3>>(reduceKernelLevel0Cfg.dG.x);
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

    // Working memory
    TCauchyPerBlockSums = allocDeviceMemorySharedPtr<Tensor2CUDA<T,3,3>>(nBlocks);
    
    // Memory report
    double usedGiB = getUsedMemoryGiB();
    std::cout << "Used Memory (GiB) = " << usedGiB << std::endl;    
}

// Main constructors
template <typename T, unsigned int N>
SpectralPolycrystalCUDA<T,N>::SpectralPolycrystalCUDA(const std::vector<SpectralCrystalCUDA<T>>& crystals, const CrystalPropertiesCUDA<T, N>& crystalProps, const SpectralDatabase<T>& dbIn){    
    // Not using unified database
    useUnifiedDB = false;
    
    // Set up crystals
    this->doCrystalSetup(crystals, crystalProps);
    
    // Set up GPU parameters based on crystals and available hardware
    this->doGPUSetup();
    
    // Set up database
    std::vector<SpectralDatasetID> dsetIDs = defaultCrystalSpectralDatasetIDs();
    dbH = SpectralDatabaseCUDA<T,4>(dbIn, dsetIDs);    
    dbD = makeDeviceCopySharedPtr(this->dbH);
}

template <typename T, unsigned int N>
SpectralPolycrystalCUDA<T,N>::SpectralPolycrystalCUDA(const std::vector<SpectralCrystalCUDA<T>>& crystals, const CrystalPropertiesCUDA<T, N>& crystalProps, const SpectralDatabaseUnified<T>& dbIn){
    // Using unified database
    useUnifiedDB = true;
    
    // Set up crystals
    this->doCrystalSetup(crystals, crystalProps);
    
    // Set up GPU parameters
    this->doGPUSetup();
    
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
Tensor2CUDA<T,3,3> RStretchingTensor, Tensor2AsymmCUDA<T,3> WNext, T theta, T strainIncrement, T dt, SpectralDatabaseCUDA<T,4>* db, Tensor2CUDA<T,3,3> *TCauchyPerBlockSums) 
{
    // Get crystal index
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    // Shared memory for IDFT
    const unsigned int nSharedSpectral = 1024;
    __shared__ SpectralCoordCUDA<4> sharedCoords[nSharedSpectral];
    __shared__ SpectralCoeffCUDA<T> sharedCoeffs[nSharedSpectral];
    
    // If out of bounds, go through motions of calculation, but don't update at end
    /// @fixme: extend this approach to the Cauchy stress sum at the end
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
    T gammaScaling = strainIncrement;
    gammaNext = gammaScaling*db->getIDFTRealDShared(GAMMA, spatialCoord, nSharedSpectral, sharedCoords, sharedCoeffs);
    
    // Update slip deformation resistance
    crystal.s = slipDeformationResistanceStepSpectralSolver(props, crystal.s, gammaNext, dt);
    
    // Sigma
    T sigmaScaling = (crystal.s*powIntrinsic(fabs(strainIncrement), props->m));
    
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
    T WpScaling = strainIncrement;
    
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
    Tensor2CUDA<T,3,3> TCauchyBlockSum = TCauchy;
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
Tensor2CUDA<T,3,3> RStretchingTensor, Tensor2AsymmCUDA<T,3> WNext, T theta, T strainIncrement, T dt, SpectralDatabaseUnifiedCUDA<T,4,P>* db, Tensor2CUDA<T,3,3> *TCauchyPerBlockSums) 
{
    // Get crystal index
    unsigned int pairIdx = blockDim.x*blockIdx.x + threadIdx.x;
    
    // Shared memory for IDFT
    const unsigned int nSharedSpectral = (8/sizeof(T))*128;
    __shared__ SpectralDataUnifiedCUDA<T,4,P> sharedData[nSharedSpectral];
    
    // If out of bounds, go through motions of calculation, but don't update at end
    /// @fixme: extend this approach to the Cauchy stress sum at the end
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
    T gammaScaling = strainIncrement;
    T gammaNext0 = gammaScaling*dbVars0[GAMMA];
    T gammaNext1 = gammaScaling*dbVars1[GAMMA];
    
    // Update slip deformation resistance
    crystal0.s = slipDeformationResistanceStepSpectralSolver(props, crystal0.s, gammaNext0, dt);
    crystal1.s = slipDeformationResistanceStepSpectralSolver(props, crystal1.s, gammaNext1, dt);
    
    // Sigma
    T sigmaScaling0 = (crystal0.s*powIntrinsic(fabs(strainIncrement), props->m));
    T sigmaScaling1 = (crystal1.s*powIntrinsic(fabs(strainIncrement), props->m));
    Tensor2CUDA<T,3,3> sigmaPrimeNext0 = transformOutOfFrame(getSigmaPrime(sigmaScaling0, dbVars0), RStretchingTensor);
    Tensor2CUDA<T,3,3> sigmaPrimeNext1 = transformOutOfFrame(getSigmaPrime(sigmaScaling1, dbVars1), RStretchingTensor);
 
    // Wp
    T WpScaling = strainIncrement;
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
    
    // Calculate Cauchy stress. 
    // Transform out of the stretching tensor frame
    Tensor2CUDA<T,3,3> pairTCauchySum;
    pairTCauchySum += sigmaPrimeNext0;
    pairTCauchySum += sigmaPrimeNext1;
    
    // Add up the Cauchy stresses for this block    
    Tensor2CUDA<T,3,3> TCauchyBlockSum = pairTCauchySum;
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
 * @detail @todo Write a good parallel reduction.
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

template<typename T, unsigned int N>
__global__ void HISTOGRAM_POLES_EQUAL_AREA(unsigned int nCrystals, const SpectralCrystalCUDA<T>* crystals, VecCUDA<T,3>* planeNormalG, Tensor2CUDA<T,N,N>* histG) {
    // Get absolute thread index
    unsigned int baseIdx = blockDim.x*blockIdx.x + threadIdx.x;
    
    // Read in normal from global memory
    VecCUDA<T,3> planeNormal = *planeNormalG;

    // Maximum R value from northern hemisphere projection
    T maxR = (1.00001)*2*sinIntrinsic(M_PI/4);
    
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
        T R = 2*sinIntrinsic(phi/2);
        T x, y;
        sincosIntrinsic(theta, &y, &x);
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

// Step host function
template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::step(const hpp::Tensor2<T>& F_next, const hpp::Tensor2<T>& L_next, T dt) 
{        
    // Get stretching tensor decomposition
    StretchingTensorDecomposition<T> stretchingTensorDecomp = getStretchingTensorDecomposition(L_next); 
    T theta = stretchingTensorDecomp.theta;
    T strainRate = stretchingTensorDecomp.DNorm;
    T strainIncrement = strainRate*dt;
    
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
        SPECTRAL_POLYCRYSTAL_STEP_UNIFIED<<<dG,dB>>>(nCrystalPairs, crystalsD.get(), crystalPropsD.get(), RStretchingTensor, WNext, theta, strainIncrement, dt, dbUnifiedD.get(), TCauchyPerBlockSums.get());
    }
    else {
        SPECTRAL_POLYCRYSTAL_STEP<<<dG,dB>>>(nCrystals, crystalsD.get(), crystalPropsD.get(), RStretchingTensor, WNext, theta, strainIncrement, dt, dbD.get(), TCauchyPerBlockSums.get());    
    }
    
    // Stop solve timer
    CUDA_CHK(cudaDeviceSynchronize());
    solveTimer.stop();
    
    // Sum up the per-block stress
    // TCauchyGlobalD will now contain the global sum of the stress
    unsigned int nBlocks = stepKernelCfg.dG.x;
    
    // Single level reduction
    if (reduceKernelLevel0Cfg.dG.x <= 1) {
        BLOCK_REDUCE_KEPLER_TENSOR2<<<reduceKernelLevel0Cfg.dG, reduceKernelLevel0Cfg.dB>>>(TCauchyPerBlockSums.get(), TCauchyGlobalD.get(), nBlocks);
    }
    // Two level reduction
    else{        
        BLOCK_REDUCE_KEPLER_TENSOR2<<<reduceKernelLevel0Cfg.dG, reduceKernelLevel0Cfg.dB>>>(TCauchyPerBlockSums.get(), TCauchyLevel0Sums.get(), nBlocks);
        BLOCK_REDUCE_KEPLER_TENSOR2<<<reduceKernelLevel1Cfg.dG, reduceKernelLevel1Cfg.dB>>>(TCauchyLevel0Sums.get(), TCauchyGlobalD.get(), reduceKernelLevel0Cfg.dG.x);
    }
    
    // Sync device and host before copying across memory
    CUDA_CHK(cudaDeviceSynchronize());
    
    // Move required quantities to host
    TCauchyGlobalH = getHostValue(TCauchyGlobalD)/(T)nCrystals;
}

template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::evolve(T tStart, T tEnd, T dt, std::function<hpp::Tensor2<T>(T t)> F_of_t, std::function<hpp::Tensor2<T>(T t)> L_of_t) {
    // Initial data
    tHistory.push_back(tStart);
    TCauchyHistory.push_back(TCauchyGlobalH);
    
    // Stepping
    unsigned int nsteps = (tEnd-tStart)/dt;    
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
    }
    
    // Report clock rate
    std::cout << "Clock rate (GHz) = " << getClockRateGHz(deviceID) << std::endl;
}

/**
 * @brief Writes out pole histograms as a Python dictionary
 * @detail Keys are pole specifiers of the form '110' for the pole 110. Values
 * are the pole histograms.
 * @param outfile the output file
 * @param poles the poles to plot
 */
template <typename T, unsigned int N>
void SpectralPolycrystalCUDA<T,N>::writePoleHistogramsHDF5(H5::H5File& outfile, std::string dsetBaseName, const std::vector<VecCUDA<T,3>>& poles) {
    // Histogram configuration
    const unsigned int histDim = 1024;
    CudaKernelConfig histKernelCfg = getKernelConfigMaxOccupancy(devProp, (void*)HISTOGRAM_POLES_EQUAL_AREA<T, histDim>, nCrystals);
    dim3 dG = histKernelCfg.dG;
    dim3 dB = histKernelCfg.dB;
    
    // Begin writing
    for (unsigned int i=0; i<poles.size(); i++) {        
        // Generate histogram
        auto poleH = poles[i];
        std::shared_ptr<VecCUDA<T,3>> poleD = makeDeviceCopySharedPtr(poleH);    
        Tensor2CUDA<T,histDim,histDim>* histHPtr = new Tensor2CUDA<T,histDim,histDim>;
        std::shared_ptr<Tensor2CUDA<T,histDim,histDim>> histD = makeDeviceCopySharedPtrFromPtr(histHPtr);        
        CUDA_CHK(cudaDeviceSynchronize());
        HISTOGRAM_POLES_EQUAL_AREA<T, histDim><<<dG,dB>>>(nCrystals, crystalsD.get(), poleD.get(), histD.get());
        CUDA_CHK(cudaDeviceSynchronize());
        copyToHost(histD, histHPtr);
        
        // Write histogram
        std::string dsetName = dsetBaseName + "_";        
        for (auto val : poleH) {
            dsetName += std::to_string((int)val);
        }
        std::vector<hsize_t> dataDims = {histDim, histDim};
        auto dset = createHDF5Dataset<T>(outfile, dsetName, dataDims);
        std::vector<hsize_t> offset; //no offset
        histHPtr->writeToExistingHDF5Dataset(dset, offset);
        
        // Free
        delete histHPtr;
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
    std::vector<VecCUDA<T,3>> poles;
    poles.push_back(VecCUDA<T,3>{1,1,1});
    poles.push_back(VecCUDA<T,3>{1,1,0});
    poles.push_back(VecCUDA<T,3>{1,0,0});
    poles.push_back(VecCUDA<T,3>{0,0,1});
    poles.push_back(VecCUDA<T,3>{0,1,1});
    this->writePoleHistogramsHDF5(outfile, "poleHistograms", poles);
    
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
        ///@todo implement
        return 0;
    }
}

// Explicit instantiations
template class SpectralCrystalListCUDA<float>;
template class SpectralCrystalListCUDA<double>;
template class CrystalPropertiesCUDA<float,12>;
template class CrystalPropertiesCUDA<double,12>;
template class SpectralPolycrystalCUDA<float,12>;
template class SpectralPolycrystalCUDA<double,12>;

#endif /* HPP_USE_CUDA */
}//END NAMESPACE HPP