/** @file crystalCUDA.h
* @author Michael Malahe
* @brief Header file for crystal classes CUDA implementations.
* @details
*/

#ifndef HPP_CRYSTAL_CUDA_H
#define HPP_CRYSTAL_CUDA_H

#include <hpp/config.h>
#include <hpp/crystal.h>
#include <hpp/cudaUtils.h>
#include <hpp/tensorCUDA.h>
#include <hpp/spectralUtilsCUDA.h>

namespace hpp
{
#ifdef HPP_USE_CUDA

#define HPP_POLE_FIG_HIST_DIM 1024

// Forward declarations
template <typename T, unsigned int N> class SpectralPolycrystalCUDA;
template <typename T> struct SpectralCrystalCUDA;

/**
 * @class CrystalPropertiesCUDA
 * @author Michael Malahe
 * @date 06/04/17
 * @file crystalCUDA.h
 * @tparam T scalar type
 * @tparam N number of slip systems
 * @brief 
 */
template <typename T, unsigned int N>
class CrystalPropertiesCUDA {
public:
    // Friends
    friend class SpectralPolycrystalCUDA<T,N>;

    // Constructor
    CrystalPropertiesCUDA(const CrystalProperties<T>& propsIn);
public:
    unsigned int n_alpha;
    T mu;
    T kappa;
    T m;
    T gammadot_0;
    T h_0;
    T s_s;
    T a;
    T q;
    T volume = 1.0;
    VecCUDA<T,3> m_0[N];
    VecCUDA<T,3> n_0[N];
    Tensor2CUDA<T,3,3> S_0[N];
    Tensor4CUDA<T,3,3,3,3> L;
    Tensor2CUDA<T,12,12> Q;
};

template <typename T>
struct SpectralCrystalCUDA{    
    // Euler angles defining the current crystal orientation
    EulerAngles<T> angles;
    
    // The slip-system deformation resistance
    T s;
};

/**
 * @class SpectralCrystalListCUDA
 * @author Michael Malahe
 * @date 17/04/17
 * @file crystalCUDA.h
 * @brief A class for storing crystal information in a way that makes coalesced 
 * global memory reads/writes faster.
 */
template <typename T>
class SpectralCrystalListCUDA{
public:
    SpectralCrystalListCUDA(){;}
    SpectralCrystalListCUDA(unsigned int nCrystals, const SpectralCrystalCUDA<T> *crystals);
    SpectralCrystalListCUDA(const std::vector<SpectralCrystalCUDA<T>>& crystals);
    
    // Get
    __device__ SpectralCrystalCUDA<T> getCrystalD(unsigned int iCrystal) {
        SpectralCrystalCUDA<T> crystal;
        crystal.angles.alpha = anglesA[iCrystal];
        crystal.angles.beta = anglesB[iCrystal];
        crystal.angles.gamma = anglesC[iCrystal];
        crystal.s = s[iCrystal];
        return crystal;
    }
    
    // Set
    __device__ void setCrystalD(unsigned int iCrystal, const SpectralCrystalCUDA<T>& crystal) {
        anglesA[iCrystal] = crystal.angles.alpha;
        anglesB[iCrystal] = crystal.angles.beta;
        anglesC[iCrystal] = crystal.angles.gamma;
        s[iCrystal] = crystal.s;
    }
private:
    T *anglesA;
    T *anglesB;
    T *anglesC;
    T *s;
    std::vector<std::shared_ptr<T>> sharedPtrs;
};

// Slip deformation resistance update
template <typename T, unsigned int N>
__device__ T slipDeformationResistanceStepSpectralSolver(const CrystalPropertiesCUDA<T,N>* props, 
const T s_alpha, const T gammaSum, const T dt) 
{ 
    // Return
    T sDot = (props->h_0)*powIntrinsic((T)1.0 - s_alpha/(props->s_s), (props->a))*gammaSum;
    return s_alpha + sDot*dt;
}

// Get database coordinate
template<typename T, unsigned int P>
__device__ void getSpectralCrystalDatabaseCoordinate(SpectralCrystalCUDA<T>& crystal, SpectralDatabaseUnifiedCUDA<T,4,P>* db, Tensor2CUDA<T,3,3>& RStretchingTensor, T theta, unsigned int *dbCoord) {    
    // Get orientation in lab frame
    Tensor2CUDA<T,3,3> RLab = EulerZXZRotationMatrixCUDA(crystal.angles);
    
    // Transform rotation matrix into stretching tensor frame
    Tensor2CUDA<T,3,3> RInStretchingTensorFrame = RStretchingTensor.trans()*RLab;
    
    // Euler angles
    EulerAngles<T> angles = getEulerZXZAngles(RInStretchingTensorFrame);   

    // Database coordinate
    T gridPos[4] = {angles.alpha, angles.beta, angles.gamma, theta};    
    T *gridStarts = db->getGridStarts();
    T *gridSteps = db->getGridSteps();
    unsigned int *gridDims = db->getGridDims();
    for (unsigned int i=0; i<4; i++) {
        dbCoord[i] = nearbyint((gridPos[i] - gridStarts[i])/gridSteps[i]);
        // Periodicity
        dbCoord[i] = dbCoord[i]%gridDims[i];
    }
}

// Step kernels
template<typename T, unsigned int N>
__global__ void SPECTRAL_POLYCRYSTAL_STEP(unsigned int nCrystals, SpectralCrystalCUDA<T>* crystals, CrystalPropertiesCUDA<T,N>* props, 
Tensor2CUDA<T,3,3> RStretchingTensor, Tensor2AsymmCUDA<T,3> WNext, T theta, T eDot,  T dt, SpectralDatabaseCUDA<T,4>* db, Tensor2CUDA<T,3,3> *TCauchyPerBlockSums);

template<typename T, unsigned int N, unsigned int P>
__global__ void SPECTRAL_POLYCRYSTAL_STEP_UNIFIED(unsigned int nCrystals, SpectralCrystalCUDA<T>* crystals, CrystalPropertiesCUDA<T,N>* props, 
Tensor2CUDA<T,3,3> RStretchingTensor, Tensor2AsymmCUDA<T,3> WNext, T theta, T eDot,  T dt, SpectralDatabaseUnifiedCUDA<T,4,P>* db, Tensor2CUDA<T,3,3> *TCauchyPerBlockSums);

// Average kernel
template<typename T>
__global__ void GET_AVERAGE_TCAUCHY(unsigned int nCrystals, const SpectralCrystalCUDA<T>* crystals, Tensor2CUDA<T,3,3> *TCauchyGlobal);

/**
 * @class SpectralPolycrystalCUDA
 * @author Michael Malahe
 * @date 06/04/17
 * @file crystalCUDA.h
 * @brief
 * @details Instances of this class only ever live on the host.
 * @tparam T scalar type
 * @tparam N number of slip systems
 */
template <typename T, unsigned int N>
class SpectralPolycrystalCUDA
{
public:
    
    // Dataset index ordering
    unsigned int nDsets = 9;
    
    // Constructors
    SpectralPolycrystalCUDA(){;}
    SpectralPolycrystalCUDA(std::vector<SpectralCrystalCUDA<T>>& crystals, const CrystalPropertiesCUDA<T, N>& crystalProps, const SpectralDatabase<T>& dbIn);
    SpectralPolycrystalCUDA(std::vector<SpectralCrystalCUDA<T>>& crystals, const CrystalPropertiesCUDA<T, N>& crystalProps, const SpectralDatabaseUnified<T>& dbIn);    
    
    // Construction helpers
    void doGPUSetup();
    void doSetup(std::vector<SpectralCrystalCUDA<T>>& crystals, const CrystalPropertiesCUDA<T,N>& crystalProps);
    
    // Simulation
    void step(const hpp::Tensor2<T>& F_next, const hpp::Tensor2<T>& L_next, T dt);
    void evolve(T t_start, T t_end, T dt, std::function<hpp::Tensor2<T>(T t)> F_of_t, std::function<hpp::Tensor2<T>(T t)> L_of_t);

    // Write
    void writeResultHDF5(std::string filename);
    
    // Extras
    unsigned int getNTimestepsTaken();
    unsigned int getNComponents();
    unsigned long long int getNTermsComputedHardware();
    
    // Automatically-generated getters
    const std::vector<T>& getTHistory() const {return tHistory;}
protected:

private:
    // List of crystals
    unsigned int nCrystals;
    unsigned int nCrystalPairs;
    std::shared_ptr<SpectralCrystalCUDA<T>> crystalsD;
    
    // Crystal properties
    std::shared_ptr<CrystalPropertiesCUDA<T,N>> crystalPropsD;
    
    // Spectral database to use for solving.
    
    // We maintain a host copy of the database, since
    // the dynamically-allocated device memory in the host
    // is freed when the host destructor is called. The alternative
    // is to write a deep copy function from host to device memory
    // for the database.
    bool useUnifiedDB = false;
    SpectralDatabaseCUDA<T,4> dbH;
    std::shared_ptr<SpectralDatabaseCUDA<T,4>> dbD;    
    SpectralDatabaseUnifiedCUDA<T,4,9> dbUnifiedH;
    std::shared_ptr<SpectralDatabaseUnifiedCUDA<T,4,9>> dbUnifiedD;
    
    // Global
    Tensor2CUDA<T,3,3> TCauchyGlobalH;
    std::shared_ptr<Tensor2CUDA<T,3,3>> TCauchyGlobalD;
    
    // Hardware configuration
    int deviceID;
    cudaDeviceProp devProp;
    CudaKernelConfig stepKernelCfg;
    CudaKernelConfig reduceKernelLevel0Cfg;
    CudaKernelConfig reduceKernelLevel1Cfg;
    
    // Working memory
    std::shared_ptr<Tensor2CUDA<T,3,3>> TCauchyPerBlockSums;
    std::shared_ptr<Tensor2CUDA<T,3,3>> TCauchyLevel0Sums;
    
    // Stress-strain history
    std::vector<T> tHistory;
    std::vector<Tensor2CUDA<T,3,3>> TCauchyHistory;
    
    // Texture history
    std::shared_ptr<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> getPoleHistogram(const VecCUDA<T,3>& pole);
    void getPoleHistogram(Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>& hist, const VecCUDA<T,3>& pole);
    void writePoleHistogramHistoryHDF5(H5::H5File& outfile, std::string dsetBaseName, std::vector<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>>& history, const VecCUDA<T,3>& pole);
    std::vector<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> poleHistogramHistory111;
    std::vector<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> poleHistogramHistory110;
    std::vector<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> poleHistogramHistory100;
    std::vector<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> poleHistogramHistory001;
    std::vector<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> poleHistogramHistory011;

    // Profiling
    hpp::Timer solveTimer;
    double maxMemUsedGB = 0.0;
};

template <typename T>
__device__ Tensor2CUDA<T,3,3> EulerZXZRotationMatrixCUDA(EulerAngles<T> angles) {
    return EulerZXZRotationMatrixCUDA(angles.alpha, angles.beta, angles.gamma);
}

#endif /* HPP_USE_CUDA */
}//END NAMESPACE HPP

#endif /* HPP_CRYSTAL_CUDA_H */