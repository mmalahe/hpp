/** @file crystalCUDA.h
* @author Michael Malahe
* @brief Header file for crystal classes CUDA implementations.
* @details
*/

#ifndef HPP_CRYSTAL_CUDA_H
#define HPP_CRYSTAL_CUDA_H

#include <hpp/config.h>
HPP_CHECK_CUDA_ENABLED_BUILD
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <type_traits>
#include <hpp/rotation.h>
#include <hpp/cudaUtils.h>
#include <hpp/tensorCUDA.h>
#include <hpp/spectralUtilsCUDA.h>
#include <hpp/gshCUDA.h>

namespace hpp
{

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
    Tensor2CUDA<T,N,N> Q;
};

template <typename T>
struct SpectralCrystalCUDA{    
    // Euler angles defining the current crystal orientation
    EulerAngles<T> angles;
    
    // The slip-system deformation resistance
    T s;
    
    // Getters/setters (mainly intended for Python interface)
    T getS() const {return s;}
    void setS(const T& s) {this->s = s;}
    EulerAngles<T> getAngles() const {return angles;}
    void setAngles(const EulerAngles<T>& angles) {this->angles = angles;}
};

template <typename T>
bool operator==(const SpectralCrystalCUDA<T>& l, const SpectralCrystalCUDA<T>& r) {
    if (l.angles != r.angles) {
        return false;
    }
    if (l.s != r.s) {
        return false;
    }
    
    // All checks passed
    return true;
}

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
    T sDot = (props->h_0)*powIntr((T)1.0 - s_alpha/(props->s_s), (props->a))*gammaSum;
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

// GSH conversions
template<typename T>
__global__ void GET_ODF_FROM_GSH(const GSHCoeffsCUDA<T>* coeffsPtr, const SpectralCrystalCUDA<T>* crystals, unsigned int ncrystals, T* densities);

template<typename T>
__global__ void GET_GSH_FROM_ORIENTATIONS(const SpectralCrystalCUDA<T>* crystals, unsigned int ncrystals, GSHCoeffsCUDA<T>* coeffsPerBlockSums);

// Average kernel
template<typename T>
__global__ void GET_AVERAGE_TCAUCHY(unsigned int nCrystals, const SpectralCrystalCUDA<T>* crystals, Tensor2CUDA<T,3,3> *TCauchyGlobal);

/**
 * @class SpectralPolycrystalCUDA
 * @author Michael Malahe
 * @date 06/04/17
 * @file crystalCUDA.h
 * @brief
 * @details Instances of this class only ever live on the host. As it stands
 * this implementation isn't even remotely thread safe, and it is not safe
 * to make and operate on multiple copies of these objects.
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
    void resetRandomOrientations(T init_s, unsigned long int seed);
    void resetGivenOrientations(T init_s, const std::vector<EulerAngles<T>>& angleList);
    void resetHistories();
    void step(const hpp::Tensor2<T>& L_next, T dt);
    void step(const hpp::Tensor2<T>& F_next, const hpp::Tensor2<T>& L_next, T dt);
    void evolve(T t_start, T t_end, T dt, std::function<hpp::Tensor2<T>(T t)> F_of_t, std::function<hpp::Tensor2<T>(T t)> L_of_t);
    
    // Output
    std::vector<EulerAngles<T>> getEulerAnglesZXZActive();
    GSHCoeffsCUDA<T> getGSHCoeffs();
    GSHCoeffsCUDA<T> getDensityWeightedGSH(const std::vector<T>& densities);
    std::vector<T> getDensitiesFromGSH(const GSHCoeffsCUDA<T> coeffs);    
    Tensor2<T> getPoleHistogram(int p0, int p1, int p2);
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
    thrust::device_vector<SpectralCrystalCUDA<T>> crystalsD;
    
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
    thrust::device_vector<Tensor2CUDA<T,3,3>> TCauchyGlobalD;
    
    // Hardware configuration
    int deviceID;
    cudaDeviceProp devProp;
    CudaKernelConfig stepKernelCfg;
    CudaKernelConfig reduceKernelLevel0Cfg;
    CudaKernelConfig reduceKernelLevel1Cfg;
    CudaKernelConfig gshKernelCfg;
    CudaKernelConfig gshReduceKernelLevel0Cfg;
    CudaKernelConfig gshReduceKernelLevel1Cfg;
    
    // Working memory
    thrust::device_vector<Tensor2CUDA<T,3,3>> TCauchyPerBlockSums;
    thrust::device_vector<Tensor2CUDA<T,3,3>> TCauchyLevel0Sums;
    thrust::device_vector<GSHCoeffsCUDA<T>> gshPerBlockSums;
    thrust::device_vector<GSHCoeffsCUDA<T>> gshLevel0Sums;
    
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

/**
 * @class SpectralPolycrystalGSHCUDA
 * @author Michael Malahe
 * @date 27/03/18
 * @file crystalCUDA.h
 * @brief A class for simulating a polycrystal using a Fourier compressed database.
 * @detail The class presents an interface that allows it to be modified only
 * through the Generalized Spherical Harmonic representation.
 * 
 * The implementation takes an approach where the polycrystal is composed of
 * a number of single crystals that are evenly distributed in orientation space
 * within the fundamental zone. Each crystal represents a volume in the 
 * fundamental zone. Associated with each crystal is a density, with the summed 
 * density for all of the crystals coming to 1.
 * 
 * Instances of this class only ever live on the host.
 * @tparam T scalar type
 * @tparam N number of slip systems in the polycrystal
 */
template <typename T, CrystalType CRYSTAL_TYPE>
class SpectralPolycrystalGSHCUDA
{
public:    
    // Constructors
    SpectralPolycrystalGSHCUDA(){;}    
    SpectralPolycrystalGSHCUDA(CrystalPropertiesCUDA<T, nSlipSystems(CRYSTAL_TYPE)>& crystalProps, const SpectralDatabaseUnified<T>& dbIn, T init_s) {
        // Number of points in the orientation space = 72*8^{r}, where r is the resolution
        unsigned int orientationSpaceResolution = 6; 
        switch (CRYSTAL_TYPE) {
            case CRYSTAL_TYPE_NONE:
                orientationSpace = SO3Discrete<T>(orientationSpaceResolution);
                break;
            case CRYSTAL_TYPE_FCC:
                orientationSpace = SO3Discrete<T>(orientationSpaceResolution, SYMMETRY_TYPE_C4);
                break;
            default:
                std::cerr << "No implementation for crystal type = " << CRYSTAL_TYPE << std::endl;
                throw std::runtime_error("No implementation.");
        }
        
        // Generate representative crystals
        auto crystals = std::vector<SpectralCrystalCUDA<T>>(orientationSpace.size());
        for (unsigned int i=0; i<orientationSpace.size(); i++) {
            crystals[i].s = init_s;
            crystals[i].angles = orientationSpace.getEulerAngle(i);            
        }
        polycrystal = SpectralPolycrystalCUDA<T, nSlipSystems(CRYSTAL_TYPE)>(crystals, crystalProps, dbIn);
        
        // Initialize densities to a uniform distribution
        densities = std::vector<T>(orientationSpace.size(), 1.0/orientationSpace.size());
    }    
    
    void resetSamplingOrientations(T init_s) {
        std::vector<EulerAngles<T>> angleList(orientationSpace.size());
        for (unsigned int i=0; i<orientationSpace.size(); i++) {
            angleList[i] = orientationSpace.getEulerAngle(i);            
        }
        polycrystal.resetGivenOrientations(init_s, angleList);
    }
    
    // Simulation
    void resetUniformRandomOrientations(T init_s) {
        // Orientation sampling points remain uniform
        this->resetSamplingOrientations(init_s);
        
        // Density at each point is uniform
        for (auto& density : densities) {
            density = 1.0/orientationSpace.size();
        }
    }
    
    void resetGivenGSHCoeffs(T init_s, const GSHCoeffsCUDA<T>& coeffs) {
        // Orientation sampling points remain uniform
        this->resetSamplingOrientations(init_s);
        
        // Density at each point is determined by GSH coefficients
        densities = polycrystal.getDensitiesFromGSH(coeffs);
    }
    
    void step(const hpp::Tensor2<T>& L_next, T dt) {
        polycrystal.step(L_next, dt);
    }
    
    // Output
    GSHCoeffsCUDA<T> getGSHCoeffs() {
        return polycrystal.getDensityWeightedGSH(densities);
    }
protected:

private:
    /// The underlying spectral polycrystal
    SpectralPolycrystalCUDA<T, nSlipSystems(CRYSTAL_TYPE)> polycrystal;
    
    /// The density of each of the crystals
    std::vector<T> densities;
    
    /// A dicrete colelction of points determining the orientation space for the crystal
    SO3Discrete<T> orientationSpace;
};

template <typename T>
__device__ Tensor2CUDA<T,3,3> EulerZXZRotationMatrixCUDA(EulerAngles<T> angles) {
    return EulerZXZRotationMatrixCUDA(angles.alpha, angles.beta, angles.gamma);
}

}//END NAMESPACE HPP

#endif /* HPP_CRYSTAL_CUDA_H */