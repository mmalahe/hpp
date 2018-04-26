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
#include <hpp/crystal.h>

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
    EulerAngles<T> angles = toEulerAngles(RInStretchingTensorFrame);   

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
__global__ void GET_PER_CRYSTAL_SCALAR_FROM_GSH(const GSHCoeffsCUDA<T>* coeffsPtr, const SpectralCrystalCUDA<T>* crystals, unsigned int ncrystals, T* densities);

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
    SpectralPolycrystalCUDA(size_t ncrystals, T initS, const CrystalPropertiesCUDA<T, N>& crystalProps, const SpectralDatabaseUnified<T>& dbIn, unsigned int seed=0);
    
    // Construction helpers
    void doGPUSetup();
    void doSetup(std::vector<SpectralCrystalCUDA<T>>& crystals, const CrystalPropertiesCUDA<T,N>& crystalProps);
    void setupDatabase(const SpectralDatabaseUnified<T>& dbIn);
    void setupDatabase(const SpectralDatabase<T>& dbIn);
    
    // Simulation
    void setOrientations(const std::vector<EulerAngles<T>>& angleList);
    void setSlipResistances(const std::vector<T>& slipResistances);
    std::vector<T> getSlipResistances() const;
    void setToInitialConditionsRandomOrientations(T init_s, unsigned long int seed);
    void setToInitialConditions(T init_s, const std::vector<EulerAngles<T>>& angleList);
    void resetHistories();
    void step(const hpp::Tensor2<T>& L_next, T dt);
    void step(const hpp::Tensor2<T>& F_next, const hpp::Tensor2<T>& L_next, T dt);
    void evolve(T t_start, T t_end, T dt, std::function<hpp::Tensor2<T>(T t)> F_of_t, std::function<hpp::Tensor2<T>(T t)> L_of_t);
    
    // Output
    std::vector<EulerAngles<T>> getEulerAnglesZXZActive();
    GSHCoeffsCUDA<T> getGSHCoeffs();
    GSHCoeffsCUDA<T> getGSHOfPerCrystalScalar(const std::vector<T>& scalars);
    GSHCoeffsCUDA<T> getGSHOfCrystalDensities(const std::vector<T>& densities);
    std::vector<T> getPerCrystalScalarFromGSH(const GSHCoeffsCUDA<T> coeffs);    
    Tensor2<T> getPoleHistogram(int p0, int p1, int p2);
    void writeResultHDF5(std::string filename);
    
    // Extras
    unsigned int getNTimestepsTaken();
    unsigned int getNComponents();
    unsigned long long int getNTermsComputedHardware();
    
    // Automatically-generated getters
    const Tensor2CUDA<T,3,3>& getTCauchy() const {return TCauchyGlobalH;}
    const std::vector<T>& getTHistory() const {return tHistory;}    
    double getMaxMemUsedGB() const {return maxMemUsedGB;}
    const hpp::Timer& getSolveTimer() const {return solveTimer;}
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
    std::shared_ptr<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> getPoleHistogramDensityWeighted(const VecCUDA<T,3>& pole, const std::vector<T>& densities);
    void getPoleHistogram(Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>& hist, const VecCUDA<T,3>& pole);
    void getPoleHistogramDensityWeighted(Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>& hist, const VecCUDA<T,3>& pole, const std::vector<T>& densities);
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
GSHCoeffsCUDA<T> getGSHFromCrystalOrientations(const std::vector<SpectralCrystalCUDA<T>>& crystals);

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
    SpectralPolycrystalGSHCUDA(const CrystalPropertiesCUDA<T, nSlipSystems(CRYSTAL_TYPE)>& crystalProps, const SpectralDatabaseUnified<T>& dbIn, const T initS, unsigned int orientationSpaceResolution = 5) {
        // The number of points in the orientation is 72*8^{r}, where r is the resolution
        orientationSpace = SO3Discrete<T>(orientationSpaceResolution, toSymmetryType(CRYSTAL_TYPE));
        
        // Generate representative crystals
        this->initS = initS;
        auto crystals = std::vector<SpectralCrystalCUDA<T>>(orientationSpace.size());
        for (unsigned int i=0; i<orientationSpace.size(); i++) {
            crystals[i].s = this->initS;
            crystals[i].angles = orientationSpace.getEulerAngle(i);          
        }
        polycrystal = SpectralPolycrystalCUDA<T, nSlipSystems(CRYSTAL_TYPE)>(crystals, crystalProps, dbIn);
        
        // Initialize densities to a uniform distribution
        densities = std::vector<T>(orientationSpace.size(), 1.0/orientationSpace.size());
    }    
    
    // Setting slip resistances
    void setSlipResistances(const GSHCoeffsCUDA<T>& slipResistanceCoeffs) {
        this->resetSamplingOrientations();
        this->setSlipResistances(polycrystal.getPerCrystalScalarFromGSH(slipResistanceCoeffs));
    }
    void setSlipResistancesToInitialConditions() { 
        this->setSlipResistances(this->initS);
    }
    
    // Setting orientations
    void setOrientations(const GSHCoeffsCUDA<T>& densityCoeffs) {
        this->resetSamplingOrientations();
        densities = polycrystal.getPerCrystalScalarFromGSH(densityCoeffs);        
    }
    
    void setUniformRandomOrientations() {
        this->resetSamplingOrientations();
        // Density at each point is uniform
        for (auto& density : densities) {
            density = 1.0/orientationSpace.size();
        }
    }
    
    // Full resets
    void setToInitialConditions(const GSHCoeffsCUDA<T>& coeffs) {
        this->setOrientations(coeffs);
        this->setSlipResistancesToInitialConditions();
    }
    
    // Simulation
    void step(const hpp::Tensor2<T>& L_next, T dt) {
        polycrystal.step(L_next, dt);
    }    
    
    std::shared_ptr<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> getPoleHistogram(const VecCUDA<T,3>& pole) {
        return polycrystal.getDensityWeightedPoleHistogram(pole, densities);
    }
    
    /**
     * @brief Evolve strictly though the GSH interface
     * @detail Note here that we could for example just call the evolve method
     * of the underlying spectral polycrystal, but the aim of this method is for
     * testing the evolution of the system through a state defined by the GSH
     * coefficients only.
     * @param tStart
     * @param tEnd
     * @param dt
     * @param t
     * @param t
     */
    void evolve(T tStart, T tEnd, T dt, std::function<hpp::Tensor2<T>(T t)> F_of_t, std::function<hpp::Tensor2<T>(T t)> L_of_t) {
        // Initial data
        tHistory.push_back(tStart);
        TCauchyHistory.push_back(polycrystal.getTCauchy());
        
        // Stepping
        unsigned int nsteps = (tEnd-tStart)/dt;
        poleHistogramHistory111.resize(nsteps+1);
        poleHistogramHistory110.resize(nsteps+1);
        poleHistogramHistory100.resize(nsteps+1);
        poleHistogramHistory001.resize(nsteps+1);
        poleHistogramHistory011.resize(nsteps+1);
        this->getPoleHistogram(this->poleHistogramHistory111[0], VecCUDA<T,3>{1,1,1});
        this->getPoleHistogram(this->poleHistogramHistory110[0], VecCUDA<T,3>{1,1,0});
        this->getPoleHistogram(this->poleHistogramHistory100[0], VecCUDA<T,3>{1,0,0});
        this->getPoleHistogram(this->poleHistogramHistory001[0], VecCUDA<T,3>{0,0,1});
        this->getPoleHistogram(this->poleHistogramHistory011[0], VecCUDA<T,3>{0,1,1});
        for (unsigned int i=0; i<nsteps; i++) {
            // Inputs for the next step
            T t = tStart + (i+1)*dt;
            std::cout << "t = " << t << std::endl;
            hpp::Tensor2<T> LNext = L_of_t(t);     
            
            // Step
            this->setOrientations(this->getGSHCoeffs());
            this->setSlipResistances(this->getSlipResistanceGSHCoeffs());
            this->step(LNext, dt);
            
            // Store quantities
            tHistory.push_back(t);
            TCauchyHistory.push_back(polycrystal.getTCauchy());
            this->getPoleHistogram(this->poleHistogramHistory111[i+1], VecCUDA<T,3>{1,1,1});
            this->getPoleHistogram(this->poleHistogramHistory110[i+1], VecCUDA<T,3>{1,1,0});
            this->getPoleHistogram(this->poleHistogramHistory100[i+1], VecCUDA<T,3>{1,0,0});
            this->getPoleHistogram(this->poleHistogramHistory001[i+1], VecCUDA<T,3>{0,0,1});
            this->getPoleHistogram(this->poleHistogramHistory011[i+1], VecCUDA<T,3>{0,1,1});
        }
    }
    
    // Output
    GSHCoeffsCUDA<T> getDensityGSHCoeffs() {
        return polycrystal.getGSHOfPerCrystalScalar(densities);
    }
    GSHCoeffsCUDA<T> getSlipResistanceGSHCoeffs() {
        auto slipResistances = polycrystal.getSlipResistances();
        return polycrystal.getGSHOfPerCrystalScalar(slipResistances);
    }
    GSHCoeffsCUDA<T> getGSHCoeffs() {
        return this->getDensityGSHCoeffs();
    }

    const std::vector<T>& getDensities() const {return densities;}
    const T getDensitySum() const {
        T sum = 0.0;
        for (const auto& density : densities) {
            sum += density;
        }
        return sum;
    }
    
    void writeResultHDF5(std::string filename)
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
        addAttribute(outfile, "spectralPolycrystalSolveTime", polycrystal.getSolveTimer().getDuration());
        addAttribute(outfile, "nTimestepsTaken", polycrystal.getNTimestepsTaken());
        addAttribute(outfile, "nComponents", polycrystal.getNComponents());
        addAttribute(outfile, "nFourierTermsComputedHardware", polycrystal.getNTermsComputedHardware());
        addAttribute(outfile, "maxMemUsedGB", polycrystal.getMaxMemUsedGB());
     
        // Close
        outfile.close();
    }
    
    // Conversions
    unsigned int getNumRepresentativeCrystals(){return densities.size();}
protected:

private:
    /// The underlying spectral polycrystal
    SpectralPolycrystalCUDA<T, nSlipSystems(CRYSTAL_TYPE)> polycrystal;
    
    /// The density associated with each representative crystal
    std::vector<T> densities;
    
    /// A dicrete colelction of points determining the orientation space for the crystal
    SO3Discrete<T> orientationSpace;
    
    /// Initial slip resistance
    T initS;
    
    // Simulation
    void resetSamplingOrientations() {
        std::vector<EulerAngles<T>> angleList(orientationSpace.size());
        for (unsigned int i=0; i<orientationSpace.size(); i++) {
            angleList[i] = orientationSpace.getEulerAngle(i);            
        }
        polycrystal.setOrientations(angleList);
    }
    void setSlipResistances(const std::vector<T>& slipResistances) {
        polycrystal.setSlipResistances(slipResistances);
    }
    void setSlipResistances(T s) {
        std::vector<T> slipResistances(orientationSpace.size());
        for (auto& slip : slipResistances) {
            slip = s;
        }
        this->setSlipResistances(slipResistances);
    }
    
    // Output
    void getPoleHistogram(Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>& hist, const VecCUDA<T,3>& pole) {
        polycrystal.getPoleHistogramDensityWeighted(hist, pole, densities);
    }
    
    // Histories
    std::vector<T> tHistory;
    std::vector<Tensor2CUDA<T,3,3>> TCauchyHistory;
    std::vector<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> poleHistogramHistory111;
    std::vector<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> poleHistogramHistory110;
    std::vector<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> poleHistogramHistory100;
    std::vector<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> poleHistogramHistory001;
    std::vector<Tensor2CUDA<T,HPP_POLE_FIG_HIST_DIM,HPP_POLE_FIG_HIST_DIM>> poleHistogramHistory011;
};

template <typename T>
__device__ Tensor2CUDA<T,3,3> EulerZXZRotationMatrixCUDA(EulerAngles<T> angles) {
    return EulerZXZRotationMatrixCUDA(angles.alpha, angles.beta, angles.gamma);
}

}//END NAMESPACE HPP

#endif /* HPP_CRYSTAL_CUDA_H */