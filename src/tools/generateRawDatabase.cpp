/** @file mihaila2014GenerateDatabase.cpp
* @author Michael Malahe
* @brief Generating the results database from Mihaila2014
*/

#include <hpp/tensor.h>
#include <hpp/crystal.h>
#include <cassert>
#include <iostream>
#include <math.h>
#include <vector>
#include <functional>
#include <stdio.h>
#include <hpp/mpiUtils.h>
#include <hpp/hdfUtils.h>
#include <tclap/CmdLine.h>

namespace mihaila2014
{

template <typename U>
hpp::Tensor2<U> velocityGradient(U theta, U eDot) {
    hpp::Tensor2<U> D = hpp::stretchingVelocityGradient(theta, eDot);
    hpp::Tensor2<U> W(3,3);//yes, this is just zeros in this case
    hpp::Tensor2<U> L = D + W;
    return L;
}

template <typename U>
hpp::Tensor2<U> latticeRotationTensor(hpp::Tensor2<U> F_e) {
    hpp::PolarDecomposition<U> decomp = F_e.polarDecomposition();
    return decomp.R;
}

template <typename U>
struct MainQuantities {
    hpp::Tensor2<U> sigmaPrime;
    hpp::Tensor2<U> W_p;
    U gammadotAbsSum;
};

/**
 * @brief The deformation gradient subject to a constant velocity gradient
 * @details The velocity gradient is further constrained to being diagonal
 * @param t
 * @param L
 * @return 
 */
template <typename U>
hpp::Tensor2<U> deformationGradientFromDiagonalVelocityGradient(U t, hpp::Tensor2<U> L) {
    hpp::Tensor2<U> F(3,3);
    for (unsigned int i=0; i<3; i++) {
        F(i,i) = std::exp(L(i,i)*t);
    }
    return F;
}

/**
 * @brief From a constant diagonal velocity gradient, get the deformation gradient at time \f$t\f$
 * @details Additionally, transform the resulting velocity into the frame defined by the columns of Q
 * @param t
 * @param L
 * @param Q
 * @return 
 */
template <typename U>
hpp::Tensor2<U> deformationGradientFromDiagonalVelocityGradientAndTransformIntoFrame(U t, hpp::Tensor2<U> L, hpp::Tensor2<U> Q) {
    hpp::Tensor2<U> F(3,3);
    for (unsigned int i=0; i<3; i++) {
        F(i,i) = std::exp(L(i,i)*t);
    }
    // NB: The deformation gradient transforms like a vector
    return Q.trans()*F;
}

template <typename U>
MainQuantities<U> solveForMainQuantities(hpp::CrystalProperties<U> props, hpp::Tensor2<U> T_0, U s_0, hpp::Tensor2<U> F_p_0, 
                                         hpp::EulerAngles<U> g_p, U theta, U strainRate, U dtInitial, U strainIncrement) 
{    
    // Solver configuration
    hpp::CrystalSolverConfig<U> config = hpp::defaultConservativeCrystalSolverConfig<U>();
    
    // Initial conditions
    hpp::CrystalInitialConditions<U> init;
    init.T_init = T_0;
    init.s_0 = s_0;
    init.F_p_0 = F_p_0;
    
    // Rotate the crystal properties
    hpp::Tensor2<U> crystalRotation = hpp::EulerZXZRotationMatrix(g_p.alpha, g_p.beta, g_p.gamma);
    init.crystalRotation = crystalRotation;
    
    // Construct crystal
    std::vector<hpp::Crystal<U>> initialCrystals = {hpp::Crystal<U>(props, config, init)};
    hpp::Polycrystal<U> polycrystal = hpp::Polycrystal<U>(initialCrystals, MPI_COMM_SELF);
    
    // Calculate the velocity gradient
    hpp::Tensor2<U> L = velocityGradient(theta, strainRate);
    
    // Timespan
    U tStart = 0.0;
    U tEnd = strainIncrement/strainRate;
    
    // Deformation gradient as a function of time
    std::function<hpp::Tensor2<U>(U)> F_of_t = 
    std::bind(deformationGradientFromDiagonalVelocityGradient<U>, std::placeholders::_1, L);
    
    // Simulate
    polycrystal.evolve(tStart, tEnd, dtInitial, F_of_t);
    hpp::Crystal<U> crystal = polycrystal.getCrystal(0);
    
    // Cauchy stress
    hpp::Tensor2<U> TCauchy = crystal.getTCauchy();
    hpp::Tensor2<U> W_p = crystal.getPlasticSpinTensor();
    
    // Determine the average slip system resistance for scaling
    // If the Voce law is used in the underlying solver, each s will be identical anyway.
    U s = 0;
    std::vector<U> s_alphas = crystal.getSAlphas();
    for (auto s_alpha : s_alphas) {
        s += s_alpha;
    }
    s /= s_alphas.size();
    
    // Determine main quantities to store
    MainQuantities<U> mainQuantities;
    mainQuantities.sigmaPrime = TCauchy.deviatoricComponent()/(s*std::pow(std::abs(strainRate), props.m));
    mainQuantities.W_p = W_p/strainRate;
    std::vector<U> gammadot_alphas = hpp::operator/(crystal.getShearStrainRates(), strainRate);
    mainQuantities.gammadotAbsSum = 0.0;
    for (auto&& gammadot_alpha : gammadot_alphas) {
        mainQuantities.gammadotAbsSum += std::abs(gammadot_alpha);
    }
    
    // Return
    return mainQuantities;
}

template <typename U>
void generateDatabase(std::string output_filename, unsigned int gridLength, MPI_Comm comm)
{   
    // Message
    std::cout << "Generating database" << std::endl;
    
    // Initial conditions
    hpp::Tensor2<U> T_0(3,3);
    U s_0 = 16.0e-3; // (Gpa)
    hpp::Tensor2<U> F_p_0 = hpp::identityTensor2<U>(3);

    // Material parameters
    hpp::CrystalProperties<U> mprops = hpp::defaultCrystalProperties<U>();

    // Increment size for simulation
    U eDot = 1e-3;
    U dtInitial = 1e-3;

    // Minimum strain to be considered a sufficient increment
    U minStrain = 2e-2;
    
    // Grid lengths
    unsigned int nAlpha = gridLength;
    unsigned int nBeta = gridLength;
    unsigned int nGamma = gridLength;
    unsigned int nTheta = gridLength;
    std::vector<hsize_t> gridDims = {nAlpha, nBeta, nGamma, nTheta};
    
    // Grid lengths accounting for symmetries
    // Gamma uses the F(gamma) = F(gamma - pi/2) symmetry
    // Theta uses the F(theta) = -F(theta - pi) symmetry    
    unsigned int nAlphaSymm = nAlpha; 
    unsigned int nBetaSymm = nBeta;
    unsigned int nGammaSymm = nGamma/4;
    unsigned int nThetaSymm = nTheta/2;
    unsigned int nGridpointsSymm = nAlphaSymm*nBetaSymm*nGammaSymm*nThetaSymm;
    std::vector<hsize_t> gridDimsSymm = {nAlphaSymm, nBetaSymm, nGammaSymm, nThetaSymm};
    
    // Spatial grids
    std::vector<U> spatialGridStarts = {0,0,0,0};
    std::vector<U> spatialGridEnds = {2*M_PI, 2*M_PI, 2*M_PI, 2*M_PI};
    U alphaIncrement = 2*M_PI/nAlpha;
    U betaIncrement = 2*M_PI/nBeta;
    U gammaIncrement = 2*M_PI/nGamma;
    U thetaIncrement = 2*M_PI/nTheta;
    
    // Divide up grid responsibilities
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Basic division
    unsigned int nGridpointsLocalSymm = nGridpointsSymm/comm_size;
    if (nGridpointsSymm % comm_size != 0) nGridpointsLocalSymm++;
    unsigned int gridStartSymm = comm_rank*nGridpointsLocalSymm;
    unsigned int gridEndSymm = gridStartSymm+nGridpointsLocalSymm-1;
    
    // Re-align slabs to break cleanly on final two dimensions
    std::vector<hsize_t> chunkCountSymm = {1,1,gridDimsSymm[2],gridDimsSymm[3]};
    int chunkSizeSymm = 1;
    for (auto&& a : chunkCountSymm) {
        chunkSizeSymm *= a;
    }
    if (gridStartSymm % chunkSizeSymm != 0) {
        gridStartSymm -= (gridStartSymm%chunkSizeSymm);
    }
    if ((gridEndSymm+1) % chunkSizeSymm != 0) {
        gridEndSymm -= ((gridEndSymm+1)%chunkSizeSymm);
    }
    
    // Make sure final slab fits into end of range
    if (gridEndSymm > nGridpointsSymm-1) gridEndSymm = nGridpointsSymm-1;   
    nGridpointsLocalSymm = gridEndSymm - gridStartSymm + 1;
    printf("Proc %d taking care of %u-%u.\n", comm_rank, gridStartSymm, gridEndSymm);
    
    // Get final chunk count
    if (nGridpointsLocalSymm % chunkSizeSymm != 0) {
        throw std::runtime_error("Chunks not allocated correctly.");
    }   
    
    // Create file and datasets //
    //////////////////////////////

    // File
    hpp::HDF5Handler file(output_filename, comm, true);
    hid_t plist_id_xfer_independent = file.getPropertyListTransferIndependent();
    std::vector<hsize_t> tensorDims = {3,3};
    
    // Useful metadata. This should only be written by one process
    std::vector<hsize_t> nDimsArray = {4};
    hid_t dsetGridDims = file.createDataset<unsigned short int>("grid_dims", nDimsArray);
    hid_t dsetGridStarts = file.createDataset<U>("grid_starts", nDimsArray);
    hid_t dsetGridEnds = file.createDataset<U>("grid_ends", nDimsArray);
    if (comm_rank == 0) {
        std::vector<unsigned short int> gridDimsOut(gridDims.begin(), gridDims.end());        
        hpp::writeSingleHDF5Array(dsetGridDims, plist_id_xfer_independent, nDimsArray, gridDimsOut.data());
        hpp::writeSingleHDF5Array(dsetGridStarts, plist_id_xfer_independent, nDimsArray, spatialGridStarts.data());
        hpp::writeSingleHDF5Array(dsetGridEnds, plist_id_xfer_independent, nDimsArray, spatialGridEnds.data());
    }  
    
    // Datasets
    std::string sigmaPrimeDatasetName("sigma_prime");
    file.createDataset<U>(sigmaPrimeDatasetName, tensorDims, gridDims);
    hid_t dsetSigmaPrime = file.getDataset(sigmaPrimeDatasetName);
    
    std::string WpDatasetName("W_p");   
    file.createDataset<U>(WpDatasetName, tensorDims, gridDims);
    hid_t dsetWp = file.getDataset(WpDatasetName);   
    
    std::string gammadotAbsSumDatasetName("gammadot_abs_sum");
    file.createDataset<U>(gammadotAbsSumDatasetName, gridDims);
    hid_t dsetGammadotAbsSum = file.getDataset(gammadotAbsSumDatasetName);
    
    // I/O buffering
    std::vector<hsize_t> scalarDims;
    std::vector<U> sigmaPrimeWriteBuffers[3][3];
    std::vector<U> WpWriteBuffers[3][3];
    std::vector<U> sigmaPrimeWriteBuffersThetaShift[3][3];
    std::vector<U> WpWriteBuffersThetaShift[3][3];
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            sigmaPrimeWriteBuffers[i][j].resize(chunkSizeSymm);
            WpWriteBuffers[i][j].resize(chunkSizeSymm);
            sigmaPrimeWriteBuffersThetaShift[i][j].resize(chunkSizeSymm);
            WpWriteBuffersThetaShift[i][j].resize(chunkSizeSymm);
        }
    }
    std::vector<U> gammadotAbsSumWriteBuffer(chunkSizeSymm);
    std::vector<U> gammadotAbsSumWriteBufferThetaShift(chunkSizeSymm);
    
    // Loop over grid //
    ////////////////////
    
    for (unsigned int i=0; i<nGridpointsLocalSymm; i++) {
        // Get grid coordinates from  linear index
        unsigned int flatIdx = gridStartSymm+i;
        hpp::idx4d gridIdx = hpp::unflat(flatIdx, gridDimsSymm[0], gridDimsSymm[1],
                                    gridDimsSymm[2], gridDimsSymm[3]);
        std::vector<hsize_t> gridOffset = {gridIdx.i, gridIdx.j, gridIdx.k, gridIdx.l};
        
        // Status
        if (i % chunkSizeSymm == 0) {
            float completionPercentage = (100.0*i)/nGridpointsLocalSymm;
            printf("Proc %d, point %u/%u (%1.1f%%)\n", comm_rank, i, nGridpointsLocalSymm, completionPercentage);
        }
        
        // Parameters
        hpp::EulerAngles<U> g_p;
        g_p.alpha = alphaIncrement*gridOffset[0];
        g_p.beta = betaIncrement*gridOffset[1];
        g_p.gamma = gammaIncrement*gridOffset[2];
        U theta = thetaIncrement*gridOffset[3];
        
        // Solve
        MainQuantities<U> mainQuantities = solveForMainQuantities(mprops, T_0, s_0,
        F_p_0, g_p, theta, eDot, dtInitial, minStrain);
        
        // Store quantities
        int inChunkIdx = i%chunkSizeSymm;
        for (unsigned int ix=0; ix<3; ix++) {
            for (unsigned int iy=0; iy<3; iy++) {
                sigmaPrimeWriteBuffers[ix][iy][inChunkIdx] = mainQuantities.sigmaPrime(ix,iy);
                WpWriteBuffers[ix][iy][inChunkIdx] = mainQuantities.W_p(ix,iy);
            }
        }
        gammadotAbsSumWriteBuffer[inChunkIdx] = mainQuantities.gammadotAbsSum;        
        
        // Write quantities, including symmetric extensions
        if ((i+1) % chunkSizeSymm == 0) {
            assert(gridOffset[2] == gridDims[2]-1 && gridOffset[3] == gridDims[3]-1);
            std::vector<hsize_t> gridWriteOffsetBase = {gridIdx.i, gridIdx.j, 0, 0};
            
            // Calculate negations of theta for easy writing of the theta symmetry later
            for (int flatIdxInChunk=0; flatIdxInChunk<chunkSizeSymm; flatIdxInChunk++) {
                for (unsigned int ix=0; ix<3; ix++) {
                    for (unsigned int iy=0; iy<3; iy++) {
                        sigmaPrimeWriteBuffersThetaShift[ix][iy][flatIdxInChunk] = -sigmaPrimeWriteBuffers[ix][iy][flatIdxInChunk];
                        WpWriteBuffersThetaShift[ix][iy][flatIdxInChunk] = -WpWriteBuffers[ix][iy][flatIdxInChunk];
                    }
                }
                gammadotAbsSumWriteBufferThetaShift[flatIdxInChunk] = gammadotAbsSumWriteBuffer[flatIdxInChunk];
            }
            
            // Extend Gamma symmetry: F(gamma) = F(gamma - pi/2)
            for (unsigned int iGammaSymm=0; iGammaSymm<4; iGammaSymm++) {
                // Add a multiple of pi/2 to the index offset
                auto gridWriteOffset = gridWriteOffsetBase;
                gridWriteOffset[2] += iGammaSymm*nGammaSymm;
                
                // Extend theta symmetry: F(theta) = -F(theta - pi)
                for (unsigned int iThetaSymm=0; iThetaSymm<2; iThetaSymm++) {
                    gridWriteOffset[3] += iThetaSymm*nThetaSymm;                    
                    
                    // Tensor coordinates
                    for (unsigned int ix=0; ix<3; ix++) {
                        for (unsigned int iy=0; iy<3; iy++) {
                            // Offset and data count
                            std::vector<hsize_t> gridWriteOffsetTensor = {ix,iy};
                            gridWriteOffsetTensor.insert(gridWriteOffsetTensor.end(), gridWriteOffset.begin(), gridWriteOffset.end());
                            std::vector<hsize_t> chunkCountTensor = {1,1};
                            chunkCountTensor.insert(chunkCountTensor.end(), chunkCountSymm.begin(), chunkCountSymm.end());

                            // Writing
                            if (iThetaSymm==0) {
                                hpp::writeMultipleHDF5Arrays(dsetSigmaPrime, plist_id_xfer_independent, gridWriteOffsetTensor, scalarDims, chunkCountTensor,sigmaPrimeWriteBuffers[ix][iy].data());
                                hpp::writeMultipleHDF5Arrays(dsetWp, plist_id_xfer_independent, gridWriteOffsetTensor, scalarDims, chunkCountTensor, WpWriteBuffers[ix][iy].data());
                            }
                            else {
                                hpp::writeMultipleHDF5Arrays(dsetSigmaPrime, plist_id_xfer_independent, gridWriteOffsetTensor, scalarDims, chunkCountTensor,sigmaPrimeWriteBuffersThetaShift[ix][iy].data());
                                hpp::writeMultipleHDF5Arrays(dsetWp, plist_id_xfer_independent, gridWriteOffsetTensor, scalarDims, chunkCountTensor, WpWriteBuffersThetaShift[ix][iy].data());
                            }
                        }
                    }
                    if (iThetaSymm==0) {
                        hpp::writeMultipleHDF5Arrays(dsetGammadotAbsSum, plist_id_xfer_independent, gridWriteOffset, scalarDims, chunkCountSymm, gammadotAbsSumWriteBuffer.data());
                    }
                    else {
                        hpp::writeMultipleHDF5Arrays(dsetGammadotAbsSum, plist_id_xfer_independent, gridWriteOffset, scalarDims, chunkCountSymm, gammadotAbsSumWriteBufferThetaShift.data());
                    }
                }
            }
        }        
    }
}

} // END NAMESPACE mihaila2014

int main(int argc, char *argv[]) 
{    
    // MPI init
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    
    // Options
    std::string databaseFilename;
    unsigned int gridLength = 0;
    
    // Get options
    try {
        // The parser
        TCLAP::CmdLine parser("Generate the raw database of responses.", ' ', "0.3");
        
        // Filename option
        std::string filenameArgChar = "o";
        std::string defaultFilename = "database.hdf5";
        bool filenameRequired = true;
        std::string filenameDescription = "Output filename";
        TCLAP::ValueArg<std::string> filenameArg(filenameArgChar,
        "filename", filenameDescription, filenameRequired, defaultFilename, "string", parser);
        
        // Gridlength option
        std::string gridlengthArgChar = "n";
        unsigned int defaultGridlength = 0;
        bool gridlengthRequired = true;
        std::string gridlengthDescription = "Gridlength n of nxnxnxn grid of values.";
        TCLAP::ValueArg<unsigned int> gridlengthArg(gridlengthArgChar,
        "gridlength", gridlengthDescription, gridlengthRequired, defaultGridlength, "integer", parser);

        // Parse the argv array
        parser.parse(argc, argv);

        // Get the value parsed by each arg
        databaseFilename = filenameArg.getValue();
        gridLength = gridlengthArg.getValue();
    } 
    catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
    }
    
    // Run
    mihaila2014::generateDatabase<double>(databaseFilename, gridLength, comm);
    
    // MPI finalize
    MPI_Finalize();
    
    // Return
    return 0;
}