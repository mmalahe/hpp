/** @file mihaila2014GenerateSpectralDatabase.cpp
* @author Michael Malahe
* @brief Convert raw output database to spectral database
*/

#include <iostream>
#include <string>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <tclap/CmdLine.h>
#include <fftw3-mpi.h>
#include <hpp/tensor.h>
#include <hpp/mpiUtils.h>
#include <hpp/hdfUtils.h>
#include <hpp/spectralUtils.h>
#include <hpp/crystal.h>

namespace mihaila2014
{

inline double complexMag(double im, double re) {
    return std::sqrt(std::pow(im,2) + std::pow(re,2));
}
    
void readPerformDFTThenWriteOrderedCoeffs(hpp::HDF5Handler& infile, std::string dsetInName, hpp::FFTWConfigRealND& cfg, std::vector<hsize_t> componentIdx, hpp::HDF5Handler& outfile, hid_t dsetOutCoords, hid_t dsetOutVals, unsigned int nCoeffs, MPI_Comm comm) 
{   
    // Comm
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // File transfer property lists
    hid_t plist_in = infile.getPropertyListTransferIndependent();
    hid_t plist_out = outfile.getPropertyListTransferIndependent();
    
    // Input dataset
    hid_t dsetIn = infile.getDataset(dsetInName);
    
    // Input values
    hpp::HDFReadWriteParamsC parms;
    std::vector<hsize_t> gridOffset = {(hsize_t)cfg.local0Start,0,0,0};
    parms.dataOffset = componentIdx;
    parms.dataOffset.insert(parms.dataOffset.end(), gridOffset.begin(), gridOffset.end());
    for (unsigned int i=0; i<componentIdx.size(); i++) {
        parms.dataCount.push_back(1);
    }
    parms.dataCount.insert(parms.dataCount.end(), cfg.realDimsLocal.begin(), cfg.realDimsLocal.end());
    parms.datatype = H5Dget_type(dsetIn);    
    std::vector<double> inVals(cfg.nLocalReal);
    
    // Read data
    if (comm_rank == 0) std::cout << "Reading data..." << std::endl;
    hpp::readHDF5SimpleArray(dsetIn, plist_in, parms, inVals.data());   
    if (comm_rank == 0) std::cout << "...done reading data." << std::endl;   
    
    // Place the data into the padded FFTW array
    for (unsigned int flatIdxData=0; flatIdxData<cfg.nLocalReal; flatIdxData++) {
        std::vector<unsigned int> gridIdx = hpp::unflatC(flatIdxData, cfg.realDimsLocal);
        unsigned int flatIdxFFTW = hpp::flatC(gridIdx, cfg.realDimsPaddedLocal);
        cfg.in[flatIdxFFTW] = inVals[flatIdxData];
    }

    // Execute FFT
    if (comm_rank == 0) std::cout << "Doing FFT..." << std::endl;
    fftw_execute(cfg.forwardPlan);
    if (comm_rank == 0) std::cout << "...done with FFT" << std::endl;
    
    // Scale correctly
    for (int i=0; i<cfg.nLocalComplex; i++) {
        cfg.out[i][0] /= cfg.NReal;
        cfg.out[i][1] /= cfg.NReal;
    }
    
    // Execute IFFT and check
    fftw_execute(cfg.backwardPlan);
    double closeEnough = 1000*cfg.NReal*std::numeric_limits<double>::epsilon();
    for (unsigned int flatIdxData=0; flatIdxData<cfg.nLocalReal; flatIdxData++) {
        std::vector<unsigned int> gridIdx = hpp::unflatC(flatIdxData, cfg.realDimsLocal);
        unsigned int flatIdxFFTW = hpp::flatC(gridIdx, cfg.realDimsPaddedLocal);
        if (std::abs(cfg.in[flatIdxFFTW]-cfg.backin[flatIdxFFTW]) > closeEnough) {
            std::cout << "WARNING: error in IFFT greater than " << closeEnough << std::endl;
            std::cout << flatIdxData << " " << cfg.in[flatIdxFFTW] << " != " << cfg.backin[flatIdxFFTW] << std::endl;
        }
    }
    
    // Components
    std::vector<double> localReVec(cfg.nLocalComplex);
    std::vector<double> localImVec(cfg.nLocalComplex);
    for (int i=0; i<cfg.nLocalComplex; i++) {
        localReVec[i] = cfg.out[i][0];
        localImVec[i] = cfg.out[i][1];
    }
    
    // Confirm sizes line up
    if (hpp::MPISum(cfg.nLocalComplex, comm) != cfg.NComplex) throw std::runtime_error("Sizes don't match.");
    
    // Gather full vector on root
    std::vector<double> Re = hpp::MPIConcatOnRoot(localReVec, comm);
    std::vector<double> Im = hpp::MPIConcatOnRoot(localImVec, comm);
    
    // Limit number of coeffs to the number that are actually available
    nCoeffs = std::min(cfg.NComplex, nCoeffs);
    
    // Get output dimensions
    std::vector<hsize_t> dsetOutCoordsDims = hpp::getDatasetDims(dsetOutCoords);
    std::vector<hsize_t> coordsDims = {dsetOutCoordsDims.back()};
    
    // Remainder of operations happen on root only    
    if (comm_rank == 0) {
        // Check size
        if (Re.size() != cfg.NComplex) throw std::runtime_error("Incorrect size.");
        if (Im.size() != cfg.NComplex) throw std::runtime_error("Incorrect size.");
        
        // Unsorted indices
        std::vector<size_t> idxs(Re.size());
        std::iota(idxs.begin(), idxs.end(), 0);
        
        // Get magnitudes
        std::cout << "Calculating mags..." << std::endl;
        std::vector<double> mags (Re.size());
        for (unsigned int i=0; i<Re.size(); i++) {
            mags[i] = complexMag(Re[i], Im[i]);
        }
        std::cout << "..done calculating mags." << std::endl;
        
        // Sort indices based on descending complex magnitude
        std::cout << "Sorting..." << std::endl;
        std::sort(idxs.begin(), idxs.end(), 
        [&mags](size_t i1, size_t i2) 
        {return (mags[i1] > mags[i2]);});
        std::cout << "...done sorting" << std::endl;
        
        // Buffer writes to datasets
        std::cout << "Buffering writes..." << std::endl;
        std::vector<unsigned short> coordsArray(nCoeffs*4);
        std::vector<hpp::hdf_complex_t> valsArray(nCoeffs);
        for (unsigned int iCoeff=0; iCoeff<nCoeffs; iCoeff++) {
            // Write the grid coordinates
            int flatIdx = idxs[iCoeff];
            std::vector<int> idx = hpp::unflatC(flatIdx, cfg.complexDims);
            coordsArray[iCoeff*4 + 0] = (unsigned short)idx[0];
            coordsArray[iCoeff*4 + 1] = (unsigned short)idx[1];
            coordsArray[iCoeff*4 + 2] = (unsigned short)idx[2];
            coordsArray[iCoeff*4 + 3] = (unsigned short)idx[3];            
            
            // Write the coefficient value
            valsArray[iCoeff].r = Re[flatIdx];
            valsArray[iCoeff].i = Im[flatIdx];            
        }
        std::cout << "...done buffering writes." << std::endl;
        
        // Write to datasets
        std::vector<hsize_t> coordsGridOffset = componentIdx;
        coordsGridOffset.push_back(0);
        std::vector<hsize_t> coordsArrayCount(componentIdx.size(), 1);
        coordsArrayCount.push_back(nCoeffs);
        std::cout << "Writing coords..." << std::endl;
        hpp::writeMultipleHDF5Arrays(dsetOutCoords, plist_out, coordsGridOffset, coordsDims, coordsArrayCount, coordsArray.data());
        std::cout << "...done writing coords." << std::endl;
        
        std::cout << "Writing vals..." << std::endl;
        std::vector<hsize_t> valsGridOffset = componentIdx;
        std::vector<hsize_t> valsArrayCount = {nCoeffs};
        hpp::writeSingleHDF5Array(dsetOutVals, plist_out, valsGridOffset, valsArrayCount, valsArray.data());
        std::cout << "...done writing vals." << std::endl;
    }
}  

void generateSpectralDatabase(std::string rawDatabaseFilename, std::string spectralDatabaseFilename, unsigned int nCoeffs, MPI_Comm comm) {    
    // Comm
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Open handle to input file and get transfer property list
    hpp::HDF5Handler infile(rawDatabaseFilename, comm, false);
    
    // Input dataset names
    std::string gammadotAbsSumNameIn = "gammadot_abs_sum";
    std::string sigmaPrimeNameIn = "sigma_prime";
    std::string WpNameIn = "W_p";
    
    // Common grid dimensions
    std::vector<hsize_t> gridDims = hpp::getDatasetDims(infile.getDataset(gammadotAbsSumNameIn));
    unsigned int nDims = gridDims.size();
    
    // Maximum number of Fourier coefficients that will be returned by FFTW
    unsigned int maxNCoeffs = 1;
    for (unsigned int i=0; i<nDims-1; i++) {
        maxNCoeffs *= gridDims[i];
    }
    maxNCoeffs *= (gridDims[nDims-1]/2+1);
    nCoeffs = std::min(nCoeffs, maxNCoeffs);
    
    // FFTW SETUP //
    std::vector<ptrdiff_t> gridDimsPtrDiff(gridDims.begin(), gridDims.end());
    hpp::FFTWConfigRealND cfg = hpp::prepareFFTWConfigRealND(gridDimsPtrDiff, comm);
    
    // Open handle to output file and get transfer property list
    hpp::HDF5Handler outfile(spectralDatabaseFilename, comm, true);
    hid_t plist_out = outfile.getPropertyListTransferIndependent();
    
    // Write dimensions
    std::vector<hsize_t> nDimsArray = {nDims};
    
    hid_t dsetGridDims = outfile.createDataset<unsigned short int>("grid_dims", nDimsArray);
    hid_t dsetGridStarts = outfile.createDataset<double>("grid_starts", nDimsArray);
    hid_t dsetGridEnds = outfile.createDataset<double>("grid_ends", nDimsArray);
    
    if (comm_rank == 0) {
        std::vector<unsigned short int> gridDimsOut(gridDims.begin(), gridDims.end());
        /// @todo match these to values read from the raw dataset
        std::vector<double> gridStarts = {0,0,0,0};
        std::vector<double> gridEnds = {2*M_PI, 2*M_PI, 2*M_PI, 2*M_PI};
        hpp::writeSingleHDF5Array(dsetGridDims, plist_out, nDimsArray, gridDimsOut.data());
        hpp::writeSingleHDF5Array(dsetGridStarts, plist_out, nDimsArray, gridStarts.data());
        hpp::writeSingleHDF5Array(dsetGridEnds, plist_out, nDimsArray, gridEnds.data());
    }
    
    // Output dataset names
    std::string gammadotAbsSumNameBase = "gammadot_abs_sum";
    std::string sigmaPrimeNameBase = "sigma_prime";
    std::string WpNameBase = "W_p";
    
    std::string coordsSuffix = HPP_DEFAULT_COORDS_SUFFIX;
    std::string gammadotAbsSumNameCoords = gammadotAbsSumNameBase + coordsSuffix;
    std::string sigmaPrimeNameCoords = sigmaPrimeNameBase + coordsSuffix;
    std::string WpNameCoords = WpNameBase + coordsSuffix;
    
    std::string valsSuffix = HPP_DEFAULT_COEFFS_SUFFIX;
    std::string gammadotAbsSumNameVals = gammadotAbsSumNameBase + valsSuffix;
    std::string sigmaPrimeNameVals = sigmaPrimeNameBase + valsSuffix;
    std::string WpNameVals = WpNameBase + valsSuffix;
    
    // Create output datasets
    std::vector<hsize_t> scalarCoordDims = {nCoeffs, nDims};
    std::vector<hsize_t> scalarValDims = {nCoeffs};
    hid_t dsetGammadotAbsSumCoords = outfile.createDataset<unsigned short int>(gammadotAbsSumNameCoords, scalarCoordDims);
    hid_t dsetGammadotAbsSumVals = outfile.createDataset<hpp::hdf_complex_t>(gammadotAbsSumNameVals, scalarValDims);
    
    std::vector<hsize_t> tensorCoordDims = {3, 3, nCoeffs, nDims};
    std::vector<hsize_t> tensorValDims = {3, 3, nCoeffs};
    hid_t dsetSigmaPrimeCoords = outfile.createDataset<unsigned short int>(sigmaPrimeNameCoords, tensorCoordDims);
    hid_t dsetSigmaPrimeVals = outfile.createDataset<hpp::hdf_complex_t>(sigmaPrimeNameVals, tensorValDims);
    hid_t dsetWpCoords = outfile.createDataset<unsigned short int>(WpNameCoords, tensorCoordDims);
    hid_t dsetWpVals = outfile.createDataset<hpp::hdf_complex_t>(WpNameVals, tensorValDims);
    
    // Do FFTs
    
    // Scalar
    std::vector<hsize_t> scalarComponentIdx;
    readPerformDFTThenWriteOrderedCoeffs(infile, gammadotAbsSumNameIn, cfg, scalarComponentIdx, outfile, dsetGammadotAbsSumCoords, dsetGammadotAbsSumVals, nCoeffs, comm);
    if (comm_rank==0) std::cout << "Done with " << gammadotAbsSumNameBase << std::endl;
    
    // Tensors
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            std::vector<hsize_t> tensorComponentIdx = {(hsize_t)i, (hsize_t)j};
            readPerformDFTThenWriteOrderedCoeffs(infile, sigmaPrimeNameIn, cfg, tensorComponentIdx, outfile, dsetSigmaPrimeCoords, dsetSigmaPrimeVals, nCoeffs, comm);
            readPerformDFTThenWriteOrderedCoeffs(infile, WpNameIn, cfg, tensorComponentIdx, outfile, dsetWpCoords, dsetWpVals, nCoeffs, comm);
            if (comm_rank==0) std::cout << "Done with " << sigmaPrimeNameBase << "(" << i << "," << j << ")." << std::endl;
            if (comm_rank==0) std::cout << "Done with " << WpNameBase << "(" << i << "," << j << ")." << std::endl;
        }
    }
    
    // Free
    hpp::destroyConfigRealND(cfg); 
}

void readHDFDataToFFTWInput4D(hid_t dsetIn, hid_t plist_in, std::vector<hsize_t> componentIdx, hpp::FFTWConfigRealND& cfg){    
    // HDF Read parameters
    hpp::HDFReadWriteParamsC parms;
    std::vector<hsize_t> gridOffset = {(hsize_t)cfg.local0Start,0,0,0};
    parms.dataOffset = componentIdx;
    parms.dataOffset.insert(parms.dataOffset.end(), gridOffset.begin(), gridOffset.end());
    for (unsigned int i=0; i<componentIdx.size(); i++) {
        parms.dataCount.push_back(1);
    }
    parms.dataCount.insert(parms.dataCount.end(), cfg.realDimsLocal.begin(), cfg.realDimsLocal.end());
    parms.datatype = H5Dget_type(dsetIn); 

    // Buffer to read into
    std::vector<double> inVals(cfg.nLocalReal);
    
    // Read data into buffer
    hpp::readHDF5SimpleArray(dsetIn, plist_in, parms, inVals.data());  
    
    // Move data from the buffer to the padded FFTW array
    for (unsigned int flatIdxData=0; flatIdxData<cfg.nLocalReal; flatIdxData++) {
        std::vector<unsigned int> gridIdx = hpp::unflatC(flatIdxData, cfg.realDimsLocal);
        unsigned int flatIdxFFTW = hpp::flatC(gridIdx, cfg.realDimsPaddedLocal);
        cfg.in[flatIdxFFTW] = inVals[flatIdxData];
    }
}

void executeAndCheckFFTW(hpp::FFTWConfigRealND& cfg) {
    // Execute FFT
    fftw_execute(cfg.forwardPlan);
    
    // Scale correctly
    for (int i=0; i<cfg.nLocalComplex; i++) {
        cfg.out[i][0] /= cfg.NReal;
        cfg.out[i][1] /= cfg.NReal;
    }
    
    // Execute IFFT and check
    fftw_execute(cfg.backwardPlan);
    double closeEnough = 1000*cfg.NReal*std::numeric_limits<double>::epsilon();
    for (unsigned int flatIdxData=0; flatIdxData<cfg.nLocalReal; flatIdxData++) {
        std::vector<unsigned int> gridIdx = hpp::unflatC(flatIdxData, cfg.realDimsLocal);
        unsigned int flatIdxFFTW = hpp::flatC(gridIdx, cfg.realDimsPaddedLocal);
        if (std::abs(cfg.in[flatIdxFFTW]-cfg.backin[flatIdxFFTW]) > closeEnough) {
            std::cout << "WARNING: error in IFFT greater than " << closeEnough << std::endl;
            std::cout << flatIdxData << " " << cfg.in[flatIdxFFTW] << " != " << cfg.backin[flatIdxFFTW] << std::endl;
        }
    }
}

void readPerformDFTThenWriteOrderedCoeffsUnified(hpp::HDF5Handler& infile, hid_t dsetOutCoords, std::vector<hpp::SpectralDatasetID> spectralDatasetIDs , hpp::FFTWConfigRealND& cfg, hpp::HDF5Handler& outfile, unsigned int nCoeffs, MPI_Comm comm) 
{   
    // Comm
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Confirm sizes line up
    if (hpp::MPISum(cfg.nLocalComplex, comm) != cfg.NComplex) throw std::runtime_error("MPI array sizes don't match.");
    
    // Limit number of coeffs to the number that are actually available
    nCoeffs = std::min(cfg.NComplex, nCoeffs);
    
    // Create the output datasets and store handles to the datasets
    std::vector<hsize_t> scalarValDims = {nCoeffs};
    std::map<hpp::SpectralDatasetID, hid_t> HDFOutputDsetIDs;
    for (auto dsetID : spectralDatasetIDs) {
        std::string dsetName = getDefaultCoeffsDatasetName(dsetID);
        HDFOutputDsetIDs[dsetID] = outfile.createDataset<hpp::hdf_complex_t>(dsetName, scalarValDims);
    }
    
    // File transfer property lists
    hid_t plist_in = infile.getPropertyListTransferIndependent();
    hid_t plist_out = outfile.getPropertyListTransferIndependent();
    
    // 0.a GET THE COMPONENT SUMMED MAGNITUDES AND AVERAGE MAGNITUDES
    double sigmaMagSum = 0.0;
    double WpMagSum = 0.0;
    std::vector<double> componentSums(spectralDatasetIDs.size(), 0.0);
    std::vector<double> componentMagSums(spectralDatasetIDs.size(), 0.0);    
    for (unsigned int iDset=0; iDset<spectralDatasetIDs.size(); iDset++) {
        auto dsetID = spectralDatasetIDs[iDset];
        std::string dsetInName = dsetID.baseName;
        std::vector<hsize_t> componentIdx(dsetID.component.begin(), dsetID.component.end());
        hid_t dsetIn = infile.getDataset(dsetInName);        
        
        // Read input data
        readHDFDataToFFTWInput4D(dsetIn, plist_in, componentIdx, cfg);
        
        // Get the sum and sum of magnitudes of each component
        for (int i=0; i<cfg.nLocalReal; i++) {
            auto val = cfg.in[i];
            auto mag = std::abs(val);
            componentSums[iDset] += val;            
            componentMagSums[iDset] += mag;
        }
    }
    
    // Calculate mean
    for (auto&& componentSum : componentSums) {
        componentSum = hpp::MPISum(componentSum, comm);
    }
    std::vector<double> componentAvgs;
    for (const auto& componentSum : componentSums) {
        componentAvgs.push_back(componentSum/cfg.NReal);
    }
    if (comm_rank == 0) {
        std::cout << "Component averages: ";
        hpp::operator<<(std::cout, componentAvgs);
        std::cout << std::endl;
    }    
    
    // Calculate mean of magnitudes
    for (auto&& componentMagSum : componentMagSums) {
        componentMagSum = hpp::MPISum(componentMagSum, comm);
    }
    std::vector<double> componentAvgMags;
    for (const auto& componentMagSum : componentMagSums) {
        componentAvgMags.push_back(componentMagSum/cfg.NReal);
    }
    if (comm_rank == 0) {
        std::cout << "Component magnitude averages: ";
        hpp::operator<<(std::cout, componentAvgMags);
        std::cout << std::endl;
    }
    
    // 0.b GET THE COMPONENT MAGNITUDE VARIANCES
    std::vector<double> componentVarianceSums(spectralDatasetIDs.size(), 0.0);
    for (unsigned int iDset=0; iDset<spectralDatasetIDs.size(); iDset++) {
        auto dsetID = spectralDatasetIDs[iDset];
        std::string dsetInName = dsetID.baseName;
        std::vector<hsize_t> componentIdx(dsetID.component.begin(), dsetID.component.end());
        hid_t dsetIn = infile.getDataset(dsetInName);        
        
        // Read input data
        readHDFDataToFFTWInput4D(dsetIn, plist_in, componentIdx, cfg);
        
        // Get the variance of the raw component to scale by
        for (int i=0; i<cfg.nLocalReal; i++) {            
            auto val = std::abs(cfg.in[i]);
            componentVarianceSums[iDset] += std::pow(val-componentAvgs[iDset], 2.0);
        }    
    }
    for (auto&& componentVarianceSum : componentVarianceSums) {
        componentVarianceSum = hpp::MPISum(componentVarianceSum, comm);
    }
    std::vector<double> componentVariances;
    for (const auto& componentVarianceSum : componentVarianceSums) {
        componentVariances.push_back(componentVarianceSum/cfg.NReal);
    }
    
    // Get standard deviations from variances
    std::vector<double> componentStandardDeviations(componentVariances);
    for (auto&& componentStandardDeviation : componentStandardDeviations) {
        componentStandardDeviation = std::sqrt(componentStandardDeviation);
    }
    
    if (comm_rank == 0) {
        std::cout << "Component standard deviations: ";
        hpp::operator<<(std::cout, componentStandardDeviations);
        std::cout << std::endl;
    }
    
    // 1. CREATE LIST OF THE SQUARED MAGNITUDES OF THE COMPONENTS
    std::vector<double> orderingMagnitudes(cfg.nLocalComplex, 0.0);
    for (unsigned int iDset=0; iDset<spectralDatasetIDs.size(); iDset++) {
        auto dsetID = spectralDatasetIDs[iDset];
        std::string dsetInName = dsetID.baseName;
        std::vector<hsize_t> componentIdx(dsetID.component.begin(), dsetID.component.end());
        hid_t dsetIn = infile.getDataset(dsetInName);        
        
        // Read input data
        readHDFDataToFFTWInput4D(dsetIn, plist_in, componentIdx, cfg);
        
        // Pre-scale data to mean zero, variance 1
        for (int i=0; i<cfg.nLocalReal; i++) {            
            cfg.in[i] = (cfg.in[i]-componentAvgs[iDset])/componentStandardDeviations[iDset];
        }    
        
        // Execute and check FFTW
        executeAndCheckFFTW(cfg);
        
        // Add the squared magnitudes
        for (int i=0; i<cfg.nLocalComplex; i++) {
            auto mag = std::sqrt(std::pow(cfg.out[i][0],2.0)+std::pow(cfg.out[i][1],2.0));
            orderingMagnitudes[i] += std::pow(mag, 2.0);
        }
    }
    
    // Gather full vector on root
    std::vector<double> orderingMagnitudesGlobalOnRoot = hpp::MPIConcatOnRoot(orderingMagnitudes, comm);
    
    // 2. ORDER THE SQUARED MAGNITUDES ON THE ROOT AND BROADCAST RESULT
    std::vector<size_t> idxs;
    if (comm_rank == 0) {
        // Check size
        if (orderingMagnitudesGlobalOnRoot.size() != cfg.NComplex) throw std::runtime_error("Incorrect size.");
        
        // Unsorted indices
        idxs.resize(orderingMagnitudesGlobalOnRoot.size());
        std::iota(idxs.begin(), idxs.end(), 0);
        
        // Sort indices based on descending magnitude
        std::sort(idxs.begin(), idxs.end(), 
        [&orderingMagnitudesGlobalOnRoot](size_t i1, size_t i2) 
        {return (orderingMagnitudesGlobalOnRoot[i1] > orderingMagnitudesGlobalOnRoot[i2]);});
        
        // Free the memory from the magnitudes
        orderingMagnitudesGlobalOnRoot.resize(0);
        
        // Truncate to only the number of coefficients we want
        idxs.resize(nCoeffs);
    }
    
    // Propagate the ordering indices to all processors
    idxs = hpp::MPIBroadcastFromRoot(idxs, comm);    
    
    // Write the coordinates in the correct order on root only
    if (comm_rank == 0) {
        // Buffer writes to dataset
        std::vector<unsigned short> coordsArray(nCoeffs*4);
        for (unsigned int iCoeff=0; iCoeff<nCoeffs; iCoeff++) {
            // Write the grid coordinates
            int flatIdx = idxs[iCoeff];
            std::vector<int> idx = hpp::unflatC(flatIdx, cfg.complexDims);
            coordsArray[iCoeff*4 + 0] = (unsigned short)idx[0];
            coordsArray[iCoeff*4 + 1] = (unsigned short)idx[1];
            coordsArray[iCoeff*4 + 2] = (unsigned short)idx[2];
            coordsArray[iCoeff*4 + 3] = (unsigned short)idx[3];            
        }
        
        // Coords output dataset
        std::vector<hsize_t> dsetOutCoordsDims = hpp::getDatasetDims(dsetOutCoords);
        std::vector<hsize_t> coordsDims = {dsetOutCoordsDims.back()};
        
        // Write to dataset
        std::vector<hsize_t> componentIdx;//no component, just scalar
        std::vector<hsize_t> coordsGridOffset = componentIdx;
        coordsGridOffset.push_back(0);
        std::vector<hsize_t> coordsArrayCount(componentIdx.size(), 1);
        coordsArrayCount.push_back(nCoeffs);
        hpp::writeMultipleHDF5Arrays(dsetOutCoords, plist_out, coordsGridOffset, coordsDims, coordsArrayCount, coordsArray.data());
    }
    
    /* Perform the DFT again, and this time store the coefficients in the 
     * correct order
     */
    for (unsigned int iDset=0; iDset<spectralDatasetIDs.size(); iDset++) {
        auto dsetID = spectralDatasetIDs[iDset];
        // Input dataset details
        std::string dsetInName = dsetID.baseName;
        std::vector<hsize_t> componentIdx(dsetID.component.begin(), dsetID.component.end());
        hid_t dsetIn = infile.getDataset(dsetInName);
        
        // Read input data
        readHDFDataToFFTWInput4D(dsetIn, plist_in, componentIdx, cfg);
        
        // Execute and check FFTW
        executeAndCheckFFTW(cfg);        
        
        // Components
        std::vector<double> localReVec(cfg.nLocalComplex);
        std::vector<double> localImVec(cfg.nLocalComplex);
        for (int i=0; i<cfg.nLocalComplex; i++) {
            localReVec[i] = cfg.out[i][0];
            localImVec[i] = cfg.out[i][1];
        }
    
        // Gather full vector on root
        std::vector<double> Re = hpp::MPIConcatOnRoot(localReVec, comm);
        std::vector<double> Im = hpp::MPIConcatOnRoot(localImVec, comm);
        
        // Write out coefficients on root
        if (comm_rank == 0) {        
            hid_t dsetOutVals = HDFOutputDsetIDs[dsetID]; 
            std::vector<hpp::hdf_complex_t> valsArray(nCoeffs);
            double capturedMag = 0.0;
            for (unsigned int iCoeff=0; iCoeff<nCoeffs; iCoeff++) {                
                // Write the coefficient value
                int flatIdx = idxs[iCoeff];  
                valsArray[iCoeff].r = Re[flatIdx];
                valsArray[iCoeff].i = Im[flatIdx];

                // Tally up the captured coefficient magnitudes
                capturedMag += std::sqrt(std::pow(Re[flatIdx],2.0)+std::pow(Im[flatIdx],2.0));
            }
            std::vector<hsize_t> outComponentIdx;//scalar output, no component
            std::vector<hsize_t> valsGridOffset = outComponentIdx;
            std::vector<hsize_t> valsArrayCount = {nCoeffs};
            hpp::writeSingleHDF5Array(dsetOutVals, plist_out, valsGridOffset, valsArrayCount, valsArray.data());
            
            // Diagnostic report
            double percentMagCaptured = 100*capturedMag/componentMagSums[iDset];
            std::cout << dsetID.baseName + hpp::getComponentSuffix(dsetID.component);
            std::cout << " % captured = " << percentMagCaptured << std::endl;
        }
    }
}

void generateSpectralDatabaseUnified(std::string rawDatabaseFilename, std::string spectralDatabaseFilename, unsigned int nCoeffs, MPI_Comm comm) {    
    // Comm
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Open handle to input file and get transfer property list
    hpp::HDF5Handler infile(rawDatabaseFilename, comm, false);
    hid_t plist_in = infile.getPropertyListTransferIndependent();
    
    // Read in number of dimensions
    hid_t dsetGridDimsIn = infile.getDataset("grid_dims");
    std::vector<hsize_t> nDimsArray = hpp::getDatasetDims(dsetGridDimsIn);
    unsigned int nDims = nDimsArray[0];
    
    // Read in grid dimensions
    std::vector<unsigned short int> gridDimsBuffer(nDims);
    hpp::readSingleHDF5Array(dsetGridDimsIn, plist_in, nDimsArray, gridDimsBuffer.data());
    std::vector<hsize_t> gridDims(gridDimsBuffer.begin(), gridDimsBuffer.end());
    
    // Maximum number of Fourier coefficients that will be returned by FFTW
    unsigned int maxNCoeffs = 1;
    for (unsigned int i=0; i<nDims-1; i++) {
        maxNCoeffs *= gridDims[i];
    }
    maxNCoeffs *= (gridDims[nDims-1]/2+1);
    nCoeffs = std::min(nCoeffs, maxNCoeffs);
    
    // FFTW SETUP //
    std::vector<ptrdiff_t> gridDimsPtrDiff(gridDims.begin(), gridDims.end());
    hpp::FFTWConfigRealND cfg = hpp::prepareFFTWConfigRealND(gridDimsPtrDiff, comm);
    
    // Open handle to output file and get transfer property list
    hpp::HDF5Handler outfile(spectralDatabaseFilename, comm, true);
    hid_t plist_out = outfile.getPropertyListTransferIndependent();
    
    // Write dimensions
    hid_t dsetGridDims = outfile.createDataset<unsigned short int>("grid_dims", nDimsArray);
    hid_t dsetGridStarts = outfile.createDataset<double>("grid_starts", nDimsArray);
    hid_t dsetGridEnds = outfile.createDataset<double>("grid_ends", nDimsArray);
    
    if (comm_rank == 0) {
        std::vector<unsigned short int> gridDimsOut(gridDims.begin(), gridDims.end());
        /// @todo match these to values read from the raw dataset
        std::vector<double> gridStarts = {0,0,0,0};
        std::vector<double> gridEnds = {2*M_PI, 2*M_PI, 2*M_PI, 2*M_PI};
        hpp::writeSingleHDF5Array(dsetGridDims, plist_out, nDimsArray, gridDimsOut.data());
        hpp::writeSingleHDF5Array(dsetGridStarts, plist_out, nDimsArray, gridStarts.data());
        hpp::writeSingleHDF5Array(dsetGridEnds, plist_out, nDimsArray, gridEnds.data());
    }
    
    // Create coordinates dataset
    std::string coordsName = HPP_DEFAULT_UNIFIED_COORDS_NAME;
    std::vector<hsize_t> scalarCoordDims = {nCoeffs, nDims};
    hid_t dsetOutCoords = outfile.createDataset<unsigned short int>(coordsName, scalarCoordDims);
    
    // Do FFTs
    std::vector<hpp::SpectralDatasetID> spectralDatasetIDs = hpp::defaultCrystalSpectralDatasetIDs();
    readPerformDFTThenWriteOrderedCoeffsUnified(infile, dsetOutCoords, spectralDatasetIDs, cfg, outfile, nCoeffs, comm);

    // Free
    hpp::destroyConfigRealND(cfg); 
}

} // END NAMESPACE mihaila2014

int main(int argc, char *argv[]) 
{    
    // MPI init
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    
    // Options
    std::string rawDatabaseFilename;
    std::string spectralDatabaseFilename;
    unsigned int nCoeffs;
    bool unifiedCoeffOrder;
    
    // Get options
    try {
        // The parser
        TCLAP::CmdLine parser("Generate the spectral database of responses.", ' ', "0.3");
        
        // Raw database filename option
        std::string rawFilenameArgChar = "i";
        std::string defaultRawFilename = "databaseRaw.hdf5";
        bool rawFilenameRequired = true;
        std::string rawFilenameDescription = "Raw database input filename";
        TCLAP::ValueArg<std::string> rawFilenameArg(rawFilenameArgChar,
        "rawfilename", rawFilenameDescription, rawFilenameRequired, defaultRawFilename, "string", parser);
        
        // Spectral database filename option
        std::string spectralFilenameArgChar = "o";
        std::string defaultSpectralFilename = "databaseSpectral.hdf5";
        bool spectralFilenameRequired = true;
        std::string spectralFilenameDescription = "Spectral database output filename";
        TCLAP::ValueArg<std::string> spectralFilenameArg(spectralFilenameArgChar,
        "spectralfilename", spectralFilenameDescription, spectralFilenameRequired, defaultSpectralFilename, "string", parser);
        
        // Number of coefficients to write
        std::string nCoeffsArgChar = "n";
        unsigned int defaultNCoeffs = 0;
        bool nCoeffsRequired = true;
        std::string nCoeffsDescription = "Write only n coefficients per component, ordered by decreasing magnitude";
        TCLAP::ValueArg<unsigned int> nCoeffsArg(nCoeffsArgChar,
        "ncoeffs", nCoeffsDescription, nCoeffsRequired, defaultNCoeffs, "integer", parser);
        
        // Whether or not to use a unified coefficient ordering
        std::string unifiedCoeffOrderArgChar = "u";
        std::string unifiedCoeffOrderDescription = "Use a single ordering for all Fourier coefficients";
        TCLAP::SwitchArg unifiedCoeffOrderArg(unifiedCoeffOrderArgChar, "unifiedcoefforder", unifiedCoeffOrderDescription, parser);
        
        // Parse the argv array
        parser.parse(argc, argv);

        // Get the value parsed by each arg
        rawDatabaseFilename = rawFilenameArg.getValue();
        spectralDatabaseFilename = spectralFilenameArg.getValue();
        nCoeffs = nCoeffsArg.getValue();
        unifiedCoeffOrder = unifiedCoeffOrderArg.getValue();
    } 
    catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
    }
    
    // Run
    if (unifiedCoeffOrder) {
        mihaila2014::generateSpectralDatabaseUnified(rawDatabaseFilename, spectralDatabaseFilename, nCoeffs, comm);
    }
    else {
        mihaila2014::generateSpectralDatabase(rawDatabaseFilename, spectralDatabaseFilename, nCoeffs, comm);
    }    
    
    // MPI finalize
    MPI_Finalize();
    
    // Return
    return 0;
}
