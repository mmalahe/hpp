/** @file mihaila2014EvaluateCompressionError.cpp
* @author Michael Malahe
* @brief Evaluate the error incurred from the spectral compression of the database
*/

#include <hpp/tensor.h>
#include <hpp/crystal.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <hpp/mpiUtils.h>
#include <hpp/hdfUtils.h>
#include <hpp/spectralUtils.h>
#include <tclap/CmdLine.h>
#include <complex>

int main(int argc, char *argv[]) 
{
    // MPI init
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    
    // Options
    std::string rawDatabaseFilename;
    std::string spectralDatabaseFilename;
    std::string errorDatabaseFilename;
    unsigned int nTerms = 0;
    unsigned int refineMult = 1;
    std::vector<int> axisSlice;
    bool unifiedCoeffOrder = false;
    std::string outFilename;
    
    // Get options
    try {
        // The parser
        TCLAP::CmdLine parser("Evaluate the error incurred from the spectral compression of the database.", ' ', "0.3");
        
        // Raw database filename option
        std::string rawDatabaseFilenameArgChar = "d";
        std::string defaultRawDatabaseFilename = "rawDatabase.hdf5";
        bool rawDatabaseFilenameRequired = true;
        std::string rawDatabaseFilenameDescription = "Raw database filename";
        TCLAP::ValueArg<std::string> rawDatabaseFilenameArg(rawDatabaseFilenameArgChar,
        "rawdatabasefilename", rawDatabaseFilenameDescription, rawDatabaseFilenameRequired, defaultRawDatabaseFilename, "string", parser);
        
        // Spectral database filename option
        std::string spectralDatabaseFilenameArgChar = "s";
        std::string defaultSpectralDatabaseFilename = "spectralDatabase.hdf5";
        bool spectralDatabaseFilenameRequired = true;
        std::string spectralDatabaseFilenameDescription = "Spectral database filename";
        TCLAP::ValueArg<std::string> spectralDatabaseFilenameArg(spectralDatabaseFilenameArgChar,
        "spectraldatabasefilename", spectralDatabaseFilenameDescription, spectralDatabaseFilenameRequired, defaultSpectralDatabaseFilename, "string", parser);
        
        // Errors database filename option
        std::string errorDatabaseFilenameArgChar = "e";
        std::string defaultErrorDatabaseFilename = "errorDatabase.hdf5";
        bool errorDatabaseFilenameRequired = true;
        std::string errorDatabaseFilenameDescription = "Spectral database filename";
        TCLAP::ValueArg<std::string> errorDatabaseFilenameArg(errorDatabaseFilenameArgChar,
        "errordatabasefilename", errorDatabaseFilenameDescription, errorDatabaseFilenameRequired, defaultErrorDatabaseFilename, "string", parser);
        
        // Number of terms option
        std::string nTermsArgChar = "t";
        unsigned int defaultNTerms = 0;
        bool nTermsRequired = true;
        std::string nTermsDescription = "Number of Fourier terms to use";
        TCLAP::ValueArg<unsigned int> nTermsArg(nTermsArgChar,
        "nTerms", nTermsDescription, nTermsRequired, defaultNTerms, "integer", parser);
        
        // Spectral refinement option
        std::string argChar = "r";
        unsigned int defaultVal = 1;
        bool required = false;
        std::string description = "Spectral refinement multiplier";
        TCLAP::ValueArg<unsigned int> refinementMultiplierArg(argChar,
        "refinementmultiplier", description, required, defaultVal, "integer", parser);
        
        // Slice option
        argChar = "a";
        required = false;
        description = "Axis slice specification of the form [axis, otherCoord1, otherCoord2, ..., otherCoordN]";
        TCLAP::MultiArg<int> axisSliceArg(argChar, "axis", description, required, "integer", parser);

        // Output filename option
        argChar = "o";
        description = "Output filename";
        required = true;
        std::string defaultOutfilename = "spectralError.txt";
        TCLAP::ValueArg<std::string> outFilenameArg(argChar, "outfile", description, required, defaultOutfilename, "string", parser);
        
        // Whether or not to use a unified coefficient ordering
        std::string unifiedCoeffOrderArgChar = "u";
        std::string unifiedCoeffOrderDescription = "Use a single ordering for all Fourier coefficients";
        TCLAP::SwitchArg unifiedCoeffOrderArg(unifiedCoeffOrderArgChar, "unifiedcoefforder", unifiedCoeffOrderDescription, parser);
        
        // Parse the argv array
        parser.parse(argc, argv);

        // Get the value parsed by each arg
        rawDatabaseFilename = rawDatabaseFilenameArg.getValue();
        spectralDatabaseFilename = spectralDatabaseFilenameArg.getValue();
        errorDatabaseFilename = errorDatabaseFilenameArg.getValue();
        nTerms = nTermsArg.getValue();
        refineMult = refinementMultiplierArg.getValue();
        axisSlice = axisSliceArg.getValue();
        unifiedCoeffOrder = unifiedCoeffOrderArg.getValue();
        outFilename = outFilenameArg.getValue();
    } 
    catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
    }
    
    // Default dataset choices
    std::vector<std::string> dsetBasenames = {"sigma_prime", "W_p", "gammadot_abs_sum"};
    
    // Run
    if (axisSlice.size() == 0) {
        if (unifiedCoeffOrder) {
            hpp::evaluateSpectralCompressionErrorFullUnified(rawDatabaseFilename, spectralDatabaseFilename, errorDatabaseFilename, nTerms, outFilename, comm);
        }
        else {
            hpp::evaluateSpectralCompressionErrorFull(rawDatabaseFilename, spectralDatabaseFilename, errorDatabaseFilename, nTerms, outFilename, comm);
        }
    }
    else {
        if (unifiedCoeffOrder) {
            hpp::evaluateSpectralCompressionErrorAxisSliceUnified(rawDatabaseFilename, spectralDatabaseFilename, dsetBasenames, nTerms, refineMult, axisSlice, outFilename, comm);
        }
        else {
            hpp::evaluateSpectralCompressionErrorAxisSlice(rawDatabaseFilename, spectralDatabaseFilename, dsetBasenames, nTerms, refineMult, axisSlice, outFilename, comm);
        }
    }
    
    // MPI finalize
    MPI_Finalize();
    
    // Return
    return 0;
}