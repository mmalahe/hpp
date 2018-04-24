/** @file mihaila2014SimulateFromSpectralDatabase.cpp
* @author Michael Malahe
* @brief Using the spectral database from Mihaila2014 to run a simulation
*/

#include <iostream>
#include <math.h>
#include <vector>
#include <valarray>
#include <functional>
#include <stdexcept>
#include <stdio.h>
#include <complex>
#include <tclap/CmdLine.h>
#include <hpp/mpiUtils.h>
#include <hpp/hdfUtils.h>
#include <hpp/tensor.h>
#include <hpp/spectralUtils.h>
#include <hpp/crystal.h>
#include <hpp/casesUtils.h>
#include <hpp/spectralUtilsCUDA.h>
#include <hpp/crystalCUDA.h>

template <typename U>
void solve(std::string experimentName, std::string databaseFilename, bool unifiedCoeffOrder, unsigned int refinementMultiplier, unsigned int ncrystals, unsigned int nTerms, std::string outputFilename, bool defaultSeed=false)
{    
    // Experiment parameters
    hpp::Experiment<U> experiment(experimentName);
    
    // Method parameters
    U strainIncrement = 2e-2;
    
    // Derived method parameters
    U dt = std::abs(strainIncrement/experiment.strainRate);
    
    // Crystal properties and initial conditions
    hpp::CrystalPropertiesCUDA<U,12> props(hpp::defaultCrystalProperties<U>());
    hpp::CrystalInitialConditions<U> init = hpp::defaultCrystalInitialConditions<U>();
    
    // Database
    std::vector<hpp::SpectralDatasetID> dsetIDs = hpp::defaultCrystalSpectralDatasetIDs();
    hpp::SpectralDatabaseUnified<U> db(databaseFilename, dsetIDs, nTerms, refinementMultiplier);
    
    // Create polycrystal
    const CrystalType crystalType = CRYSTAL_TYPE_FCC;
    auto orientationSpaceResolution = hpp::orientationSpaceResolutionRequiredForNumberOfPoints(ncrystals, hpp::toSymmetryType(crystalType));
    auto polycrystal = hpp::SpectralPolycrystalGSHCUDA<U, crystalType>(props, db, init.s_0, orientationSpaceResolution);
    
    // Evolve
    polycrystal.evolve(experiment.tStart, experiment.tEnd, dt, experiment.F_of_t, experiment.L_of_t);

    // Write the result
    polycrystal.writeResultHDF5(outputFilename);
    
    // Write out true strain history
    std::vector<U> trueStrainHistory = hpp::operator*(experiment.strainRate, polycrystal.getTHistory());        
    H5::H5File outfile(outputFilename, H5F_ACC_RDWR);
    hpp::writeVectorToHDF5Array(outfile, "trueStrainHistory", trueStrainHistory);  
}

int main(int argc, char *argv[])
{    
    // Options
    std::string experimentName;
    std::string databaseFilename;    
    unsigned int nCrystals = 0;
    unsigned int nTerms = 0;
    std::string resultsFilename;
    unsigned int refinementMultiplier;
    bool defaultSeed = false;
    
    // Get options
    try {
        // The parser
        TCLAP::CmdLine parser("Simulate the conditions in the kalidindi1992 problem using a spectral database of responses.", ' ', "0.3");
        
        // Experiment name option
        std::string expNameArgChar = "e";
        std::string defaultExpname = "kalidindi1992_simple_shear";
        bool expNameRequired = true;
        std::string expNameDescription = "Experiment name";
        TCLAP::ValueArg<std::string> expNameArg(expNameArgChar,
        "experimentname", expNameDescription, expNameRequired, defaultExpname, "string", parser);
        
        // Filename option
        std::string databaseFilenameArgChar = "i";
        std::string defaultDatabaseFilename = "database.hdf5";
        bool databaseFilenameRequired = true;
        std::string databaseFilenameDescription = "Spectral database filename";
        TCLAP::ValueArg<std::string> databaseFilenameArg(databaseFilenameArgChar,
        "databasefilename", databaseFilenameDescription, databaseFilenameRequired, defaultDatabaseFilename, "string", parser);
        
        // Number of crystals option
        std::string nCrystalsArgChar = "n";
        unsigned int defaultNCrystals = 0;
        bool nCrystalsRequired = true;
        std::string nCrystalsDescription = "Number of crystals in the polycrystal";
        TCLAP::ValueArg<unsigned int> nCrystalsArg(nCrystalsArgChar,
        "ncrystals", nCrystalsDescription, nCrystalsRequired, defaultNCrystals, "integer", parser);
        
        // Number of crystals option
        std::string nTermsArgChar = "t";
        unsigned int defaultNTerms = 0;
        bool nTermsRequired = true;
        std::string nTermsDescription = "Number of Fourier terms to use";
        TCLAP::ValueArg<unsigned int> nTermsArg(nTermsArgChar,
        "nterms", nTermsDescription, nTermsRequired, defaultNTerms, "integer", parser);
        
        // Output filename option
        std::string filenameArgChar = "o";
        std::string defaultFilename = "results.txt";
        bool filenameRequired = true;
        std::string filenameDescription = "Output filename";
        TCLAP::ValueArg<std::string> filenameArg(filenameArgChar,
        "outputfilename", filenameDescription, filenameRequired, defaultFilename, "string", parser);
        
        // Spectral refinement option
        std::string argChar = "r";
        unsigned int defaultVal = 1;
        bool required = false;
        std::string description = "Spectral refinement multiplier";
        TCLAP::ValueArg<unsigned int> refinementMultiplierArg(argChar,
        "refinementmultiplier", description, required, defaultVal, "integer", parser);
        
        // Whether or not to use consistent random seed between runs
        std::string defaultSeedArgChar = "d";
        std::string defaultSeedDescription = "Use a consistent random seed between runs";
        TCLAP::SwitchArg defaultSeedArg(defaultSeedArgChar, "defaultseed", defaultSeedDescription, parser);
        
        // Parse the argv array
        parser.parse(argc, argv);

        // Get the value parsed by each arg
        experimentName = expNameArg.getValue();
        databaseFilename = databaseFilenameArg.getValue();
        nCrystals = nCrystalsArg.getValue();
        nTerms = nTermsArg.getValue();
        resultsFilename = filenameArg.getValue();
        refinementMultiplier = refinementMultiplierArg.getValue();
        defaultSeed = defaultSeedArg.getValue();
    } 
    catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        throw e;
    }
    
    // Run
    solve<float>(experimentName, databaseFilename, refinementMultiplier, nCrystals, nTerms, resultsFilename, defaultSeed);
    
    // Return
    return 0;
}