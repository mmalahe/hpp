/** @file mihaila2014SimulateFromSpectralDatabase.cpp
* @author Michael Malahe
* @brief Using the spectral database from Mihaila2014 to run a simulation
*/

#include <cassert>
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

namespace mihaila2014
{

template <typename U>
void spectralSolve(std::string experimentName, std::string databaseFilename, bool unifiedCoeffOrder, unsigned int refinementMultiplier, unsigned int ncrystalsGlobal, unsigned int nTerms, std::string outputFilename, MPI_Comm comm, bool defaultSeed=false)
{    
    // Experiment parameters
    hpp::Experiment<U> experiment(experimentName);
    
    // Method parameters
    U strainIncrement = 2e-2;
    
    // Derived method parameters
    U dt = std::abs(strainIncrement/experiment.strainRate);
    
    // Crystal properties and initial conditions
    hpp::CrystalProperties<U> props = hpp::defaultCrystalProperties<U>();
    hpp::CrystalInitialConditions<U> init = hpp::defaultCrystalInitialConditions<U>();
    
    // Divide crystals between processes
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    if (comm_size != 1) {
        throw std::runtime_error("This GPU implementation should not be invoked with mpirun/mpiexec on more than one process.");
    }
    int ncrystals = ncrystalsGlobal;
    
    // Crystal list
    std::vector<hpp::SpectralCrystalCUDA<U>> crystalList(ncrystals);
    
    // Add crystals
    unsigned int reportInterval = 10000000;
    hpp::Tensor2<U> rotation(3,3);
    for (unsigned int i=0; i<crystalList.size(); i++) {
        // Report
        if (i%reportInterval == 0 && i/reportInterval > 0) {
            std::cout << "Generating crystal " << i << std::endl;
        }
        
        // Initial slip-system deformation resistance
        crystalList[i].s = init.s_0;
        
        // Initial rotation        
        if (ncrystalsGlobal == 1) {
            rotation = hpp::identityTensor2<U>(3);
        }
        else {
            hpp::randomRotationTensorInPlace<U>(3, rotation, defaultSeed);
        }
        
        // Get angles
        crystalList[i].angles = hpp::getEulerZXZAngles<U>(rotation);
    }
    
    // Convert properties and database
    hpp::CrystalPropertiesCUDA<U,12> crystalPropsCUDA(props);
    
    // Create polycrystal
    hpp::SpectralPolycrystalCUDA<U,12> polycrystal;
    if (unifiedCoeffOrder) {
        std::vector<hpp::SpectralDatasetID> dsetIDs = hpp::defaultCrystalSpectralDatasetIDs();
        hpp::SpectralDatabaseUnified<U> db(databaseFilename, dsetIDs, nTerms, comm, refinementMultiplier);
        polycrystal = hpp::SpectralPolycrystalCUDA<U,12>(crystalList, crystalPropsCUDA, db);
    }
    else {
        std::vector<std::string> dsetBasenames = {"sigma_prime", "W_p", "gammadot_abs_sum"};
        hpp::SpectralDatabase<U> db(databaseFilename, dsetBasenames, nTerms, comm, refinementMultiplier);
        polycrystal = hpp::SpectralPolycrystalCUDA<U,12>(crystalList, crystalPropsCUDA, db);     
    }        
    
    // Evolve
    polycrystal.evolve(experiment.tStart, experiment.tEnd, dt, experiment.F_of_t, experiment.L_of_t);

    // Write the result
    polycrystal.writeResultHDF5(outputFilename);
    
    // Write out strain history
    if (comm_rank == 0) {
        // Strain history
        std::vector<U> trueStrainHistory = hpp::operator*(experiment.strainRate, polycrystal.getTHistory());        
        H5::H5File outfile(outputFilename, H5F_ACC_RDWR);
        hpp::writeVectorToHDF5Array(outfile, "trueStrainHistory", trueStrainHistory);  
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
    std::string experimentName;
    unsigned int nCrystals = 0;
    unsigned int nTerms = 0;
    std::string resultsFilename;
    unsigned int refinementMultiplier;
    bool unifiedCoeffOrder = false;
    bool defaultSeed = false;
    
    // Get options
    try {
        // The parser
        TCLAP::CmdLine parser("Simulate the conditions in the kalidindi1992 problem using a spectral database of responses.", ' ', "0.3");
        
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
        
        // Experiment name option
        std::string expNameArgChar = "e";
        std::string defaultExpname = "kalidindi1992_simple_shear";
        bool expNameRequired = true;
        std::string expNameDescription = "Experiment name";
        TCLAP::ValueArg<std::string> expNameArg(expNameArgChar,
        "experimentname", expNameDescription, expNameRequired, defaultExpname, "string", parser);
        
        // Whether or not to use a unified coefficient ordering
        std::string unifiedCoeffOrderArgChar = "u";
        std::string unifiedCoeffOrderDescription = "Use a single ordering for all Fourier coefficients";
        TCLAP::SwitchArg unifiedCoeffOrderArg(unifiedCoeffOrderArgChar, "unifiedcoefforder", unifiedCoeffOrderDescription, parser);
        
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
        unifiedCoeffOrder = unifiedCoeffOrderArg.getValue();
        defaultSeed = defaultSeedArg.getValue();
    } 
    catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
    }
    
    // Run
    mihaila2014::spectralSolve<float>(experimentName, databaseFilename, unifiedCoeffOrder, refinementMultiplier, nCrystals, nTerms, resultsFilename, comm, defaultSeed);
    
    // MPI finalize
    MPI_Finalize();
    
    // Return
    return 0;
}