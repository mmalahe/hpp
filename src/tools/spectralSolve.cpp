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

namespace mihaila2014
{

template <typename U>
void replicateKalidindi1992Results(std::string databaseFilename, unsigned int refinementMultiplier, unsigned int ncrystalsGlobal, unsigned int nTerms, std::string outputFilename, MPI_Comm comm, unsigned int nOmpThreads, bool defaultSeed=false)
{    
    // Problem parameters
    U tStart = 0.0;
    U tEnd = 882.0;
    U shearRate = -1.7e-3;
    
    // Prescribed deformation
    std::function<hpp::Tensor2<U>(U)> L_of_t = 
    std::bind(hpp::simpleShearVelocityGradient<U>, std::placeholders::_1, shearRate);
    std::function<hpp::Tensor2<U>(U)> F_of_t = 
    std::bind(hpp::simpleShearDeformationGradient<U>, std::placeholders::_1, shearRate);
    
    // Method parameters
    U strainIncrement = -2e-2;
    
    // Derived method parameters
    U dt = std::abs(strainIncrement/shearRate);
    
    // Input dataset
    std::vector<std::string> dsetBasenames = {"sigma_prime", "W_p", "gammadot_abs_sum"};
    hpp::SpectralDatabase<U> db(databaseFilename, dsetBasenames, nTerms, comm, refinementMultiplier);
    
    // Crystal properties and initial conditions
    hpp::CrystalProperties<U> props = hpp::defaultCrystalProperties<U>();
    hpp::CrystalInitialConditions<U> init = hpp::defaultCrystalInitialConditions<U>();
    
    // Divide crystals between processes
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    if (comm_size != 1) {
        throw std::runtime_error("This shared memory implementation should not be invoked with mpirun/mpiexec on more than one process.");
    }
    int ncrystals = ncrystalsGlobal;    
    
    // Spectral crystal solver configuration
    hpp::SpectralCrystalSolverConfig<U> config;
    config.nTerms = nTerms;
    
    // Crystal list
    std::vector<hpp::SpectralCrystal<U>> crystalList(ncrystals);
    
    // For a single crystal only, do not randomize the rotation
    //std::cout<<"WARNING: not rotating crystals for profiling/debugging purposes."<<std::endl;
    for (auto&& crystal : crystalList) {
        if (ncrystalsGlobal == 1) {
            // Do not rotate if there's only a single crystal
            // This is a special case for debugging purposes
            crystal = hpp::SpectralCrystal<U>(props, config, init);
        }
        else {
            hpp::CrystalInitialConditions<U> initRotated = init;
            initRotated.crystalRotation = hpp::randomRotationTensor<U>(3, defaultSeed);
            crystal = hpp::SpectralCrystal<U>(props, config, initRotated);
        }
    }
    
    // Polycrystal
    hpp::SpectralPolycrystal<U> polycrystal(crystalList, nOmpThreads);
    
    // Evolve the polycrystal
    polycrystal.evolve(tStart, tEnd, dt, F_of_t, L_of_t, db);
    
    // Write the result
    polycrystal.writeResultNumpy(outputFilename);
}
    
} // END NAMESPACE mihaila2014

int main(int argc, char *argv[])
{
    // MPI init
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    
    // Options
    std::string databaseFilename;
    unsigned int nCrystals = 0;
    unsigned int nTerms = 0;
    std::string resultsFilename;
    unsigned int nOmpThreads;
    unsigned int refinementMultiplier;
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
        
        // Number of openmp threads option
        argChar = "m";
        defaultVal = 1;
        required = false;
        description = "Number of OpenMP threads";
        TCLAP::ValueArg<unsigned int> nThreadsArg(argChar,
        "nthreads", description, required, defaultVal, "integer", parser);
        
        // Whether or not to use consistent random seed between runs
        std::string defaultSeedArgChar = "d";
        std::string defaultSeedDescription = "Use a consistent random seed between runs";
        TCLAP::SwitchArg defaultSeedArg(defaultSeedArgChar, "defaultseed", defaultSeedDescription, parser);

        // Parse the argv array
        parser.parse(argc, argv);

        // Get the value parsed by each arg
        databaseFilename = databaseFilenameArg.getValue();
        nCrystals = nCrystalsArg.getValue();
        nTerms = nTermsArg.getValue();
        resultsFilename = filenameArg.getValue();
        nOmpThreads = nThreadsArg.getValue();
        refinementMultiplier = refinementMultiplierArg.getValue();
        defaultSeed = defaultSeedArg.getValue();
    } 
    catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
    }
    
    // Run
    mihaila2014::replicateKalidindi1992Results<float>(databaseFilename, refinementMultiplier, nCrystals, nTerms, resultsFilename, comm, nOmpThreads, defaultSeed);
    
    // MPI finalize
    MPI_Finalize();
    
    // Return
    return 0;
}