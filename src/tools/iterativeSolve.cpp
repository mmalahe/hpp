/** @file testKalidindi1992.cpp
* @author Michael Malahe
* @brief Tests for replicating the results in Kalidindi1992
*/

#include <hpp/tensor.h>
#include <hpp/crystal.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <hdf5/serial/H5Cpp.h>
#include <tclap/CmdLine.h>
#include <hpp/casesUtils.h>

namespace kalidindi1992
{

template <typename U>
void replicate(std::string output_filename, unsigned int ncrystalsGlobal, std::string experimentName, bool defaultSeed=false)
{    
    // Crystal properties
    hpp::CrystalProperties<U> props = hpp::defaultCrystalProperties<U>();
    
    // Solver configuration
    hpp::CrystalSolverConfig<U> config = hpp::defaultConservativeCrystalSolverConfig<U>();
    
    // Initial conditions
    hpp::CrystalInitialConditions<U> init = hpp::defaultCrystalInitialConditions<U>();
    U dt_initial = 1e-3;
    
    // Output
    hpp::PolycrystalOutputConfig outputConfig;
    outputConfig.verbose = true;
    outputConfig.writeTextureHistory = true;
    outputConfig.textureHistoryTimeInterval = 20.0;
    
    // Experiment parameters
    hpp::Experiment<U> experiment(experimentName);
    
    // Divide crystals between processes
    MPI_Comm comm =  MPI_COMM_WORLD;
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Generate all Euler angles on root
    std::vector<hpp::EulerAngles<U>> anglesRoot;
    if (comm_rank == 0) {
        anglesRoot.resize(ncrystalsGlobal);
        for (unsigned int i=0; i<ncrystalsGlobal; i++) {
            experiment.orientationGenerator->generateNext(anglesRoot[i]);
        }
    }
    
    // Distribute Euler Angles to processes
    auto EulerAnglesTypeMPI = hpp::getEulerAnglesTypeMPI<U>();
    std::vector<hpp::EulerAngles<U>> anglesLocal = hpp::MPISplitVectorEvenly(anglesRoot, comm, EulerAnglesTypeMPI);
    unsigned int ncrystalsLocal = anglesLocal.size();
    unsigned int ncrystalsSum = hpp::MPISum(ncrystalsLocal, comm);
    if (ncrystalsGlobal != ncrystalsSum) {
        throw std::runtime_error("Mismatch between distributed angles and number of crystals.");
    }
    
    // Generate crystals locally
    std::vector<hpp::Crystal<U>> crystal_list(ncrystalsLocal);
    hpp::Tensor2<U> rotTensor(3,3);
    for (unsigned int i=0; i<crystal_list.size(); i++) {
        // Generate rotation
        rotTensor = hpp::toRotationMatrix(anglesLocal[i]);
        
        // Add initial rotation
        hpp::CrystalInitialConditions<U> initRotated = init;
        initRotated.crystalRotation = rotTensor;
        
        // Add to crystal list
        crystal_list[i] = hpp::Crystal<U>(props, config, initRotated);
    }
    
    // Generate polycrystal and solve
    hpp::Polycrystal<U> kalidindi_polycrystal(crystal_list, comm, outputConfig);
    kalidindi_polycrystal.evolve(experiment.tStart, experiment.tEnd, dt_initial, experiment.F_of_t);
    
    // Write out results
    kalidindi_polycrystal.writeResultHDF5(output_filename);
    
    // Write out strain history
    if (comm_rank == 0) {
        std::vector<U> trueStrainHistory = hpp::operator*(experiment.strainRate, kalidindi_polycrystal.getTHistory());        
        H5::H5File outfile(output_filename, H5F_ACC_RDWR);
        hpp::writeVectorToHDF5Array(outfile, "trueStrainHistory", trueStrainHistory);
    }
}

} // END NAMESPACE kalidindi1992

int main(int argc, char *argv[]) 
{
    // MPI init
    MPI_Init(&argc, &argv);
    
    // Options
    std::string resultsFilename;
    std::string experimentName;
    unsigned int nCrystals = 0;
    bool defaultSeed = false;
    
    // Get options
    try {
        // The parser
        TCLAP::CmdLine parser("Replicate the results in kalidindi1992.", ' ', "0.3");
        
        // Filename option
        std::string filenameArgChar = "o";
        std::string defaultFilename = "results.txt";
        bool filenameRequired = true;
        std::string filenameDescription = "Output filename";
        TCLAP::ValueArg<std::string> filenameArg(filenameArgChar,
        "filename", filenameDescription, filenameRequired, defaultFilename, "string", parser);
        
        // Experiment name option
        std::string expNameArgChar = "e";
        std::string defaultExpname = "kalidindi1992_simple_shear";
        bool expNameRequired = true;
        std::string expNameDescription = "Experiment name";
        TCLAP::ValueArg<std::string> expNameArg(expNameArgChar,
        "experimentname", expNameDescription, expNameRequired, defaultExpname, "string", parser);
        
        // Number of crystals
        std::string nCrystalsArgChar = "n";
        unsigned int defaultNCrystals = 0;
        bool nCrystalsRequired = true;
        std::string nCrystalsDescription = "Number of crystals in the polycrystal";
        TCLAP::ValueArg<unsigned int> nCrystalsArg(nCrystalsArgChar,
        "ncrystals", nCrystalsDescription, nCrystalsRequired, defaultNCrystals, "integer", parser);
        
        // Whether or not to use consistent random seed between runs
        std::string defaultSeedArgChar = "d";
        std::string defaultSeedDescription = "Use a consistent random seed between runs";
        TCLAP::SwitchArg defaultSeedArg(defaultSeedArgChar, "defaultseed", defaultSeedDescription, parser);
        
        // Parse the argv array
        parser.parse(argc, argv);

        // Get the value parsed by each arg
        experimentName = expNameArg.getValue();
        resultsFilename = filenameArg.getValue();
        nCrystals = nCrystalsArg.getValue();
        defaultSeed = defaultSeedArg.getValue();
    } 
    catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
    }
    
    // Run
    kalidindi1992::replicate<double>(resultsFilename, nCrystals, experimentName, defaultSeed);
    
    // MPI finalize
    MPI_Finalize();
    
    // Return
    return 0;
}