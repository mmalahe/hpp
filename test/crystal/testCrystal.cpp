/** @file testCrystal.cpp
* @author Michael Malahe
* @brief Tests for functions in crystal.h.
*/

#include <hpp/tensor.h>
#include <hpp/crystal.h>
#include <cassert>
#include <functional>

namespace hpp{

template <typename U>
void testCrystal() 
{
    // Settings for both single and polycrystal tests
    CrystalProperties<U> props = defaultCrystalProperties<U>();
    CrystalSolverConfig<U> config = defaultCrystalSolverConfig<U>();
    CrystalInitialConditions<U> init = defaultCrystalInitialConditions<U>();
    hpp::Tensor2<U> F_next = hpp::identityTensor2<U>(3);
    U dt = 1e-3;
    U dt_new;
    
    // Single crystal
    Crystal<U> crystal(props, config, init);
    bool step_good = crystal.tryStep(F_next, dt);
    std::cout << "step_good: " << step_good << std::endl;
//    crystal.acceptStep();
//    crystal.rejectStep();
//    dt_new = crystal.recommendNextTimestepSize(dt);
//    std::cout << "dt_new: " << dt_new << std::endl;
//    std::vector<std::vector<U>> m_alphas = crystal.getM_alphas();
//    std::vector<std::vector<U>> n_alphas = crystal.getN_alphas();
    
    // Polycrystal composed of a single crystal
    std::vector<Crystal<U>> crystal_list(1);
    crystal_list[0] = Crystal<U>(props, config, init);
    Polycrystal<U> polycrystal(crystal_list, MPI_COMM_WORLD);
    step_good = polycrystal.step(F_next, dt);
    std::cout << "step_good: " << step_good << std::endl;
//    dt_new = crystal.recommendNextTimestepSize(dt);
//    std::cout << "dt_new: " << dt_new << std::endl;
}

} //end namespace hpp

int main(int argc, char *argv[]) 
{
    // MPI init
    MPI_Init(&argc, &argv);
    
    // Test
    hpp::testCrystal<float>();
    hpp::testCrystal<double>();  
    
    // MPI finalize
    MPI_Finalize();
    
    // Return
    return 0;
}