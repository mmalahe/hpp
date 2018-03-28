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
void testUtil() 
{
    hpp::SO3Discrete<U> so3(2);
    for (unsigned int i=0; i<5; i++) {
        std::cout << so3.getEulerAngle(i) << std::endl;
    }
}
    
template <typename U>
void testCrystal() 
{
    // Settings for both single and polycrystal tests
    CrystalProperties<U> props = defaultCrystalProperties<U>();
    CrystalSolverConfig<U> config = defaultCrystalSolverConfig<U>();
    CrystalInitialConditions<U> init = defaultCrystalInitialConditions<U>();
    hpp::Tensor2<U> F_next = hpp::identityTensor2<U>(3);
    U dt = 1e-3;
    
    // Single crystal
    Crystal<U> crystal(props, config, init);
    bool step_good = crystal.tryStep(F_next, dt);
    std::cout << "step_good: " << step_good << std::endl;
    
    // Polycrystal composed of a single crystal
    std::vector<Crystal<U>> crystal_list(1);
    crystal_list[0] = Crystal<U>(props, config, init);
    Polycrystal<U> polycrystal(crystal_list, MPI_COMM_WORLD);
    step_good = polycrystal.step(F_next, dt);
    std::cout << "step_good: " << step_good << std::endl;
}

} //end namespace hpp

int main(int argc, char *argv[]) 
{
    // MPI init
    MPI_Init(&argc, &argv);
    
    // Test
    hpp::testUtil<float>();
    hpp::testUtil<double>();  
    hpp::testCrystal<float>();
    hpp::testCrystal<double>();  
    
    // MPI finalize
    MPI_Finalize();
    
    // Return
    return 0;
}