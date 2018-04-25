#include <hpp/rotation.h>

namespace hpp {   

/**
 * @brief Creates a uniform grid on SO3
 * @detail The resolution parameter \f$ r \f$ specifies a grid with a total of
 * \f$ 72 \times 8^{r} \f$ points. 
 * @param resolution
 * @todo Add and test accounting for C4 symmetry
 */
template <typename T>
SO3Discrete<T>::SO3Discrete(unsigned int resolution, SymmetryType symmetryType) {  
    if (resolution >= 7) {
        unsigned long int nPoints = 72*std::pow(8, resolution);
        std::cerr << "WARNING: you are about to contruct " << nPoints << " points." << std::endl; 
    }
    
    if (symmetryType == SYMMETRY_TYPE_C4) {
        std::cerr << "WARNING: C4 symmetry implementation is not complete." << std::endl;
        std::cerr << "Proceeding without accounting for symmetries." << std::endl;
        symmetryType = SYMMETRY_TYPE_NONE;
    }
    
    // Create base rotation list with quaternions
    switch (symmetryType) {
        case SYMMETRY_TYPE_NONE:
            quatList = isoi::full_grid_quaternion(resolution);
            break;
        case SYMMETRY_TYPE_C4:
            quatList = isoi::fourfold_symmetry_grid_quaternion(resolution);
            break;
        default:
            std::cerr << "For rotation space generation, no implementation for symmetry type = " << symmetryType << std::endl;
            throw std::runtime_error("No implementation.");
    }
    
    // Convert to other representations        
    eulerAngleList.resize(quatList.size());
    for (unsigned int i=0; i<quatList.size(); i++) {
        eulerAngleList[i] = toEulerAngles<T>(quatList[i]);
    }
}  

// SO3Discrete is restricted to these specific instantiations
template struct SO3Discrete<float>;
template struct SO3Discrete<double>;

}//END NAMESPACE HPP