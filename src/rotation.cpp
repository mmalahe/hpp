#include <hpp/rotation.h>

namespace hpp {   

/**
 * @brief Creates a uniform grid on SO3
 * @detail The resolution parameter \f$ r \f$ specifies a grid with a total of
 * \f$ 72 \times 8^{r} \f$ points. 
 * @param resolution 
 */
template <typename T>
SO3Discrete<T>::SO3Discrete(unsigned int resolution) {
    if (resolution >= 7) {
        unsigned long int nPoints = 72*std::pow(8, resolution);
        std::cerr << "WARNING: you are about to contruct " << nPoints << " points." << std::endl; 
    }
    
    // Create lists in various representations
    quatList = isoi::simple_grid_quaternion(resolution);
    eulerAngleList.resize(quatList.size());
    for (unsigned int i=0; i<quatList.size(); i++) {
        eulerAngleList[i] = quaternionToEulerAngles<T>(quatList[i]);
    }
}  

// SO3Discrete is restricted to these specific instantiations
template struct SO3Discrete<float>;
template struct SO3Discrete<double>;

}//END NAMESPACE HPP