#include <hpp/casesUtils.h>


namespace hpp 
{

template <typename U>
Experiment<U>::Experiment(std::string experimentName) {
    // Common to most
    tStart = 0.0;
    orientationGenerator = std::make_shared<RandomOrientationGenerator<U>>();
    
    if (experimentName == "kalidindi1992_simple_shear") {
        tEnd = 823.53;
        strainRate = -1.7e-3;
        F_of_t = std::bind(simpleShearDeformationGradient<U>, std::placeholders::_1, strainRate);
        L_of_t = std::bind(simpleShearVelocityGradient<U>, std::placeholders::_1, strainRate);
        
    }
    else if (experimentName == "kalidindi1992_simple_compression") {
        tEnd = 1500.00;
        strainRate = -1.0e-3;
        F_of_t = std::bind(simpleCompressionDeformationGradient<U>, std::placeholders::_1, strainRate);
        L_of_t = std::bind(simpleCompressionVelocityGradient<U>, std::placeholders::_1, strainRate);
    }
    else if (experimentName == "savage2015_plane_strain_compression") {
        tEnd = 1000.0;
        strainRate = 1.0e-3;
        F_of_t = std::bind(planeStrainCompressionDeformationGradient<U>, std::placeholders::_1, strainRate);
        L_of_t = std::bind(planeStrainCompressionVelocityGradient<U>, std::placeholders::_1, strainRate);
    }
    else if (experimentName == "mihaila2014_simple_shear") {
        tEnd = 1000.0;
        strainRate = 1.0e-3;
        F_of_t = std::bind(simpleShearDeformationGradient<U>, std::placeholders::_1, strainRate);
        L_of_t = std::bind(simpleShearVelocityGradient<U>, std::placeholders::_1, strainRate);
    }
    else if (experimentName == "static") {
        tEnd = 1.0e-2;
        strainRate = 0.0;
        F_of_t = hpp::staticDeformationGradient<U>;
        L_of_t = hpp::staticVelocityGradient<U>;
    }
    else {
        throw std::runtime_error("No implementation for experiment with name "+experimentName);
    }
}

// Explicit instantiations
template class Experiment<float>;
template class Experiment<double>;

} // END NAMESPACE hpp