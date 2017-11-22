/// @file python.cpp
/// @author Michael Malahe
/// @brief The Python interface to the library.

#include <boost/python.hpp>
#include <hpp/tensor.h>
#include <hpp/casesUtils.h>
#include <hpp/spectralUtils.h>
#include <hpp/crystal.h>
#include <hpp/crystalCUDA.h>
#include <hpp/continuum.h>
#include <hpp/python.h>

float listDemo(const boost::python::list& list) {
    auto vec = hpp::toStdVector<hpp::SpectralCrystalCUDA<float>>(list);
    for (const auto& v : vec) {
        std::cout << v.s << std::endl;
        std::cout << v.angles << std::endl;
    }
    return vec[0].s;
}

// Wrappers/conversions
boost::python::list defaultCrystalSpectralDatasetIDsPy() {
    std::vector<hpp::SpectralDatasetID> dsetIDs = hpp::defaultCrystalSpectralDatasetIDs();
    boost::python::list dsetIDsPy = hpp::toPythonList<hpp::SpectralDatasetID>(dsetIDs);
    return dsetIDsPy;
}

// The module
BOOST_PYTHON_MODULE(hpppy) {    
    // debugging/testing/experimenting
    boost::python::def("listDemo", listDemo, boost::python::args("list"), "list demo");
    
    // tensor.h
    boost::python::class_<hpp::Tensor2<float>>("Tensor2F", 
        boost::python::init<const unsigned int, const unsigned int>())
        .def("getn1", &hpp::Tensor2<float>::getn1)
    ;
    boost::python::class_<hpp::EulerAngles<float>>("EulerAnglesF");
    
    // casesUtils.h
    boost::python::class_<hpp::Experiment<float>>("ExperimentF", 
        boost::python::init<std::string>())
        .add_property("strainRate", &hpp::Experiment<float>::getStrainRate)
        .add_property("tStart", &hpp::Experiment<float>::getTStart)
        .add_property("tEnd", &hpp::Experiment<float>::getTEnd)
        .def("generateNextOrientationAngles", &hpp::Experiment<float>::generateNextOrientationAngles)
    ;
    
    // spectralUtils.h
    boost::python::class_<hpp::SpectralDatasetID>("SpectralDatasetID");
    
    // crystal.h
    boost::python::class_<hpp::CrystalProperties<float>>("CrystalPropertiesF");
    boost::python::class_<hpp::CrystalInitialConditions<float>>("CrystalInitialConditionsF")
        .add_property("s_0", &hpp::CrystalInitialConditions<float>::getS0, &hpp::CrystalInitialConditions<float>::setS0)
    ;
    boost::python::def("defaultCrystalPropertiesF", hpp::defaultCrystalProperties<float>);
    boost::python::def("defaultCrystalInitialConditionsF", hpp::defaultCrystalInitialConditions<float>);
    boost::python::def("defaultCrystalSpectralDatasetIDs", defaultCrystalSpectralDatasetIDsPy);
    
    // crystalCUDA.h
    boost::python::class_<hpp::CrystalPropertiesCUDA<float,12>>("CrystalPropertiesCUDAF12", 
        boost::python::init<const hpp::CrystalProperties<float>&>())
    ;
    boost::python::class_<hpp::SpectralCrystalCUDA<float>>("SpectralCrystalCUDAF")
        .add_property("s", &hpp::SpectralCrystalCUDA<float>::getS, &hpp::SpectralCrystalCUDA<float>::setS)
        .add_property("angles", &hpp::SpectralCrystalCUDA<float>::getAngles, &hpp::SpectralCrystalCUDA<float>::setAngles)
    ;
}