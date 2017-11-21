/// @file python.cpp
/// @author Michael Malahe
/// @brief The Python interface to the library.

#include <boost/python.hpp>
#include <hpp/tensor.h>
#include <hpp/casesUtils.h>
#include <hpp/crystal.h>
#include <hpp/crystalCUDA.h>
#include <hpp/python.h>

std::vector<double> makeADoubleVec() {
    std::vector<double> vec;
    vec.push_back(0.2);
    vec.push_back(0.4);
    return vec;
}

boost::python::list makeADoubleList() {
    auto vec = makeADoubleVec();
    return hpp::toPythonList(vec);
}

BOOST_PYTHON_MODULE(hpppy) {    
    // debugging/testing/experimenting
    boost::python::def("makeADoubleList", makeADoubleList);
    
    // tensor.h
    boost::python::class_<hpp::Tensor2<float>>("Tensor2F", 
        boost::python::init<const unsigned int, const unsigned int>())
        .def("getn1", &hpp::Tensor2<float>::getn1)
    ;
    
    // casesUtils.h
    boost::python::class_<hpp::Experiment<float>>("ExperimentF", 
        boost::python::init<std::string>())
        .add_property("strainRate", &hpp::Experiment<float>::getStrainRate)
        .add_property("tStart", &hpp::Experiment<float>::getTStart)
        .add_property("tEnd", &hpp::Experiment<float>::getTEnd)      
    ;
    
    // crystal.h
    boost::python::class_<hpp::CrystalProperties<float>>("CrystalPropertiesF");
    boost::python::class_<hpp::CrystalInitialConditions<float>>("CrystalInitialConditionsF");
    boost::python::def("defaultCrystalPropertiesF", hpp::defaultCrystalProperties<float>);
    boost::python::def("defaultCrystalInitialConditionsF", hpp::defaultCrystalInitialConditions<float>);
    
    // crystalCUDA.h
    boost::python::class_<hpp::SpectralCrystalCUDA<float>>("SpectralCrystalCUDAF");    
}