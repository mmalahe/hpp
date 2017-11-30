/// @file python.cpp
/// @author Michael Malahe
/// @brief The Python interface to the library.

#include <boost/python.hpp>
#include <hpp/tensor.h>
#include <hpp/casesUtils.h>
#include <hpp/spectralUtils.h>
#include <hpp/crystal.h>
#include <hpp/gshCUDA.h>
#include <hpp/crystalCUDA.h>
#include <hpp/continuum.h>
#include <hpp/python.h>
#include <hpp/cudaUtils.h>

// The module
BOOST_PYTHON_MODULE(hpppy) {    
    // General
    boost::python::class_<std::vector<float> >("FVec")
        .def(boost::python::vector_indexing_suite<std::vector<float>>())
    ;
    boost::python::class_<std::function<hpp::Tensor2<float>(float)>>("Tensor2FunctionOfScalarF");
    
    // tensor.h
    boost::python::class_<hpp::Tensor2<float>>("Tensor2F", 
        boost::python::init<const unsigned int, const unsigned int>())
        .def("getn1", &hpp::Tensor2<float>::getn1)
        .def("getn2", &hpp::Tensor2<float>::getn2)
        .def("setVal", &hpp::Tensor2<float>::setVal)
    ;
    boost::python::class_<hpp::EulerAngles<float>>("EulerAnglesF")
        .add_property("alpha", &hpp::EulerAngles<float>::getAlpha)
        .add_property("beta", &hpp::EulerAngles<float>::getBeta)
        .add_property("gamma", &hpp::EulerAngles<float>::getGamma)
    ;
    boost::python::class_<std::vector<hpp::EulerAngles<float>>>("EulerAnglesFVec")
        .def(boost::python::vector_indexing_suite<std::vector<hpp::EulerAngles<float>>>())
    ;
    
    // casesUtils.h
    boost::python::class_<hpp::Experiment<float>>("ExperimentF", 
        boost::python::init<std::string>())
        .add_property("strainRate", &hpp::Experiment<float>::getStrainRate)
        .add_property("tStart", &hpp::Experiment<float>::getTStart)
        .add_property("tEnd", &hpp::Experiment<float>::getTEnd)
        .add_property("F_of_t", &hpp::Experiment<float>::getF_of_t)
        .add_property("L_of_t", &hpp::Experiment<float>::getL_of_t)
        .def("generateNextOrientationAngles", &hpp::Experiment<float>::generateNextOrientationAngles)
    ;
    
    // spectralUtils.h
    boost::python::class_<hpp::SpectralDatasetID>("SpectralDatasetID");
    boost::python::class_<std::vector<hpp::SpectralDatasetID> >("SpectralDatasetIDVec")
        .def(boost::python::vector_indexing_suite<std::vector<hpp::SpectralDatasetID>>())
    ;
    boost::python::class_<hpp::SpectralDatabaseUnified<float>>("SpectralDatabaseUnifiedF",
        boost::python::init<std::string, std::vector<hpp::SpectralDatasetID>, unsigned int, unsigned int>())
    ;
    
    // crystal.h
    boost::python::class_<hpp::CrystalProperties<float>>("CrystalPropertiesF");
    boost::python::class_<hpp::CrystalInitialConditions<float>>("CrystalInitialConditionsF")
        .add_property("s_0", &hpp::CrystalInitialConditions<float>::getS0, &hpp::CrystalInitialConditions<float>::setS0)
    ;
    boost::python::def("defaultCrystalPropertiesF", hpp::defaultCrystalProperties<float>);
    boost::python::def("defaultCrystalInitialConditionsF", hpp::defaultCrystalInitialConditions<float>);
    boost::python::def("defaultCrystalSpectralDatasetIDs", hpp::defaultCrystalSpectralDatasetIDs);
    
    // gshCUDA.h
    boost::python::class_<hpp::GSHCoeffsCUDA<float>>("GSHCoeffsCUDAF", 
        boost::python::init<>())
        .def("getl0Reals", &hpp::GSHCoeffsCUDA<float>::getl0Reals)
        .def("getl1Reals", &hpp::GSHCoeffsCUDA<float>::getl1Reals)
        .def("getl2Reals", &hpp::GSHCoeffsCUDA<float>::getl2Reals)
    ;    
    
    // crystalCUDA.h
    boost::python::class_<hpp::CrystalPropertiesCUDA<float,12>>("CrystalPropertiesCUDAF12", 
        boost::python::init<const hpp::CrystalProperties<float>&>())
    ;
    boost::python::class_<hpp::SpectralCrystalCUDA<float>>("SpectralCrystalCUDAF")
        .add_property("s", &hpp::SpectralCrystalCUDA<float>::getS, &hpp::SpectralCrystalCUDA<float>::setS)
        .add_property("angles", &hpp::SpectralCrystalCUDA<float>::getAngles, &hpp::SpectralCrystalCUDA<float>::setAngles)
    ;
    boost::python::class_<std::vector<hpp::SpectralCrystalCUDA<float>> >("SpectralCrystalCUDAFVec")
        .def(boost::python::vector_indexing_suite<std::vector<hpp::SpectralCrystalCUDA<float>>>())
    ;
    
    void (hpp::SpectralPolycrystalCUDA<float,12>::*step)(const hpp::Tensor2<float>&, float) = &hpp::SpectralPolycrystalCUDA<float,12>::step;
    boost::python::class_<hpp::SpectralPolycrystalCUDA<float,12>>("SpectralPolycrystalCUDAF12", 
        boost::python::init<std::vector<hpp::SpectralCrystalCUDA<float>>&, const hpp::CrystalPropertiesCUDA<float, 12>&, const hpp::SpectralDatabaseUnified<float>&>())
        .def("evolve", &hpp::SpectralPolycrystalCUDA<float,12>::evolve)
        .def("reset", &hpp::SpectralPolycrystalCUDA<float,12>::reset)
        .def("getEulerAnglesZXZActive", &hpp::SpectralPolycrystalCUDA<float,12>::getEulerAnglesZXZActive)
        .def("step", step)
        .def("getGSHCoeffs", &hpp::SpectralPolycrystalCUDA<float,12>::getGSHCoeffs)
    ;
}