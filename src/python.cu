/// @file python.cu
/// @author Michael Malahe
/// @brief The Python interface to the CUDA-enabled library.

#include <hpp/config.h>
HPP_CHECK_CUDA_ENABLED_BUILD
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
    #include "python_common.cpp"
    
    // gshCUDA.h //
    ///////////////
    boost::python::class_<hpp::GSHCoeffsCUDA<float>>("GSHCoeffsCUDAF", 
        boost::python::init<>())
        .def("getl0Reals", &hpp::GSHCoeffsCUDA<float>::getl0Reals)
        .def("getl1Reals", &hpp::GSHCoeffsCUDA<float>::getl1Reals)
        .def("getl2Reals", &hpp::GSHCoeffsCUDA<float>::getl2Reals)
        .def("getl3Reals", &hpp::GSHCoeffsCUDA<float>::getl3Reals)
        .def("getl4Reals", &hpp::GSHCoeffsCUDA<float>::getl4Reals)
    ;
    
    // crystalCUDA.h //
    ///////////////////
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

    // SpectralPolycrystalCUDA //
    /////////////////////////////
    
    // Selection of particular overloaded methods
    void (hpp::SpectralPolycrystalCUDA<float,12>::*step)(const hpp::Tensor2<float>&, float) = &hpp::SpectralPolycrystalCUDA<float,12>::step;
    hpp::Tensor2<float> (hpp::SpectralPolycrystalCUDA<float,12>::*getPoleHistogram)(int, int, int) = &hpp::SpectralPolycrystalCUDA<float,12>::getPoleHistogram;
    
    // Interface
    boost::python::class_<hpp::SpectralPolycrystalCUDA<float,12>>("SpectralPolycrystalCUDAF12", 
        boost::python::init<std::vector<hpp::SpectralCrystalCUDA<float>>&, const hpp::CrystalPropertiesCUDA<float, 12>&, const hpp::SpectralDatabaseUnified<float>&>())
        .def("evolve", &hpp::SpectralPolycrystalCUDA<float,12>::evolve)
        .def("resetRandomOrientations", &hpp::SpectralPolycrystalCUDA<float,12>::resetRandomOrientations)
        .def("resetGivenOrientations", &hpp::SpectralPolycrystalCUDA<float,12>::resetGivenOrientations)
        .def("getEulerAnglesZXZActive", &hpp::SpectralPolycrystalCUDA<float,12>::getEulerAnglesZXZActive)
        .def("step", step)
        .def("getGSHCoeffs", &hpp::SpectralPolycrystalCUDA<float,12>::getGSHCoeffs)
        .def("getPoleHistogram", getPoleHistogram)
    ;
}