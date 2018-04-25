/**
 * @brief Common python functionality for both the CPU-only and CUDA-enabled builds.
 * @detail This file should not (and cannot) be compiled by itself. It is included
 * in python.cpp and python.cu.
 */

// General //
/////////////
boost::python::class_<std::vector<float> >("FVec")
    .def(boost::python::vector_indexing_suite<std::vector<float>>())
;
boost::python::class_<std::vector<double> >("DVec")
    .def(boost::python::vector_indexing_suite<std::vector<double>>())
;
boost::python::class_<std::function<hpp::Tensor2<float>(float)>>("Tensor2FunctionOfScalarF");
boost::python::class_<std::function<hpp::Tensor2<double>(double)>>("Tensor2FunctionOfScalarD");

// tensor.h //
//////////////
boost::python::class_<hpp::Tensor2<float>>("Tensor2F", 
    boost::python::init<const unsigned int, const unsigned int>())
    .def("getn1", &hpp::Tensor2<float>::getn1)
    .def("getn2", &hpp::Tensor2<float>::getn2)
    .def("setVal", &hpp::Tensor2<float>::setVal)
    .def("getVal", &hpp::Tensor2<float>::getVal)
;
boost::python::class_<hpp::Tensor2<double>>("Tensor2D", 
    boost::python::init<const unsigned int, const unsigned int>())
    .def("getn1", &hpp::Tensor2<double>::getn1)
    .def("getn2", &hpp::Tensor2<double>::getn2)
    .def("setVal", &hpp::Tensor2<double>::setVal)
    .def("getVal", &hpp::Tensor2<double>::getVal)
;
boost::python::class_<hpp::EulerAngles<float>>("EulerAnglesF")
    .add_property("alpha", &hpp::EulerAngles<float>::getAlpha, &hpp::EulerAngles<float>::setAlpha)
    .add_property("beta", &hpp::EulerAngles<float>::getBeta, &hpp::EulerAngles<float>::setBeta)
    .add_property("gamma", &hpp::EulerAngles<float>::getGamma, &hpp::EulerAngles<float>::setGamma)
;
boost::python::class_<hpp::EulerAngles<double>>("EulerAnglesD")
    .add_property("alpha", &hpp::EulerAngles<double>::getAlpha, &hpp::EulerAngles<double>::setAlpha)
    .add_property("beta", &hpp::EulerAngles<double>::getBeta, &hpp::EulerAngles<double>::setBeta)
    .add_property("gamma", &hpp::EulerAngles<double>::getGamma, &hpp::EulerAngles<double>::setGamma)
;
boost::python::class_<std::vector<hpp::EulerAngles<float>>>("EulerAnglesFVec")
    .def(boost::python::vector_indexing_suite<std::vector<hpp::EulerAngles<float>>>())
;
boost::python::class_<std::vector<hpp::EulerAngles<double>>>("EulerAnglesDVec")
    .def(boost::python::vector_indexing_suite<std::vector<hpp::EulerAngles<double>>>())
;

// casesUtils.h //
//////////////////
boost::python::class_<hpp::Experiment<float>>("ExperimentF", 
    boost::python::init<std::string>())
    .add_property("strainRate", &hpp::Experiment<float>::getStrainRate)
    .add_property("tStart", &hpp::Experiment<float>::getTStart)
    .add_property("tEnd", &hpp::Experiment<float>::getTEnd)
    .add_property("F_of_t", &hpp::Experiment<float>::getF_of_t)
    .add_property("L_of_t", &hpp::Experiment<float>::getL_of_t)
    .def("generateNextOrientationAngles", &hpp::Experiment<float>::generateNextOrientationAngles)
;
boost::python::class_<hpp::Experiment<double>>("ExperimentD", 
    boost::python::init<std::string>())
    .add_property("strainRate", &hpp::Experiment<double>::getStrainRate)
    .add_property("tStart", &hpp::Experiment<double>::getTStart)
    .add_property("tEnd", &hpp::Experiment<double>::getTEnd)
    .add_property("F_of_t", &hpp::Experiment<double>::getF_of_t)
    .add_property("L_of_t", &hpp::Experiment<double>::getL_of_t)
    .def("generateNextOrientationAngles", &hpp::Experiment<double>::generateNextOrientationAngles)
;

// spectralUtils.h //
/////////////////////
boost::python::class_<hpp::SpectralDatasetID>("SpectralDatasetID");
boost::python::class_<std::vector<hpp::SpectralDatasetID> >("SpectralDatasetIDVec")
    .def(boost::python::vector_indexing_suite<std::vector<hpp::SpectralDatasetID>>())
;
boost::python::class_<hpp::SpectralDatabaseUnified<float>>("SpectralDatabaseUnifiedF",
    boost::python::init<std::string, std::vector<hpp::SpectralDatasetID>, unsigned int, unsigned int>())
;

// gsh.h //
///////////
boost::python::class_<hpp::GSHCoeffs<float>>("GSHCoeffsF", 
    boost::python::init<>())
    .def("getl0Reals", &hpp::GSHCoeffs<float>::getl0Reals)
    .def("getl1Reals", &hpp::GSHCoeffs<float>::getl1Reals)
    .def("getl2Reals", &hpp::GSHCoeffs<float>::getl2Reals)
    .def("getl3Reals", &hpp::GSHCoeffs<float>::getl3Reals)
    .def("getl4Reals", &hpp::GSHCoeffs<float>::getl4Reals)
;
boost::python::class_<hpp::GSHCoeffs<double>>("GSHCoeffsD", 
    boost::python::init<>())
    .def("getl0Reals", &hpp::GSHCoeffs<double>::getl0Reals)
    .def("getl1Reals", &hpp::GSHCoeffs<double>::getl1Reals)
    .def("getl2Reals", &hpp::GSHCoeffs<double>::getl2Reals)
    .def("getl3Reals", &hpp::GSHCoeffs<double>::getl3Reals)
    .def("getl4Reals", &hpp::GSHCoeffs<double>::getl4Reals)
;

// crystal.h //
///////////////
boost::python::class_<hpp::CrystalProperties<float>>("CrystalPropertiesF");
boost::python::class_<hpp::CrystalProperties<double>>("CrystalPropertiesD");

boost::python::class_<hpp::CrystalSolverConfig<float>>("CrystalSolverConfigF");
boost::python::class_<hpp::CrystalSolverConfig<double>>("CrystalSolverConfigD");

boost::python::class_<hpp::CrystalInitialConditions<float>>("CrystalInitialConditionsF")
    .add_property("s_0", &hpp::CrystalInitialConditions<float>::getS0, &hpp::CrystalInitialConditions<float>::setS0)
    .add_property("angles", &hpp::CrystalInitialConditions<float>::getEulerAngles, &hpp::CrystalInitialConditions<float>::setEulerAngles)
;
boost::python::class_<hpp::CrystalInitialConditions<double>>("CrystalInitialConditionsD")
    .add_property("s_0", &hpp::CrystalInitialConditions<double>::getS0, &hpp::CrystalInitialConditions<double>::setS0)
    .add_property("angles", &hpp::CrystalInitialConditions<double>::getEulerAngles, &hpp::CrystalInitialConditions<double>::setEulerAngles)
;

boost::python::def("defaultCrystalPropertiesF", hpp::defaultCrystalProperties<float>);
boost::python::def("defaultCrystalPropertiesD", hpp::defaultCrystalProperties<double>);

boost::python::def("defaultConservativeCrystalSolverConfig", hpp::defaultConservativeCrystalSolverConfig<float>);
boost::python::def("defaultConservativeCrystalSolverConfig", hpp::defaultConservativeCrystalSolverConfig<double>);

boost::python::def("defaultCrystalInitialConditionsF", hpp::defaultCrystalInitialConditions<float>);
boost::python::def("defaultCrystalInitialConditionsD", hpp::defaultCrystalInitialConditions<double>);

boost::python::class_<hpp::Crystal<float>>("CrystalF", 
    boost::python::init<const hpp::CrystalProperties<float>&, const hpp::CrystalSolverConfig<float>&, const hpp::CrystalInitialConditions<float>&>())
;
boost::python::class_<hpp::Crystal<double>>("CrystalD", 
    boost::python::init<const hpp::CrystalProperties<double>&, const hpp::CrystalSolverConfig<double>&, const hpp::CrystalInitialConditions<double>&>())
;

boost::python::class_<std::vector<hpp::Crystal<float>>>("CrystalFVec")
    .def(boost::python::vector_indexing_suite<std::vector<hpp::Crystal<float>>>())
;
boost::python::class_<std::vector<hpp::Crystal<double>>>("CrystalDVec")
    .def(boost::python::vector_indexing_suite<std::vector<hpp::Crystal<double>>>())
;

boost::python::class_<hpp::Polycrystal<float>>("PolycrystalF", 
    boost::python::init<const std::vector<hpp::Crystal<float>>&>())
    .def("setToInitialConditionsRandomOrientations", &hpp::Polycrystal<float>::setToInitialConditionsRandomOrientations)
    .def("setToInitialConditions", &hpp::Polycrystal<float>::setToInitialConditions)
    .def("getEulerAnglesZXZActive", &hpp::Polycrystal<float>::getEulerAnglesZXZActive)
    .def("step", &hpp::Polycrystal<float>::stepVelocityGradient)
    .def("getGSHCoeffs", &hpp::Polycrystal<float>::getGSHCoeffs);
    //.def("getPoleHistogram", getPoleHistogram)
//;
boost::python::class_<hpp::Polycrystal<double>>("PolycrystalD", 
    boost::python::init<const std::vector<hpp::Crystal<double>>&>())
    .def("setToInitialConditionsRandomOrientations", &hpp::Polycrystal<double>::setToInitialConditionsRandomOrientations)
    .def("setToInitialConditions", &hpp::Polycrystal<double>::setToInitialConditions)
    .def("getEulerAnglesZXZActive", &hpp::Polycrystal<double>::getEulerAnglesZXZActive)
    .def("step", &hpp::Polycrystal<double>::stepVelocityGradient)
    .def("getGSHCoeffs", &hpp::Polycrystal<double>::getGSHCoeffs);
    //.def("getPoleHistogram", getPoleHistogram)
//;

boost::python::def("defaultCrystalSpectralDatasetIDs", hpp::defaultCrystalSpectralDatasetIDs);
