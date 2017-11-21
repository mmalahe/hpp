/// @file python.cpp
/// @author Michael Malahe
/// @brief The Python interface to the lbirary.

#include <boost/python.hpp>
#include <hpp/tensor.h>

char const *firstMethod() {
    return "This is the first try.";
}

BOOST_PYTHON_MODULE(hpppy) {
    boost::python::def("getTryString", firstMethod);
    // Tensor class
    boost::python::class_<hpp::Tensor2<float>>("Tensor2F", 
        boost::python::init<const unsigned int, const unsigned int>())
        .def("getn1", &hpp::Tensor2<float>::getn1);
}