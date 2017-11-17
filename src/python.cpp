/// @file python.cpp
/// @author Michael Malahe
/// @brief The Python interface to the lbirary.

#include <boost/python.hpp>

char const *firstMethod() {
    return "This is the first try.";
}

BOOST_PYTHON_MODULE(hpppy) {
    boost::python::def("getTryString", firstMethod);
}