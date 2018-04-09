/// @file python.cpp
/// @author Michael Malahe
/// @brief The Python interface to the library.

#include <hpp/config.h>
#include <boost/python.hpp>
#include <hpp/tensor.h>
#include <hpp/casesUtils.h>
#include <hpp/spectralUtils.h>
#include <hpp/crystal.h>
#include <hpp/continuum.h>
#include <hpp/python.h>

// The module
BOOST_PYTHON_MODULE(hpppy) {
    #include "python_common.cpp"
}
