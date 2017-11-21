#ifndef HPP_PYTHON_H
#define HPP_PYTHON_H

#include <boost/python.hpp>

namespace hpp
{

template <class T>
boost::python::list toPythonList(const std::vector<T>& vec) {
    boost::python::list list;
    for (const auto& v : vec) {
        list.append(v);
    }
    return list;
}

template <class T>
boost::python::list toStdVector(const boost::python::list& list) {
    int length = boost::python::len(list);
    std::vector<T> vec(length);
    for (int i=0; i<length; i++) {
        vec[i] = boost::python::extract<T>(list[i]);
    }
    return vec;
}

} //END NAMESPACE HPP

#endif /* HPP_PYTHON_H */