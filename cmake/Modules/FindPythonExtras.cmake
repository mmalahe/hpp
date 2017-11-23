find_package(PythonInterp REQUIRED)

# Find python site packages directory
# Approach adapted from https://gitlab.com/ideasman42/blender-mathutils/blob/master/CMakeLists.txt,
# revision 2035ab457bc401848edc368e05f2b8b578a4ab02.
# No apparent license associated with the file at the time of retrieval.
if(NOT DEFINED PYTHON_SITE_PACKAGES)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
        OUTPUT_VARIABLE _site_packages
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(PYTHON_SITE_PACKAGES "${_site_packages}" CACHE PATH INTERNAL)
    unset(_site_packages)
endif()