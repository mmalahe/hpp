# Installation

To install both the mandatory and recommended dependencies and build and install the library on Ubuntu 16.04 with root:
~~~~
scripts/install_deps_xenial_sudo.sh
scripts/release.sh
cd build-release
sudo make install
~~~~

Otherwise we provide more detailed instructions below.

# Mandatory dependencies
The library cannot be built without these. Most of the mandatory dependencies are available from standard package managers. The others may require manual installation. We provide a script for installing both kinds under Ubuntu 16.04 with root at `scripts/install_deps_mandatory_xenial_sudo.sh`.

## Easy
These are easy to install through package managers or installers on most platforms:
- A C++ compiler with full support for C++11
- [Cmake](https://cmake.org/download/) 3.4 or greater
- [Openmpi](https://www.open-mpi.org/)
- [BLAS](http://www.netlib.org/blas/)
- [LAPACK](http://www.netlib.org/lapack/)
- [LAPACKE](http://www.netlib.org/lapack/lapacke)
- [Eigen 3](http://eigen.tuxfamily.org/)
- [HDF5](https://portal.hdfgroup.org/display/support/Downloads) serial and openmpi versions
- [TCLAP](https://github.com/eile/tclap)
- [FFTW](http://www.fftw.org)

## Requiring manual work

### FFTW version 3.3.5 or greater
You may need to build your own [FFTW](http://www.fftw.org/download.html) if your package manager doesn't provide 3.3.5 and above.

Configure it like this (assuming you want a non-default path, otherwise leave off "--prefix=..."):
~~~~
./configure --prefix=/your/install/prefix \
--enable-shared \
--enable-mpi \
--enable-openmp \
--enable-threads
~~~~
Optionally also include vector instructions available on your processor. E.g. if you have SSE2, AVX, AVX2 and FMA instructions append:
~~~~
--enable-sse2 \
--enable-avx \
--enable-avx2 \
--enable-fma
~~~~

### HDF5 on older distributions
On older distributions, e.g. Ubuntu 14.04, there is no way to simultaneously install the serial and openmpi versions of HDF5. On newer distributions these are separated with a folder structure like this:
~~~~
/usr/include/hdf5/serial
/usr/include/hdf5/openmpi
~~~~
We assume this folder structure, so for older distributions you will need to create this yourself with custom HDF5 installs.

# Recommended dependencies
These are required for important but non-critical functionality. We provide a script for installing these under Ubuntu 16.04 with root at `scripts/install_deps_recommended_xenial_sudo.sh`.

## For the Python interface to the library
- [Python](https://www.python.org/downloads/) 3.5 or greater
- [Boost.Python](https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/index.html)

## For batch runs, plotting and pure Python implementations
- Python modules:
	- [scipy](https://www.scipy.org/)
	- [numpy](https://www.numpy.org/)
	- [recordclass](https://pypi.python.org/pypi/recordclass/0.4.3)
	- [matplotlib](https://matplotlib.org/)

# Optional dependencies

## If you have an Nvidia GPU
- [CUDA](https://developer.nvidia.com/cuda-downloads) Toolkit version 7.5 or higher. 

## For generating the documentation
- [Doxygen](www.doxygen.org)
- [doxypypy](https://pypi.python.org/pypi/doxypypy/0.8.7)

# Building
We recommend an out of source CMake build. The easiest is to run
~~~~
scripts/release.sh
~~~~
which will create the build in the directory build-release. This build script does the following steps, which you can do manually for a more precise configuration.

Make the build directory and enter it:
~~~~
mkdir -p build-release
cd build-release
~~~~

Configure the build
~~~~
cmake .. -DCMAKE_BUILD_TYPE=Release
~~~~
You may want to provide a custom installation directory, in which case instead run
~~~~
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/my/custom/install/dir
~~~~

Build:
~~~~
make -j 8
~~~~
where 8 represents the number of cores available for building.

# Installing
Enter the build directory, which will be "build-release" if you followed the above instructions or executed `scripts/release.sh`. Run
~~~~
make install
~~~~
optionally with root if the install prefix requires root access.

# Custom build options
At the configuration step there are a number of options available, which are automatically selected based on your system's attributes but may be overridden. The options are all preceded by `HPP_` and can be overriden by adding
~~~~
-DHPP_SETTING_NAME=SETTING_VALUE
~~~~
to the CMake command, or modified using the CMake gui.

## What to build
- `HPP_BUILD_PYTHON` (OFF/ON): build the Python interface to the library. Default depends on if you have Python and Boost on your system.
- `HPP_USE_CUDA` (OFF/ON): build the GPU-accelerated functionality for the library. Default depends on whether or not you have CUDA installed on your system.

## Developer options
- `HPP_SHOW_PTX` (OFF/ON): display the PTX info (register counts, shared memory usage etc.) while building CUDA source files. Default is OFF.
- `HPP_CUDA_ARCH_LIST`: a semi-colon separated list of CUDA architectures to build SASS for. Default is the architectures currently attached to your system.

