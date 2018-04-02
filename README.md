Docs:
-----
[FFTW](mmalahe.com/hpp/doc)

Mandatory dependencies:
-----------------------
C/C++ dependencies:
- [FFTW](http://www.fftw.org/download.html) 3.3.5 or greater

CUDA:
- Toolkit version 7.5 or higher.

Python modules:
- scipy
- numpy
- recordclass
- matplotlib

Optional dependencies
---------------------

Boost.Python:
- Allows for the Python interface to the library to be built.

Doxygen and doxypypy:
- Allow for generation of documentation
- Doxygen: "sudo apt-get install doxygen"
- doxypypy: "conda install doxypypy". Likely that there won't be an official one. In that case, pick a channel from the list. e.g. "conda install -c chen doxypypy".

ffmpeg:
- Allows for creation of videos

Default building instructions
-----------------------------
We recommend an out of source CMake build. The easiest is to run
~~~~
sh release.sh
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

Default installation instructions
---------------------------------
Enter the build directory, which will be "build-release" for the default settings, and run
~~~~
make install
~~~~

Custom build options
--------------------
At the configuration step there are a number of options available.

Installing custom FFTW
----------------------
You may need to build your own FFTW (http://www.fftw.org/download.html) if your package manager doesn't provide 3.3.5 and above.

Configure it like this (assuming you want a non-default path, otherwise leave off "--prefix=..."):
~~~~
./configure --prefix=/your/install/prefix \
--enable-shared \
--enable-mpi \
--enable-openmp \
--enable-threads
~~~~
Optionally also include instructions available on your machine:
~~~~
./configure --prefix=/your/install/prefix \
--enable-shared \
--enable-mpi \
--enable-openmp \
--enable-threads \
--enable-sse2 \
--enable-avx \
--enable-avx2 \
--enable-fma
~~~~

