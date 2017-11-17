Mandatory dependencies:
-----------------------
C/C++ dependencies:
- Should all be encoded in the CMakeLists.txt

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

