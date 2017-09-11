cd build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
#~ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
cores=$(grep -c ^processor /proc/cpuinfo)
make -j $cores
cd ..
