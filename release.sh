set -e
mkdir -p build-release
cd build-release
cmake -DCMAKE_BUILD_TYPE=Release .. 
cores=$(grep -c ^processor /proc/cpuinfo)
make -j $cores
cd ..
