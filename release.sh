set -e
mkdir -p build-release
cd build-release
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/mmalahe/local .. 
cores=$(grep -c ^processor /proc/cpuinfo)
make -j $cores
cd ..
