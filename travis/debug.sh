#!/bin/sh
set -e

mkdir -p build-debug
cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DFFTW_ROOT=/usr/local ..
cores=$(grep -c ^processor /proc/cpuinfo)
make -j $cores
cd ..
