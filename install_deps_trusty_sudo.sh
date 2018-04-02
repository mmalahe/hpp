#!/bin/sh
set -e

# This script installs the dependencies for Ubuntu 14.04

# Install the dependencies available from the package manager
sudo apt install libopenmpi-dev
sudo apt install libblas-dev
sudo apt install liblapack-dev
sudo apt install libeigen3-dev
sudo apt install libhdf5-dev 
sudo apt install libhdf5-mpi-dev
sudo apt install libtclap-dev
sudo apt install libboost-python-dev

# Install the dependencies that require a manual installation
mkdir -p deps
cd deps

# FFTW
wget www.fftw.org/fftw-3.3.7.tar.gz
tar -xzf fftw-3.3.7.tar.gz
cd fftw-3.3.7
./configure \
--enable-shared \
--enable-mpi \
--enable-openmp \
--enable-threads
make
sudo make install
cd ..

# Sufficient CMake version
wget -O cmake.sh https://cmake.org/files/v3.10/cmake-3.10.0-rc1-Linux-x86_64.sh 
sudo sh cmake.sh --skip-license --exclude-subdir --prefix=/usr/local

# Exit deps folder
cd ..

# Ensure libs can be found and linked correctly
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
