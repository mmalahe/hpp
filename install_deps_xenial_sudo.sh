#!/bin/sh
set -e

# This script installs the dependencies for Ubuntu 16.04

# Install the dependencies available from the package manager
sudo apt install build-essential
sudo apt install cmake
sudo apt install python3 python3-dev libpython3-dev
sudo apt install libopenmpi-dev
sudo apt install libblas-dev
sudo apt install liblapack-dev liblapacke-dev
sudo apt install libeigen3-dev
sudo apt install libhdf5-serial-dev libhdf5-openmpi-dev
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

# Exit deps folder
cd ..
