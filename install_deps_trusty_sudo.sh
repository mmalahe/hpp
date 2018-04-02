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
tar -xczf fftw-3.3.7.tar.gz
cd fftw-3.3.7
./configure \
--enable-shared \
--enable-mpi \
--enable-openmp \
--enable-threads
make
make install
cd ..

# Exit deps folder
cd ..
