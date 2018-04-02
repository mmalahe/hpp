#!/bin/sh
set -e

# This script installs the dependencies for Ubuntu 16.04

# Install the dependencies available from the package manager
sudo apt update
sudo apt install tar --yes
sudo apt install wget --yes
sudo apt install make --yes
sudo apt install build-essential --yes
sudo apt install cmake --yes
sudo apt install python3 python3-dev libpython3-dev --yes
sudo apt install libopenmpi-dev --yes
sudo apt install libblas-dev --yes
sudo apt install liblapack-dev liblapacke-dev --yes
sudo apt install libeigen3-dev --yes
sudo apt install libhdf5-serial-dev libhdf5-openmpi-dev --yes
sudo apt install libtclap-dev --yes
sudo apt install libboost-python-dev --yes

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
