#!/bin/sh
set -e

# This script installs the dependencies for Ubuntu 16.04

# Install the dependencies available from the package manager
apt update
apt install tar --yes
apt install wget --yes
apt install make --yes
apt install build-essential --yes
apt install cmake --yes
apt install python3 python3-dev libpython3-dev --yes
apt install libopenmpi-dev --yes
apt install libblas-dev --yes
apt install liblapack-dev liblapacke-dev --yes
apt install libeigen3-dev --yes
apt install libhdf5-serial-dev libhdf5-openmpi-dev --yes
apt install libtclap-dev --yes
apt install libboost-python-dev --yes

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
make install
cd ..

# Exit deps folder
cd ..
