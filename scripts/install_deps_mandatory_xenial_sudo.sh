#!/bin/sh
set -e

# This script installs the dependencies for Ubuntu 16.04

# The directory this script is in
DIR=`dirname "$(readlink -f "$0")"`

# Install the dependencies available from the package manager
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt install --yes $(cat ${DIR}/mandatory_pkglist_xenial.txt)

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
