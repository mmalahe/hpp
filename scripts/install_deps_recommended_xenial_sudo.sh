#!/bin/sh
set -e

# The directory this script is in
DIR=`dirname "$(readlink -f "$0")"`

# This script installs the recommended dependencies for Ubuntu 16.04
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt install --yes $(cat ${DIR}/recommended_pkglist_xenial.txt)
sudo pip3 install recordclass
