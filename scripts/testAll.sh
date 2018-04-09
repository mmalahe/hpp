#!/bin/sh
set -e

# The directory this script is in
DIR=`dirname "$(readlink -f "$0")"`

while getopts 'v' flag; do
  case "${flag}" in
    v) ctest_verbose='-V' ;;
  esac
done

${DIR}/debug.sh
cd build-debug
ctest ${ctest_verbose}
cd ..

${DIR}/release.sh
cd build-release
ctest ${ctest_verbose}
cd ..
