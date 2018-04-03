#!/bin/sh
set -e

# The directory this script is in
DIR=`dirname "$(readlink -f "$0")"`

${DIR}/install_deps_mandatory_xenial.sh
${DIR}/install_deps_recommended_xenial.sh
