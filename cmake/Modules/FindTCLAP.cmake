# - Find the TCLAP library
#
# Usage:
#   find_package(TCLAP [REQUIRED] [QUIET])
#
# It sets the following variables:
#   TCLAP_FOUND               ... true if TCLAP is found on the system
#   TCLAP_INCLUDES            ... TCLAP include directory
#
# The following variables will be checked by the function
#   TCLAP_ROOT                ... if set, the libraries are exclusively searched
#                                 under this path

# Check if we can use PkgConfig
find_package(PkgConfig)

#Determine from PKG
if( PKG_CONFIG_FOUND AND NOT TCLAP_ROOT )
  pkg_check_modules( PKG_TCLAP QUIET "tclap" )
endif()

if( TCLAP_ROOT )

  #find includes
  find_path(
    TCLAP_INCLUDES
    NAMES "tclap/CmdLine.h"
    PATHS ${TCLAP_ROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
  )

else()
  
  #find includes
  find_path(
    TCLAP_INCLUDES
    NAMES "tclap/CmdLine.h"
    PATHS ${PKG_TCLAP_INCLUDE_DIRS} ${INCLUDE_INSTALL_DIR}
  )

endif( TCLAP_ROOT )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TCLAP DEFAULT_MSG
                                  TCLAP_INCLUDES)

mark_as_advanced(TCLAP_INCLUDES)

