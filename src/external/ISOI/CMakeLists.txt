# This source files form the ISOI library, available at 
# http://msl.cs.uiuc.edu/~yershova/software/so3sampling/so3sampling.htm
# under the GNU General Public License version 2. More license details in
# the source files.
set(ISOI_CPP_SOURCES_RELATIVE simple_grid.C layered_grid.C hopf2quat.C grid_s1.C nside2npix.c pix2ang_nest.c mk_pix2xy.c)
set(ISOI_CPP_SOURCES "")
foreach(fname ${ISOI_CPP_SOURCES_RELATIVE})
  set(ISOI_CPP_SOURCES ${ISOI_CPP_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/${fname}")
endforeach(fname)
set(ISOI_CPP_SOURCES ${ISOI_CPP_SOURCES} PARENT_SCOPE)