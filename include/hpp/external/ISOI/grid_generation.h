/* -----------------------------------------------------------------------------
 *
 *  Copyright (C) 2009  Anna Yershova, Swati Jain, 
 *                      Steven M. LaValle, Julie C. Mitchell
 *
 *
 *  This file is part of the Incremental Successive Orthogonal Images (ISOI)
 *
 *  ISOI is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  ISOI is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this software; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about ISOI see http://rotations.mitchell-lab.org/
 *
 *----------------------------------------------------------------------------- */

#include <iostream>
#include <fstream>
#include <stdbool.h>
#include <cmath>
#include <vector>

#define SIMPLE_GRID 1
#define LAYERED_GRID 2

//these two functions were taekn from the HEALPIX source code written in C
extern "C"
{
	long nside2npix(const long); 
	void pix2ang_nest(long,long,double*,double*);
	void mk_pix2xy(int *, int *);
}

namespace isoi {

std::vector <double> grid_s1(int);
bool healpix_wrapper(int);
bool hopf2quat(std::vector < std::vector<double> >);
bool simple_grid(int);
bool layered_grid(int);

}

 
