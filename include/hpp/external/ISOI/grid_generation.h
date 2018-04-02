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
#include <stdexcept>

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

struct Quaternion {
    double a, b, c, d;
};

struct S2Point {
    double theta, phi;
};

struct S3Point {
    double theta, phi, psi;
};

struct HealpixPsiGrid {
    std::vector<S2Point> Healpix_Points;
    std::vector<double> Psi_Points;
};

std::vector<double> grid_s1(int);
bool healpix_wrapper(int);
std::vector<Quaternion> hopf2quat(const std::vector<S3Point>&);
HealpixPsiGrid healpix_psi_grid(int);
std::vector<Quaternion> full_grid_quaternion(int);
std::vector<Quaternion> fourfold_symmetry_grid_quaternion(int);

inline Quaternion hopf2quat(S3Point point)
{
    Quaternion q;
    q.a = std::cos(point.theta/2)*std::cos(point.psi/2);
    q.b = std::cos(point.theta/2)*std::sin(point.psi/2);
    q.c = std::sin(point.theta/2)*std::cos(point.phi+point.psi/2);
    q.d = std::sin(point.theta/2)*std::sin(point.phi+point.psi/2);
	return q;
}

}

 
