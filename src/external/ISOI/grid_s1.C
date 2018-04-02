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

#include <hpp/external/ISOI/grid_generation.h>

namespace isoi {

std::vector<double> grid_s1(int resol)
{
	int grids=6;
	std::vector <double> Points;
	std::vector <float>:: iterator temp,temp2,temp1;
	Points.resize(0);

	int number_points=pow(2,resol)*grids;
	float interval=2*M_PI/number_points;
	for(int i=0;i<number_points;i++)
		Points.push_back(interval/2 + i*interval);

	return Points;
}

}
