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

std::vector<Quaternion> hopf2quat(std::vector < std::vector <double> > Points)
{
    std::vector<Quaternion> quatList(Points.size());
    
	for(int i=0;i<Points.size();i++)
	{
		quatList[i].a = std::cos(Points[i][0]/2)*std::cos(Points[i][2]/2);
		quatList[i].b = std::cos(Points[i][0]/2)*std::sin(Points[i][2]/2);
		quatList[i].c = std::sin(Points[i][0]/2)*std::cos(Points[i][1]+Points[i][2]/2);
        quatList[i].d = std::sin(Points[i][0]/2)*std::sin(Points[i][1]+Points[i][2]/2);
	}

	return quatList;
}

}
