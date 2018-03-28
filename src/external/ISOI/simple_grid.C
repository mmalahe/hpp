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

bool simple_grid(int resol)
{
	std::vector <double> Psi_Points,temp;
	std::vector < std::vector<double> > Healpix_Points;
	std::vector < std::vector<double> > S3_Points;
	long int Nside=0,numpixels=0;	
	double theta=0,pfi=0;
	bool result;
		
	Psi_Points.resize(0);
	Psi_Points=grid_s1(resol);
	if(Psi_Points.size()==0)
		return false;
		
	Nside=pow(2,resol);
	numpixels=nside2npix(Nside);
	Healpix_Points.resize(0);
	for(long int i=0;i<numpixels;i++)
	{
		temp.resize(0);
		pix2ang_nest(Nside,i,&theta,&pfi);
		temp.push_back(theta);
		temp.push_back(pfi);
		Healpix_Points.push_back(temp);
	}
	
	S3_Points.resize(0);
	for(int i=0;i<Healpix_Points.size();i++)
	{
		for(int j=0;j<Psi_Points.size();j++)
		{
			temp.resize(0);
			temp.push_back(Healpix_Points[i][0]);
			temp.push_back(Healpix_Points[i][1]);
			temp.push_back(Psi_Points[j]);
			S3_Points.push_back(temp);
		}
	}

	result=hopf2quat(S3_Points);	
	return result;
}

}

