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

HealpixPsiGrid healpix_psi_grid(int resol) {
    HealpixPsiGrid grid;

	long int Nside=0,numpixels=0;	
	double theta=0,phi=0;
		
	grid.Psi_Points=grid_s1(resol);
	if(grid.Psi_Points.size()==0)
		throw std::runtime_error("No points.");
    
	Nside=pow(2,resol);
	numpixels=nside2npix(Nside);
	grid.Healpix_Points.resize(numpixels);
	for(long int i=0;i<numpixels;i++)
	{		
		pix2ang_nest(Nside,i,&theta,&phi);
		grid.Healpix_Points[i].theta = theta;
        grid.Healpix_Points[i].phi = phi;
	}
    return grid;
}

std::vector<Quaternion> simple_grid_quaternion(int resol)
{
    HealpixPsiGrid grid = healpix_psi_grid(resol);   
    int nHealpix = grid.Healpix_Points.size();
    int nPsi = grid.Psi_Points.size();
    std::vector<Quaternion> quatList(nHealpix*nPsi);
    
    // Pre-computed things
    std::vector<double> cosPsiOver2(nPsi);
    std::vector<double> sinPsiOver2(nPsi);    
    for(int j=0;j<nPsi;j++) {
        cosPsiOver2[j] = std::cos(grid.Psi_Points[j]/2);
        sinPsiOver2[j] = std::sin(grid.Psi_Points[j]/2);        
    }
    
    double thetaOver2, phi, psi;
    double cosThetaOver2, sinThetaOver2;
    double phiPlusPsiOver2;
    for(int i=0;i<nHealpix;i++)
	{
        thetaOver2 = grid.Healpix_Points[i].theta/2;
        cosThetaOver2 = std::cos(thetaOver2);
        sinThetaOver2 = std::sin(thetaOver2);
        phi = grid.Healpix_Points[i].phi;        
		for(int j=0;j<nPsi;j++)
		{	
            long int idx = i*nPsi + j;
            psi = grid.Psi_Points[j];
            phiPlusPsiOver2 = phi + psi/2;
            quatList[idx].a = cosThetaOver2*cosPsiOver2[j];
            quatList[idx].b = cosThetaOver2*sinPsiOver2[j];
            quatList[idx].c = sinThetaOver2*std::cos(phiPlusPsiOver2);
            quatList[idx].d = sinThetaOver2*std::sin(phiPlusPsiOver2);
		}
	}
    return quatList;
}

}

