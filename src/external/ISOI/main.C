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

#include <cstdlib>
#include <hpp/external/ISOI/grid_generation.h>

int main(int argc,char ** argv)
{
	if(argc<3)
	{
		cout << "Usage: ./S3_Grid <type of grid> <resolution>" << endl;
		cout << "Grids types: 1.Simple Grid  2.Layered Grid" << endl;
		exit(-1);
	}
	int grid_choice=atoi(argv[1]);
	int resolution=atoi(argv[2]);
	bool result;
	switch(grid_choice)
	{
		case SIMPLE_GRID:
			result=simple_grid(resolution);
			break;
		case LAYERED_GRID:
			result=layered_grid(resolution);
			break;
	}
	if(result)
	{
		printf("Output file generated\n");
		exit(0);
	}
	else
	{
		printf("Terminating the program\n");
		exit(1);
	}
}
		
