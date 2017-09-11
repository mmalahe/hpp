import sys
sys.path.append("../../src")
from crystal import *

euler_angles = [[0.0,0.0,0.0]]
plane_normals = [   [0,0,1],
                    [0,1,1],
                    [0,1,0],
                    [1,0,1],
                    [1,1,1],
                    [1,0,0],
                    [1,1,0]]
                    
plotPoleFigures(euler_angles, plane_normals)
