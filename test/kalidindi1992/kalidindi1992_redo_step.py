""" Repeat a single step from a file dump of the state.
"""

# Modules
import sys
sys.path.append("../../src")
from kalidindi1992 import *

# Read in filename from command line if specified
filename = "dump.txt"
if len(sys.argv) > 1:
    filename = sys.argv[1]

# Read in state
exec(open(filename,"r").read())

# Replay step
step_good, new_dt = polycrystal.step(F_next, dt)
