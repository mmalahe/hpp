import h5py
from numpy import array

f = h5py.File("parallelOutput.hdf5", "r")
tensor_hdf5 = f['tensorArray']
tensor = array(tensor_hdf5)
print tensor[10,11,:,:]
