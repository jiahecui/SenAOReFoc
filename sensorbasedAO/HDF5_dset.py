import h5py
import numpy as np
import scipy as sp
from scipy import interpolate
from config import config

def make_dset(group, name, data):
    """
    Function to create HDF5 dataset with specified shape
    """
    group.create_dataset(name, (0,) + data.shape, maxshape = (config['DM']['actuator_num'] * 2,) + data.shape, dtype = data.dtype)

def dset_append(group, name, data):
    """
    Function to append an element to end of dataset
    """
    shape = list(group[name].shape)
    shape[0] += 1
    group[name].resize(shape)
    group[name][-1, ...] = data

def get_mat_dset():
    """
    Function to load .mat file image and interpolate to suitable size for analysis
    """
    f = h5py.File('sensorbasedAO/WrappedPhase_IMG_Blastocyte.mat','r')
    data = f.get('WrappedPhase')
    data = np.array(data[-1,...])
    x, y = (np.arange(164) for i in range(2))
    interp_func = sp.interpolate.RectBivariateSpline(x, y, data, kx = 3, ky = 3)
    xx, yy = (np.arange(656) for i in range(2))
    data_interp = interp_func(xx, yy)

    return data_interp