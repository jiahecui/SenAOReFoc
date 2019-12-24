import h5py
import numpy as np
from config import config

def make_dset(group, name, data):
    group.create_dataset(name, (0,) + data.shape, maxshape = (config['DM']['actuator_num'] * 2,) + data.shape, dtype = data.dtype)

def dset_append(group, name, data):
    shape = list(group[name].shape)
    shape[0] += 1
    group[name].resize(shape)
    group[name][-1, ...] = data

def get_mat_dset():
    f = h5py.File('sensorbasedAO/WrappedPhase_IMG_Blastocyte.mat','r')
    data = f.get('WrappedPhase')
    data = np.array(data[-1,...])

    return data