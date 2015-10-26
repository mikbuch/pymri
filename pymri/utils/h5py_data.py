'''
h5py_data.py

Allows to save and read numpy arrays (to or from your disc).
It uses h5py library.

Based on the following comment on stackoverflow.com by JoshAdel:
http://stackoverflow.com/a/20938742
'''

import numpy as np
import h5py


def save_array(data, data_file='data.h5', dataset='dataset_1'):
    h5f = h5py.File(data_file, 'w')
    h5f.create_dataset(dataset, data=data)

    h5f.close()

def load_array(data_file='data.h5', dataset='dataset_1'):
    h5f = h5py.File(data_file,'r')
    loaded_array = h5f[dataset][:]
    h5f.close()

    return loaded_array
