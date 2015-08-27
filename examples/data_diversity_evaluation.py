from pymri.model.datasets import RawDataset


'''
    FSL PREPROCESSING
'''
# fsl.preprocessing()

'''
    MODEL CREATION
'''
# Raw data collection (from nifti files)
dataset = RawDataset(
    dir_paradigm = '/amu/neurosci/ds105/sub002/model/model001/onsets/',
    nifti_locations = 'nifti_PREPROC_sub001.txt',
    tr = 2.5
    )

dataset.nifti_to_array()

import numpy as np
from scipy import stats
import nibabel as nib

house = []
face = []

out = np.zeros((40, 64, 64))

dataset.data_raw = dataset.data_raw.T

for i in range(len(dataset.data_raw)):
    print('%d/%d'%(i, len(dataset.data_raw)))
    for j in range(len(dataset.data_raw[i])):
        for k in range(len(dataset.data_raw[i][j])):
            for l in range(len(dataset.data_raw[i][j][k])):
                if l % 16 < 8:
                    house.append(dataset.data_raw[i][j][k][l])
                else:
                    face.append(dataset.data_raw[i][j][k][l])
            out[i][j][k] = float(stats.ttest_ind(np.array(house), np.array(face))[0])
            # print('%d %d %d: %f'%(i, j, k, out[i][j][k]))
            # print(house)
            # print('')
            # print(face)
            # print('')
            house = []
            face = []

img = nib.Nifti1Image(out, np.eye(4))
img.to_filename('test4d.nii.gz')
