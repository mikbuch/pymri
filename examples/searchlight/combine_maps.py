from fnmatch import fnmatch
import os
import nibabel as nib
import numpy as np

from scipy.stats import binom_test, ttest_1samp

results_path = '/tmp/results/'

files = []

for file in os.listdir(results_path):
    if fnmatch(file, '*.MNI152.nii.gz'):
        files.append(file)

mask_path = '/usr/share/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'

mask = nib.load(mask_path).get_data().astype(bool)
img_shape = mask.shape

maps = np.zeros((len(files),) + mask.sum())

for i in range(len(files)):
    maps[i] = nib.load(files[i]).get_data()[mask]

binom_results = binom_test(maps)
ttest_results = ttest_1samp(maps, 0.5)
