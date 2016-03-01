'''
name:
combine_maps.py

type:
script

Combine statistical maps generated with MVPA searchlight analysis.
1. Searchlight calculates maps in subjects' native space.
2. Then individual maps are registered to MNI152 space.

NOTICE:
This version of the script loads maps from specified directory (maps_path).
'''

from fnmatch import fnmatch
import os
import os.path
import nibabel as nib
import numpy as np

from scipy import stats

############################################
#       PATHS AND SETTINGS
############################################
mask_path = os.path.join(
    os.environ['FSLDIR'], 'data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    )
maps_path = '/tmp/results'


p_level = 0.0001

output_directory = './'
zscore_map_filename = 'thr_zscore_p%s.nii.gz' % str(p_level)
mean_map_filename = 'mean_map.nii.gz'
bg_brain_path = os.path.join(
    os.environ['FSLDIR'], 'data/standard/MNI152_T1_2mm_brain.nii.gz'
    )


############################################
#   LOAD MNI152 MASK
############################################
mask_img = nib.load(mask_path)
mask = mask_img.get_data().astype(bool)

############################################
#   COMBINE MAPS
############################################

# Load maps registered to mni152
files = []

for file in os.listdir(maps_path):
    if fnmatch(file, '*MNI152.nii.gz'):
        files.append(file)

maps = np.zeros((len(files), mask.sum()))

for i in range(len(files)):
    maps[i] = nib.load(files[i]).get_data()[mask]

# Averege maps
# mean_map = maps.mean(axis=0)
mean_map = maps.mean(axis=0)
mean_map[maps.astype('bool').sum(axis=0) != maps.shape[0]] = 0.0
mean_map_unmasked = np.zeros(mask.shape)
mean_map_unmasked[mask] = mean_map
mean_map_img = nib.Nifti1Image(mean_map_unmasked, mask_img.get_affine())
nib.save(mean_map_img, os.path.join(output_directory, mean_map_filename))

# Create maps mask. Take only voxels which has informations from all maps
maps_mask = (maps.astype('bool').sum(axis=0) == maps.shape[0])

results = stats.ttest_1samp(maps, 0.5)
t_scores = results[0][maps_mask]
p_values = results[1][maps_mask]
zscore_masked = stats.zscore(t_scores)
zscore_masked[p_values > p_level] = 0.0

# import matplotlib.pyplot as plt
# # plt.plot(t_scores[t_scores.argsort()])
# plt.plot(p_values[t_scores.argsort()])
# plt.show()

# it is allowed only in case of 2 conditions, else you can take only
# positive zscore values
# zscore_masked = np.abs(zscore_masked)
zscore_masked[zscore_masked <= 0.0] = 0.0

zscore = np.zeros(results[0].shape)
zscore[maps_mask] = zscore_masked

thr_zscore = np.zeros(mask.shape)
thr_zscore[mask] = zscore

thr_zscore_img = nib.Nifti1Image(thr_zscore, mask_img.get_affine())
nib.save(thr_zscore_img, os.path.join(output_directory, zscore_map_filename))

import subprocess as sp
cmd = 'fslview %s -l Grey %s -l Red-Yellow %s -l Blue-Lightblue -b %f,%f' % (
    bg_brain_path,
    os.path.join(output_directory, zscore_map_filename),
    os.path.join(output_directory, mean_map_filename),
    mean_map[mean_map > 0.0].min(),
    mean_map.max()
    )
process = sp.Popen(cmd, stdout=sp.PIPE, shell=True)
output = process.communicate()[0]
print(output)
