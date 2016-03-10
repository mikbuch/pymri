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
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests


############################################
#       PATHS AND SETTINGS
############################################

# Preloaded maps in numpy array format or separate nifti files
maps_path = '/amu/master/neuronus_2016_searchlight/results/maps.npy'
if maps_path.endswith('.npy'):
    numpy_array_load = True
else:
    numpy_array_load = False
mask_path = os.path.join(
    os.environ['FSLDIR'], 'data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    )

p_level = 0.05
corrected = True
correction_method = ''

output_directory = './'
if corrected:
    zscore_map_filename = 'thr_zscore_p%s_corrected.nii.gz' % str(p_level)
else:
    zscore_map_filename = 'thr_zscore_p%s_uncorrected.nii.gz' % str(p_level)

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
#   LOAD INFORMATION BASED MAPS
############################################

# Load from separate nifti files in specified directory or from single *.npy
# file (regardless of source maps are already in MNI152).
if numpy_array_load:
    maps = np.load(maps_path)
else:
    files = []

    for file in os.listdir(maps_path):
        if fnmatch(file, '*MNI152.nii.gz'):
            files.append(os.path.join(maps_path, file))

    maps = np.zeros((len(files), mask.sum()))

    for i in range(len(files)):
        maps[i] = nib.load(files[i]).get_data()[mask]


############################################
#   COMBINE MAPS
############################################

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

if corrected:
    p_reject = multipletests(p_values, alpha=p_level, method='fdr_bh')[0]
    zscore_masked[-p_reject] = 0.0
    zscore_masked[zscore_masked <= 3.0] = 0.0
else:
    zscore_masked[p_values > p_level] = 0.0
    zscore_masked[zscore_masked <= 0.0] = 0.0

print(zscore_masked.astype('bool').sum())

zscore = np.zeros(results[0].shape)
zscore[maps_mask] = zscore_masked

thr_zscore = np.zeros(mask.shape)
thr_zscore[mask] = zscore

############################################
#   SAVE RESULTS
############################################
thr_zscore_img = nib.Nifti1Image(thr_zscore, mask_img.get_affine())
nib.save(thr_zscore_img, os.path.join(output_directory, zscore_map_filename))


############################################
#   VISUALISE RESULTS
############################################

# plot classification distributions
fig, ax = plt.subplots(figsize=(20, 10))
n, bins, patches = plt.hist(mean_map[mean_map > 0], bins=31)
cm = plt.cm.get_cmap('autumn')
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p, v in zip(col, patches, bins):
    if v > 0.5:
        plt.setp(p, 'facecolor', cm(c-0.5*(1-c)))
    else:
        plt.setp(p, 'facecolor', 'white')
ax.tick_params(axis='x', pad=20)
ax.set_xlabel('accuracy score', labelpad=15)
ax.set_ylabel('classifications performed', labelpad=30)
plt.rcParams.update({'font.size': 28})
plt.tight_layout()

# open in fslview
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

plt.show()
