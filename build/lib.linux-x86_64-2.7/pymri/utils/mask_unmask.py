import nibabel as nib
import numpy as np


def get_mask_from_nifti(nifti_location):
    mask = nib.load(nifti_location).get_data()
    mask = mask.astype(bool)
    return mask


def mask(data, mask, transpose=False):
    X = data[mask]
    if transpose:
        X = X.T
    return X


def unmask(data, mask):
    unmasked_data = np.zeros(mask.shape, dtype=data.dtype)
    unmasked_data[mask] = data
    return unmasked_data
