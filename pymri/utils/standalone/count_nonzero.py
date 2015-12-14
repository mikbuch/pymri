# PARSE ARGUMENT (NIFTI IMAGE)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()

# GET DATA FROM NIFTI
import nibabel
img = nibabel.load(args.image)
data = img.get_data()

# COUNT NON ZERO VOXELS
import numpy as np
nonzero = np.count_nonzero(data)

print(nonzero)
