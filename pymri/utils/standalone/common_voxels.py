# PARSE ARGUMENT (NIFTI IMAGE)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("image_01")
parser.add_argument("image_02")
args = parser.parse_args()

# GET DATA FROM NIFTI
import nibabel
img_01 = nibabel.load(args.image_01)
img_02 = nibabel.load(args.image_02)
data_01 = img_01.get_data().astype(bool)
data_02 = img_02.get_data().astype(bool)

# COUNT NON ZERO VOXELS
import numpy as np
# common = (data_01 == data_02).sum()


cnt=0
for i in range(data_01.shape[0]):
    for j in range(data_01.shape[1]):
        for k in range(data_01.shape[2]):
            if (data_01[i][j][k] == 1) and (data_02[i][j][k] == 1):
                cnt += 1
                print('%d, %d, %d' % (i, j, k))
print cnt
