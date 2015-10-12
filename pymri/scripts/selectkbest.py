
"""
selectkbest.py
script

############################################
######  Written by: Mikolaj Buchwald  ######
############################################

sklearn's selectkbest utility used on loaded data
Transformation function

Input: high dimensional data
Output: k-dimensional data (save to CSV files as train.csv and test.csv)
"""


# ### Load Haxby dataset ######################################################
import numpy as np
import nibabel
from os.path import expanduser
from sklearn.datasets.base import Bunch


# data_dir = expanduser('~') + '/workshops/aiml/data/pymvpa-exampledata/'
data_dir = expanduser('~') + '/downloads/pymvpa-exampledata/'

# create sklearn's Bunch of data
dataset_files = Bunch(
    func=data_dir+'bold.nii.gz',
    session_target=data_dir+'attributes.txt',
    mask=data_dir+'mask.nii.gz',
    conditions_target=data_dir+'attributes_literal.txt'
    )

# fmri_data and mask are copied to break any reference to the original object
bold_img = nibabel.load(dataset_files.func)
fmri_data = bold_img.get_data().astype(float)
affine = bold_img.get_affine()
y, session = np.loadtxt(dataset_files.session_target).astype("int").T
conditions = np.recfromtxt(dataset_files.conditions_target)['f0']
mask = dataset_files.mask

# fmri_data.shape is (40, 64, 64, 1452)
# and mask.shape is (40, 64, 64)

# ### Preprocess data
# Build the mean image because we have no anatomic data
mean_img = fmri_data.mean(axis=-1)


# # ### Restrict to faces and houses
# condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
# X = fmri_data[..., condition_mask]
# y = y[condition_mask]
# # session = session[condition_mask]
# # conditions = conditions[condition_mask]

X = fmri_data
y = y


# ### Masking step
# from utils import masking, signal
from pymri.utils import masking
from nibabel import Nifti1Image

# Mask data
X_img = Nifti1Image(X, affine)
X = masking.apply_mask(X_img, mask, smoothing_fwhm=4)
# X = signal.clean(X, standardize=True, detrend=False)

# ### Sampling ################################################################
from sklearn.cross_validation import train_test_split

# split original dataset into training and testing datasets
X, X_t, y, y_t = train_test_split(
    X, y, test_size=0.4, random_state=42
    )

###############################################################################
#                                                                             #
#   SELECT K BEST                                                             #
#                                                                             #
###############################################################################

from sklearn.feature_selection import SelectKBest, f_classif

# ### Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. We set the number of features to be selected to 500
feature_selection = SelectKBest(f_classif, k=784)

# transform datasets from high import dimensional to k-dimensional
X = feature_selection.fit_transform(X, y)
X_t = feature_selection.transform(X_t)


# save output to csv files
import csv
with open('train.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    for i in range(len(X)):
        a.writerow([y[i]] + list(X[i]))

with open('test.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    for i in range(len(X_t)):
        a.writerow([y[i]] + list(X_t[i]))
