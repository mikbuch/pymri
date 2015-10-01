"""
    Example of SVC and PCA of the Haxby's database.

    Based strongly on Alexandre Abraham's code:
        https://github.com/AlexandreAbraham/frontiers2013

    Novum is sampling step (split original dataset train and test
    subsets) and classification performed on test data.

    Subject 001, data preprocessed with fsl:
        * brain extraction
        * motion correction
"""


'''
    MODEL CREATION
'''

# ### Load Haxby dataset ######################################################
# from pymri.dataset import datasets_abraham as datasets
import numpy as np
import nibabel

from ram.ram_usage_proc import usage_print
from pymri.model.visualisation import plot_haxby
from pymri.dataset.datasets import get_categories

nifti_paths = '/git/pymri/examples/file_lists/nifti_PREPROC_sub001_tst.txt'
# nifti_paths = '/git/pymri/examples/file_lists/nifti_PREPROC_sub002.txt'
session_target = '/git/pymri/examples/file_lists/session_target/st.txt'
conditions_target = '/git/pymri/examples/file_lists/conditions_target/ct.txt'
shape_out = (91, 109, 91, 216)

# get data, target and affine
X, y, affine =  \
    get_categories(nifti_paths, session_target, conditions_target, shape_out)

usage_print()

# mask nifti, file location
mask = '/git/pymri/examples/MNI152_T1_2mm_brain_mask.nii.gz'
# get background image (example functional image)
bg_img = \
    nibabel.load('/git/pymri/examples/MNI152_T1_2mm_brain.nii.gz').get_data()
# X.shape is (91,109, 91, 216)
# and mask.shape is (91, 109, 91)

usage_print()

# ### Masking step
from pymri.utils import masking, signal
from nibabel import Nifti1Image

# Mask data
X_img = Nifti1Image(X, affine)
X = masking.apply_mask(X_img, mask, smoothing_fwhm=4)
# X = signal.clean(X, standardize=True, detrend=False)

# ### Sampling ################################################################
from sklearn.cross_validation import train_test_split

# split original dataset into training and testing datasets
X, X_t, y, y_t = train_test_split(
    X, y, test_size=0.25, random_state=42
    )

###############################################################################
#
#   F-score
#
###############################################################################
from sklearn.feature_selection import f_classif
f_values, p_values = f_classif(X, y)
p_values = -np.log10(p_values)
p_values[np.isnan(p_values)] = 0
p_values[p_values > 10] = 10
p_unmasked = masking.unmask(p_values, mask)
plot_haxby(p_unmasked, bg_img, 'F-score')

# save statistical map as nifti image
img = nibabel.Nifti1Image(p_unmasked, np.eye(4))
img.to_filename('output_stats_f_classif.nii.gz')

###############################################################################
#                                                                             #
#   SVC                                                                       #
#                                                                             #
###############################################################################
# Define the estimator
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=0.01)

# ### Dimension reduction #####################################################

from sklearn.feature_selection import SelectKBest, f_classif

# ### Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. We set the number of features to be selected to 500
feature_selection = SelectKBest(f_classif, k=500)

# We have our classifier (SVC), our feature selection (SelectKBest), and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])

# ### Fit and predict #########################################################
from sklearn.metrics import precision_score

# fit data, create hyperplane (model)
anova_svc.fit(X, y)

# predict samples' classes for TRAINING dataset
y_pred = anova_svc.predict(X)
precision_X = precision_score(y, y_pred)
print('train dataset precision: %.2f' % (precision_X))

# predict samples' classes for TESTING dataset
y_pred_t = anova_svc.predict(X_t)
precision_X_t = precision_score(y_t, y_pred_t)
print('test dataset precision: %.2f' % (precision_X_t))

# ### Visualisation (SVC) #####################################################
import numpy as np

# ### Look at the discriminating weights
coef = clf.coef_
# reverse feature selection
coef = feature_selection.inverse_transform(coef)

# reverse masking
coef = masking.unmask(coef[0], mask)

# # We use a masked array so that the voxels at '-1' are displayed
# # transparently
act = np.ma.masked_array(coef, coef == 0)


plot_haxby(act, bg_img, 'SVC')

# save statistical map as nifti image
img = nibabel.Nifti1Image(act, np.eye(4))
img.to_filename('output_stats_svc.nii.gz')
