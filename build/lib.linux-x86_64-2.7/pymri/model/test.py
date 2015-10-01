from datasets import RawDataset
from ram.ram_usage_proc import usage_print


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
    nifti_locations = '../../examples/nifti_PREPROC_sub001.txt',
    tr = 2.5
    )

dataset.nifti_to_array_masked('../../examples/mask_MNI_152.nii.gz')

X = dataset.data_raw
y = dataset.data_raw_category

###############################################################################
#                                                                             #
#   SVC                                                                       #
#                                                                             #
###############################################################################
### Define the estimator
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=0.01)

### Dimension reduction #######################################################

from sklearn.feature_selection import SelectKBest, f_classif

### Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. We set the number of features to be selected to 500
feature_selection = SelectKBest(f_classif, k=500)

# We have our classifier (SVC), our feature selection (SelectKBest), and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])

### Fit and predict ###########################################################

anova_svc.fit(X, y)
y_pred = anova_svc.predict(X)

### Visualisation #############################################################

### Look at the discriminating weights
coef = clf.coef_
# reverse feature selection
coef = feature_selection.inverse_transform(coef)
# reverse masking
# coef = masking.unmask(coef[0], mask)

# # We use a masked array so that the voxels at '-1' are displayed
# # transparently
# act = np.ma.masked_array(coef, coef == 0)

# plot_haxby(act, 'SVC')
# pl.savefig('haxby/haxby_svm.pdf')
# pl.savefig('haxby/haxby_svm.eps')

# import numpy as np
# from scipy import stats
# import nibabel as nib

# house = []
# face = []

# out = np.zeros(dataset.single_vol_shape)

# dataset.data_raw = dataset.data_raw.T

# for i in range(len(dataset.data_raw)):
    # print('%d/%d'%(i+1, len(dataset.data_raw)))
    # for j in range(len(dataset.data_raw[i])):
        # for k in range(len(dataset.data_raw[i][j])):
            # for l in range(len(dataset.data_raw[i][j][k])):
                # if l % 16 < 8:
                    # house.append(dataset.data_raw[i][j][k][l])
                # else:
                    # face.append(dataset.data_raw[i][j][k][l])
            # out[i][j][k] = float(stats.ttest_ind(np.array(house), np.array(face))[0])
            # # print('%d %d %d: %f'%(i, j, k, out[i][j][k]))
            # # print(house)
            # # print('')
            # # print(face)
            # # print('')
            # house = []
            # face = []

# img = nib.Nifti1Image(out, np.eye(4))
# img.to_filename('output_stats.nii.gz')
