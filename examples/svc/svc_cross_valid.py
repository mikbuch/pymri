'''
svc_cross_valid.py
'''
k_features = 784


###############################################################################
#
#        LOAD DATA
#
###############################################################################
from pymri.dataset import DatasetManager
# dataset settings
ds = DatasetManager(
    path_input='/home/jesmasta/amu/master/nifti/bold/',
    contrast=(('PlanTool_0', 'PlanTool_5'), ('PlanCtrl_0', 'PlanCtrl_5')),
    k_features = k_features,
    normalize = False,
    sizes=(0.75, 0.25)
    )
# load data
ds.load_data()

###############################################################################
#
#        CHOOSE ROIs
#
###############################################################################
roi_selection = ('SelectKBest', 'PCA')

# select feature reduction method
ds.feature_reduction(roi_selection='SelectKBest')

k_features = ds.X_processed.shape[1]
print(k_features)

'''
roi - 'SelectKBest' xor 'PCA' xor 'RBM' xor 'path_to_mask.nii.gz'


NOTICE: the same number of features has to be used for each mask.
So if number of features in mask differ then smaller number of feautres is
chosen (k). From the more numerous masks only k-greatest is taken.

If mask and automatic selection is chosen then number of features for automatic
selection equals number of features in mask.

If two automatic methods are being compared then user has to specify number of
features. Default is 784.
'''


###############################################################################
#
#        CROSS VALIDATION
#
###############################################################################

# perform LeavePOut n times
n = 100

import numpy as np
accuracies = np.zeros(shape=(n,))

for i in range(n):

    print('testing iteration: %d' % i)

    ###########################################################################
    #
    #        SPLIT DATA
    #
    ###########################################################################

    # get training, validation and test datasets for specified roi
    training_data, validation_data, test_data = ds.split_data()

    ###########################################################################
    #
    #        CREATE MODEL
    #
    ###########################################################################

    # Define the estimator: supported vector classifier
    from sklearn.svm import SVC
    svc = SVC(kernel='linear', C=0.01)

    svc.fit(training_data[0], training_data[1])

    from sklearn.metrics import accuracy_score
    # record the best result
    accuracies[i] = accuracy_score(test_data[1], svc.predict(test_data[0]))


mean_accuracy = accuracies.mean()
print('\n\nmean accuracy: %f' % mean_accuracy)

###############################################################################
#
#   VISUALIZE
#
###############################################################################
import matplotlib.pyplot as plt

mean_accuracies = np.zeros(shape=(n,))
for i in range(n):
    mean_accuracies[i] = accuracies[:i+1].mean()

# plot best accuracy for particular validation
plt.scatter(
    range(len(accuracies)), accuracies,
    marker='o', s=120, c='r', label='accuracy'
    )
# to show wether overall average changes with number of validations
plt.plot(
    range(len(mean_accuracies)), mean_accuracies, label='mean accurcy'
    )
plt.ylim(-0.1, 1.1)
plt.legend(loc=3)
plt.show()
