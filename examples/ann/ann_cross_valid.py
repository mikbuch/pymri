'''
NOTICE: first masks (methods of feature selection/extraction) has to be chosen

1. Load data.
2. Choose ROIs (masks) to compare (n>=2).
3. Create model (classifier).
4. Train, test (and cross-validate) classifier using n-times masked data.
6. Compare performance of the classifier for diffrent ROIs -
the better classification, the more important the region is for
particular task.

Conclusion: The best classification was achieved using ROI that is crucial
in performing cognitive task.
'''

'''
k_features - only if automatic selection chosen: how many features do we take
'''
k_features = 784
# hidden_neurons = 46
hidden_neurons = 50
epoches = 100


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
    # normalize = True,
    # scale_0_1 = True,
    nnadl = True,
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
# ds.feature_reduction(roi_selection='SelectKBest')
ds.feature_reduction(roi_selection='/amu/master/nifti/bold/roi_mask_plan.nii.gz')

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
n = 10

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
    # artificial neural network
    from pymri.model import fnn

    net = fnn.Network([k_features, hidden_neurons, 2])
    # train and test network
    net.SGD(training_data, epoches, 11, 2.961, test_data=test_data)

    # record the best result
    accuracies[i] = net.best_score/float(len(test_data))

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

plt.scatter(
    range(len(accuracies)), accuracies,
    marker='o', s=120, c='r', label='accuracy'
    )
plt.scatter(
    range(len(mean_accuracies)), mean_accuracies,
    marker='+', s=140, label='mean accurcy'
    )
plt.ylim(-0.1, 1.1)
plt.legend(loc=3)
plt.show()
