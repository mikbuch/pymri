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
hidden_neurons = 46
epochs = 300
minibatch_size = 11
eta = 2.95

# perform LeavePOut n times
n_times_LeavePOut = 20


###############################################################################
#
#        LOAD DATA
#
###############################################################################
from pymri.dataset.datasets import DatasetManager
# dataset settings
path_base = '/home/jesmasta/amu/master/nifti/bold/'
ds = DatasetManager(
    path_bold=path_base + 'bold.nii.gz',
    path_attr=path_base + 'attributes.txt',
    path_attr_lit=path_base + 'attributes_literal.txt',
    path_mask_brain=path_base + 'mask.nii.gz',
    contrast=(('PlanTool_0', 'PlanTool_5'), ('PlanCtrl_0', 'PlanCtrl_5')),
    nnadl = True
    )
# load data
ds.load_data()

###############################################################################
#
#        CHOOSE ROIs
#
###############################################################################

# select feature reduction method
# ds.feature_reduction(
    # roi_selection='SelectKBest', k_features=k_features, normalize=True
    # )
ds.feature_reduction(
    roi_selection='/amu/master/nifti/bold/roi_mask_plan.nii.gz', normalize=True
    )

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

import numpy as np
accuracies = np.zeros(shape=(n_times_LeavePOut,))

for i in range(n_times_LeavePOut):

    print('testing iteration: %d' % i)

    ###########################################################################
    #
    #        SPLIT DATA
    #
    ###########################################################################

    # get training, validation and test datasets for specified roi
    # training_data, validation_data, test_data = ds.split_data()
    training_data, test_data, vd = ds.split_data(sizes=(0.75,0.25))

    ###########################################################################
    #
    #        CREATE MODEL
    #
    ###########################################################################
    # artificial neural network
    from pymri.model import fnn

    net = fnn.Network([k_features, hidden_neurons, 2])
    # train and test network
    net.SGD(training_data, epochs, minibatch_size, eta, test_data=test_data)

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

mean_accuracies = np.zeros(shape=(n_times_LeavePOut,))
for i in range(n_times_LeavePOut):
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
