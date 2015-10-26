"""
neural-networks-and-deep-learning dataset generation (transformation)
"""

__author__ = "Mikolaj Buchwald"
__contact__ = "mikolaj.buchwald@gmail.com"


import numpy as np

def load_nifti(data_dir):
    import numpy as np
    import nibabel
    from sklearn.datasets.base import Bunch

    data_dir = data_dir

    # create sklearn's Bunch of data
    dataset_files = Bunch(
        func=data_dir+'bold.nii.gz',
        session_target=data_dir+'attributes.txt',
        mask=data_dir+'mask.nii.gz',
        conditions_target=data_dir+'attributes_literal.txt'
        )

    # fmri_data and mask are copied to break reference to the original object
    bold_img = nibabel.load(dataset_files.func)
    fmri_data = bold_img.get_data().astype(float)
    affine = bold_img.get_affine()
    y, session = np.loadtxt(dataset_files.session_target).astype("int").T
    conditions = np.recfromtxt(dataset_files.conditions_target)['f0']
    mask = dataset_files.mask



    # ### Restrict to specified conditions


    # condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
    # condition_mask = np.logical_or(
        # np.logical_or(
            # conditions == 'PlanCtrl_5', conditions == 'PlanCtrl_0'
            # ),
        # conditions == 'Rest'
        # )
    condition_mask = np.logical_or(
        np.logical_or(
            conditions == 'PlanCtrl_5', conditions == 'PlanCtrl_0'
            ),
        np.logical_or(
            conditions == 'ExeCtrl_5', conditions == 'ExeCtrl_0'
            )
        )
    X = fmri_data[..., condition_mask]
    y = y[condition_mask]

    from sklearn.preprocessing import binarize
    y = binarize(y, threshold=3.0)[0]

    # ### Masking step
    from pymri.utils import masking
    from nibabel import Nifti1Image

    # Mask data
    X_img = Nifti1Image(X, affine)
    X = masking.apply_mask(X_img, mask, smoothing_fwhm=4)

    from sklearn.feature_selection import SelectKBest, f_classif

    # ### Define the dimension reduction to be used.
    # Here we use a classical univariate feature selection based on F-test,
    # namely Anova. We set the number of features to be selected to 784
    feature_selection = SelectKBest(f_classif, k=784)

    feature_selection.fit(X, y)
    X = feature_selection.transform(X)

    from sklearn import preprocessing
    X = preprocessing.normalize(X)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    n_classes = 2
    y = convertToOneOfMany(y, n_classes)
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))

    return X, y

def load_nnadl_dataset(data_dir, sizes=(0.5, 0.25, 0.25)):
    """
    00. Load data from file (X and y).
    01. Split into training phase (train dataset) and validation phase.
    02. Split validation phase into valdation dataset and test dataset.

    Datasets default proportions:
    (train/validation/test) (0.5/0.25/0.25)
    """

    import numpy as np
    import nibabel
    from sklearn.datasets.base import Bunch

    data_dir = data_dir

    # create sklearn's Bunch of data
    dataset_files = Bunch(
        func=data_dir+'bold.nii.gz',
        session_target=data_dir+'attributes.txt',
        mask=data_dir+'mask.nii.gz',
        conditions_target=data_dir+'attributes_literal.txt'
        )

    # fmri_data and mask are copied to break reference to the original object
    bold_img = nibabel.load(dataset_files.func)
    fmri_data = bold_img.get_data().astype(float)
    affine = bold_img.get_affine()
    y, session = np.loadtxt(dataset_files.session_target).astype("int").T
    conditions = np.recfromtxt(dataset_files.conditions_target)['f0']
    mask = dataset_files.mask

    # ### Restrict to specified conditions

    # condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
    # condition_mask = np.logical_or(
        # np.logical_or(
            # conditions == 'PlanCtrl_5', conditions == 'PlanCtrl_0'
            # ),
        # conditions == 'Rest'
        # )
    condition_mask = np.logical_or(
        np.logical_or(
            conditions == 'PlanCtrl_5', conditions == 'PlanCtrl_0'
            ),
        np.logical_or(
            conditions == 'ExeCtrl_5', conditions == 'ExeCtrl_0'
            )
        )
    X = fmri_data[..., condition_mask]
    y = y[condition_mask]

    from sklearn.preprocessing import binarize
    y = binarize(y, threshold=3.0)[0]

    # ### Masking step
    from pymri.utils import masking
    from nibabel import Nifti1Image

    # Mask data
    X_img = Nifti1Image(X, affine)
    X = masking.apply_mask(X_img, mask, smoothing_fwhm=4)

    from sklearn.feature_selection import SelectKBest, f_classif

    # ### Define the dimension reduction to be used.
    # Here we use a classical univariate feature selection based on F-test,
    # namely Anova. We set the number of features to be selected to 784
    feature_selection = SelectKBest(f_classif, k=784)

    feature_selection.fit(X, y)
    X = feature_selection.transform(X)

    from sklearn import preprocessing
    X = preprocessing.normalize(X)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    n_classes = 2
    y = convertToOneOfMany(y, n_classes)
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))

    # ### Splitting ###########################################################
    from sklearn.cross_validation import train_test_split


    # X, y - training dataset
    # X_v, y_v - validation dataset
    # X_t, y_t - test dataset

    if len(sizes) == 3:
        train_size, valid_size, test_size = sizes

        # split original dataset into training phase (dataset) and
        # validation phase
        X, X_v, y, y_v = train_test_split(
            X, y, train_size=train_size
            )

        # split validation phase into validation dataset and test dataset
        X_v, X_t, y_v, y_t = train_test_split(
            X_v, y_v, test_size=test_size*2
            )

        return zip(X, y), zip(X_v, y_v), zip(X_t, y_t)

    elif len(sizes) == 2:

        train_size, test_size = sizes

        # split original dataset into training phase (dataset) and
        # validation phase
        X, X_t, y, y_t = train_test_split(
            X, y, train_size=train_size
            )

        return zip(X, y), (['no_validation_set']), zip(X_t, y_t)



def convertToOneOfMany(target, n_classes):
    print(target)
    y = np.zeros(shape=(target.shape[0], n_classes))

    for sample in range(target.shape[0]):
        y[sample][target[sample]] = 1
    print(y)
    return y
