"""
Theano dataset generation (transformation)
"""

__author__ = "Mikolaj Buchwald"
__contact__ = "mikolaj.buchwald@gmail.com"


def load_theano_dataset(data_dir):
    """
    00. Load data from file (X and y).
    01. Split into training phase (train dataset) and validation phase.
    02. Split validation phase into valdation dataset and test dataset.

    Datasets proportions:
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
    condition_mask = np.logical_or(
        np.logical_or(
            conditions == 'ExeCtrl_5', conditions == 'ExeCtrl_0'
            ), conditions == 'Rest'
        )
    X = fmri_data[..., condition_mask]
    y = y[condition_mask]

    from sklearn.preprocessing import binarize
    y = binarize(y, threshold=2.0)[0]

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
    print(X.shape)

    # ### Splitting ###########################################################
    from sklearn.cross_validation import train_test_split

    # split original dataset into training phase (dataset) and validation phase
    X, X_v, y, y_v = train_test_split(
        X, y, test_size=0.5, random_state=42
        )

    # split validation phase into validation dataset and test dataset
    X_v, X_t, y_v, y_t = train_test_split(
        X_v, y_v, test_size=0.25, random_state=42
        )

    # X, y - training dataset
    # X_v, y_v - validation dataset
    # X_t, y_t - test dataset

    return (X, y), (X_v, y_v), (X_t, y_t)
