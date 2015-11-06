"""
neural-networks-and-deep-learning dataset generation (transformation)
"""

__author__ = "Mikolaj Buchwald"
__contact__ = "mikolaj.buchwald@gmail.com"


import numpy as np


def load_nifti(data_dir, Y, k_features=784, normalize=True, scale_0_1=False):
    '''
    Parameters
    ----------
    data_dir : string.
        Location of the data files (bold.nii.gz, attributes.txt,
        attributes_literal.txt).

    Y : ndtuple of strings.
        Classes. Label space will be reduced to specified conditions.


    Returns
    -------
    X : ndnumpy array.
        Samples containing features.

    y : ndnumpy array.
        Labels, targets, classes.


    conditions:
    Y = {Y_1, Y_2, ... , Y_n}
    Y_n = S = {S_1, S_2, .... , S_k}

    where:
        Y - set of classes
        S - set of subclasses
        n - number of classes
        k - number of subclasses

    classes consists of subclasses

    '''
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

    condition_mask = np.zeros(shape=(conditions.shape[0]))

    for n in Y:
        if type(n) == tuple:
            # first label
            k_uniform = n[0]
            k_uniform = y[conditions == k_uniform][0]
            for k in n:
                condition_mask += conditions == k
                # unifying subclasses into one class
                # (string label doesn't matter)
                y[conditions == k] = k_uniform
        else:
            condition_mask += conditions == n

    condition_mask = np.array(condition_mask, dtype=bool)
    X = fmri_data[..., condition_mask]
    y = y[condition_mask]

    cnt = 0
    for val in np.unique(y):
        if val > cnt:
            y[y == val] = cnt
        cnt += 1

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
    feature_selection = SelectKBest(f_classif, k=k_features)

    feature_selection.fit(X, y)
    X = feature_selection.transform(X)

    # normalize data
    if normalize:
        from sklearn import preprocessing
        X = preprocessing.normalize(X)

    if scale_0_1:
        # scale data in range (0,1)
        X = (X - X.min()) / (X.max() - X.min())

    # reshape is needed for nnadl acceptable format
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # how many classes do we have?
    n_classes = len(Y)
    # [0, 1, 1] ==> [[1, 0], [0, 1], [0, 1]]
    y = convertToOneOfMany(y, n_classes)
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))

    return X, y 

def load_nnadl_dataset(
    data_dir, Y, k_features, normalize=True, scale_0_1=False, sizes=(0.5, 0.25, 0.25)
    ):
    """

    Datasets default proportions:
    (train/validation/test) (0.5/0.25/0.25)
    """

    X, y = load_nifti(data_dir, Y, k_features, normalize, scale_0_1)

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
