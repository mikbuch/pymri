"""
neural-networks-and-deep-learning dataset generation (transformation)
"""

__author__ = "Mikolaj Buchwald"
__contact__ = "mikolaj.buchwald@gmail.com"


import numpy as np
import nibabel
from sklearn.datasets.base import Bunch
# from utils import masking, signal
from pymri.utils import masking
from nibabel import Nifti1Image


class DatasetManager(object):

    def __init__(
            self,
            contrast,
            path_bold='./bold.nii.gz',
            path_attr='./attributes.txt',
            path_attr_lit='./attributes_literal.txt',
            path_mask_brain='./mask.nii.gz',
            path_output='./',
            nnadl=False,
            scale_0_1=False,
            vectorize_target=False,
            ):

        self.contrast = contrast
        self.path_bold = path_bold
        self.path_attr = path_attr
        self.path_attr_lit = path_attr_lit
        self.path_mask_brain = path_mask_brain
        self.path_output = path_output
        self.k_features = None
        self.normalize = None
        self.scale_0_1 = scale_0_1
        self.nnadl = nnadl
        self.vectorize_target = vectorize_target
        self.sizes = None

        self.feature_reduction_method = None
        self.mask_non_brain = None

        self.training_data_max = None
        self.training_data_min = None

        self.X_raw = None
        self.X_processed = None
        self.y = None

    def load_data(self):

        # create sklearn's Bunch of data
        dataset_files = Bunch(
            func=self.path_bold,
            session_target=self.path_attr,
            mask=self.path_mask_brain,
            conditions_target=self.path_attr_lit
            )

        # fmri_data and mask are copied to break reference to
        # the original object
        bold_img = nibabel.load(dataset_files.func)
        fmri_data = bold_img.get_data().astype(float)
        affine = bold_img.get_affine()
        y, session = np.loadtxt(dataset_files.session_target).astype("int").T
        conditions = np.recfromtxt(dataset_files.conditions_target)['f0']
        mask = dataset_files.mask
        self.mask_non_brain = mask

        # ### Restrict to specified conditions
        condition_mask = np.zeros(shape=(conditions.shape[0]))

        # unifrom contrasts, e.g.: contrast=(('face', 'table'), 'house')):
        # face:0, table:1, house:2
        # [0, 0, 2, 1, 2, 0, 1, 1] ==> [0, 0, 2, 0, 2, 0, 0, 0]
        for n in self.contrast:
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

        # adjust all (target) labels to  range(0:n_classes range),
        # e.g. [4,4,2,8,9] ==> [1,1,0,2,3]
        # e.g. [0, 0, 2, 0, 2, 0, 0, 0] ==> [0, 0, 1, 0, 1, 0, 0, 0]
        cnt = 0
        for val in np.unique(y):
            if val > cnt:
                y[y == val] = cnt
            cnt += 1

        # ### Masking step

        # Mask data using brain mask (remove non-brain regions)
        X_img = Nifti1Image(X, affine)
        X = masking.apply_mask(X_img, mask, smoothing_fwhm=4)
        # X = signal.clean(X, standardize=True, detrend=False)

        print('##############################')
        print('# dataset loaded successfully ')
        print('# X shape: %s' % str(X.shape))
        print('# y shape: %s' % str(y.shape))
        print('##############################')
        self.X_raw, self.y = X, y

    def feature_reduction(self, roi_selection, k_features=784, normalize=True):
        self.k_features = k_features
        self.normalize = normalize

        if roi_selection is None:
            print('ROI selection method has to be specified'),
            print('Available values are: \'SelectKBest\', \'PCA\', \'RBM\''),
            print('or \'path_to_roi_mask\'\n')

        # Array being processed will be loaded raw data
        X = self.X_raw
        # target
        y = self.y

        # ROI selection (feature reduction) method to be applied
        if roi_selection == 'SelectKBest':
            X = self._SelectKBest(X, y)
        elif roi_selection == 'PCA':
            X = self._PCA(X, y)
        elif roi_selection == 'RBM':
            X = self._RBM(X, y)
        else:
            X = self._roi_mask_apply(X, roi_selection)
            self.k_features = X.shape[1]

        # normalize if set
        if self.normalize:
            X = self._normalize(X)

        # scale in range(0,1) if set
        if self.scale_0_1:
            X = self._scale_0_1(X)

        self.X_processed = X

        if self.nnadl:
            self.nnadl_prep()
        else:
            if self.vectorize_target:
                # how many classes do we have?
                n_classes = len(self.contrast)
                # vectorize target (labels), aka ConvertToOneOfMany
                # e.g. [0, 1, 1] ==> [[1, 0], [0, 1], [0, 1]]
                self.y = self.vectorize(self.y, n_classes)

    def _normalize(self, X):
        from sklearn import preprocessing
        X = preprocessing.normalize(X)
        return X

    def _scale_0_1(self, X):
        # scale data in range (0,1)
        X = (X - X.min()) / (X.max() - X.min())
        return X

    def _SelectKBest(self, X, y):
        from sklearn.feature_selection import SelectKBest, f_classif

        # ### Define the dimension reduction to be used.
        # Here we use a classical univariate feature selection based on F-test,
        # namely Anova. The number of features to be selected is set to 784
        feature_selection = SelectKBest(f_classif, k=self.k_features)

        feature_selection.fit(X, y)
        X = feature_selection.transform(X)

        self.feature_reduction_method = feature_selection

        return X

    def _PCA(self, X, y):

        if X.shape[0] >= self.k_features:
            print('PCA fits only when number of features to be extracted')
            print('is less than number of samples.')

        from sklearn.decomposition import PCA

        # PCA model creation, number of components
        # feature extraction method. Used here (after sampling) because we are
        # creating an universal model and not this_dataset-specific.
        feature_extraction = PCA(n_components=self.k_features)

        feature_extraction.fit(X, y)
        X = feature_extraction.transform(X)

        self.feature_reduction_method = feature_extraction

        return X

    def _RBM(self, X, y):

        from sklearn.neural_network import BernoulliRBM

        # PCA model creation, number of components
        # feature extraction method. Used here (after sampling) because we are
        # creating an universal model and not this_dataset-specific.
        neural_network = BernoulliRBM(n_components=self.k_features)

        neural_network.fit(X, y)
        X = neural_network.transform(X)

        self.feature_reduction_method = neural_network

        return X

    # TODO: finish this one
    def _roi_mask_apply(self, X, roi_mask_img):

        # roi_mask_img = nibabel.load(roi_mask_img)

        roi_mask = masking.apply_mask(roi_mask_img, self.mask_non_brain)
        roi_mask = roi_mask.astype(bool)

        X = X[..., roi_mask]

        self.feature_reduction_method = roi_mask

        return X

    def split_data(self, sizes=(0.75, 0.25)):

        self.sizes = sizes

        # ### Splitting #######################################################
        from sklearn.cross_validation import train_test_split

        # X, y - training dataset
        # X_v, y_v - validation dataset
        # X_t, y_t - test dataset

        if len(self.sizes) == 3:
            train_size, valid_size, test_size = self.sizes

            # split original dataset into training phase (dataset) and
            # validation phase
            X, X_v, y, y_v = train_test_split(
                self.X_processed, self.y, train_size=train_size
                )

            # split validation phase into validation dataset and test dataset
            X_v, X_t, y_v, y_t = train_test_split(
                X_v, y_v, test_size=test_size*2
                )

            self.training_data_max = X.max()
            self.training_data_min = X.min()

            if self.nnadl:
                return zip(X, y), zip(X_v, y_v), zip(X_t, y_t)
            else:
                return (X, y), (X_v, y_v), (X_t, y_t)

        elif len(self.sizes) == 2:

            train_size, test_size = self.sizes

            # split original dataset into training phase (dataset) and
            # validation phase
            X, X_t, y, y_t = train_test_split(
                self.X_processed, self.y, train_size=train_size
                )

            self.training_data_max = X.max()
            self.training_data_min = X.min()

            if self.nnadl:
                return zip(X, y), zip(X_t, y_t), None
            else:
                return (X, y), (X_t, y_t), None

    def nnadl_prep(self):
        ''' Neural networks and deep learning tutorial data preparation
        '''

        X = self.X_processed
        y = self.y

        # how many classes do we have?
        n_classes = len(self.contrast)
        # [0, 1, 1] ==> [[1, 0], [0, 1], [0, 1]]
        y = self.vectorize(y, n_classes)

        # reshape is needed for nnadl acceptable format
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        y = np.reshape(y, (y.shape[0], y.shape[1], 1))

        self.X_processed = X
        self.y = y

    def vectorize(self, target, n_classes):
        y = np.zeros(shape=(target.shape[0], n_classes))

        for sample in range(target.shape[0]):
            y[sample][target[sample]] = 1
        return y

    def save_as_nifti(self, activation, filename):
        # care about the file extension
        if filename.endswith('.nii'):
            filename += '.gz'
        elif filename.endswith('nii.gz'):
            pass
        else:
            filename += '.nii.gz'

        # reverse feature reduction operation
        if type(self.feature_reduction_method) == str:
            activation = masking.unmask(self.feature_reduction_method)
        else:
            activation = self.feature_reduction_method.inverse_transform(
                activation
                )

        # reverse non-brain regions masking
        activation = masking.unmask(activation, self.mask_non_brain)

        img = nibabel.Nifti1Image(activation, np.eye(4))
        img.to_filename(filename)
