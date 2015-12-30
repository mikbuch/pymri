"""
Generate Dataset for PyMRI
"""

__author__ = "Mikolaj Buchwald"
__contact__ = "mikolaj.buchwald@gmail.com"


import numpy as np
import os
import nibabel
from sklearn.datasets.base import Bunch
# from utils import masking, signal
from pymri.utils import masking
from nibabel import Nifti1Image


class DatasetManager(object):

    def __init__(
            self,
            contrast,
            mvpa_directory='./',
            bold='bold.nii.gz',
            attr='attributes.txt',
            attr_lit='attributes_literal.txt',
            mask_brain='mask.nii.gz',
            path_output='./',
            scale_0_1=False,
            vectorize_target=False,
            ):

        bold_path = os.path.join(mvpa_directory, bold)
        attr_path = os.path.join(mvpa_directory, attr)
        attr_lit_path = os.path.join(mvpa_directory, attr_lit)
        mask_brain_path = os.path.join(mvpa_directory, mask_brain)

        self.contrast = contrast
        self.bold = bold_path
        self.attr = attr_path
        self.attr_lit = attr_lit_path
        self.mask_brain = mask_brain_path
        self.output = path_output
        self.k_features = None
        self.reduction_method = None
        self.normalize = None
        self.scale_0_1 = scale_0_1
        self.nnadl = False
        self.vectorize_target = vectorize_target
        self.sizes = None

        self.feature_reduction_method = None
        self.mask_non_brain = None

        self.training_data_max = None
        self.training_data_min = None

        self.condition_mask = None
        self.X_raw = None
        self.X_processed = None
        self.y = None
        self.y_processed = None

        self.load_data()

    def load_data(self):

        # create sklearn's Bunch of data
        dataset_files = Bunch(
            func=self.bold,
            session_target=self.attr,
            mask=self.mask_brain,
            conditions_target=self.attr_lit
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
        self.condition_mask = condition_mask
        self.X_raw, self.y = X, y

    def feature_reduction(
            self, roi_path=None, k_features=0, reduction_method=None,
            normalize=False, nnadl=False, feature_arguments=None
            ):
        self.k_features = k_features
        self.reduction_method = reduction_method
        self.normalize = normalize
        self.nnadl = nnadl

        # Array being processed will be loaded raw data
        X = self.X_raw
        # target
        y = self.y

        if roi_path and roi_path != '':
            if reduction_method:
                if 'mask' in reduction_method:
                    X = self._SelectKHighest_from_mask(X, y, roi_path)
            else:
                X = self._roi_mask_apply(X, roi_path)

        if reduction_method:
            # ROI selection (feature reduction) method to be applied
            if 'SelectKBest' in reduction_method or 'SKB' in reduction_method:
                X = self._SelectKBest(X, y)
            elif 'PCA' in reduction_method:
                X = self._PCA(X, y, feature_arguments)
            elif 'RBM' in reduction_method:
                X = self._RBM(X, y, feature_arguments)

        self.X_processed = X


    def _normalize(self, X, y, X_t):
        from sklearn.preprocessing import Normalizer
        NORM = Normalizer()

        X = NORM.fit_transform(X, y)
        X_t = NORM.transform(X_t)

        return X, X_t

    def _scale_0_1(self, X):
        # scale data in range (0,1)
        X = (X - X.min()) / (X.max() - X.min())
        return X

    def _SelectKBest(self, X, y):

        print('Selecting K Best from whole image')

        from sklearn.feature_selection import SelectKBest, f_classif

        # ### Define the dimension reduction to be used.
        # Here we use a classical univariate feature selection based on F-test,
        # namely Anova. The number of features to be selected is set to 784
        feature_selection = SelectKBest(f_classif, k=self.k_features)

        feature_selection.fit(X, y)
        print('SelectKBest data reduction from: %s' % str(X.shape))
        X = feature_selection.transform(X)
        print('SelectKBest data reduction to: %s' % str(X.shape))

        self.feature_reduction_method = feature_selection

        return X

    def _SelectKHighest_from_mask(self, X, y, roi_path):

        print('Selecting K Highest from mask: %s' % roi_path)

        '''
        roi_mask = masking.apply_mask(
            roi_path, self.mask_non_brain, k_features=self.k_features
            )
        '''

        roi_mask = masking.apply_mask(roi_path, self.mask_non_brain)

        print('SelectKHighest ROI mask size: %d' % roi_mask.sum())

        from pymri.utils.masking import separate_k_highest

        roi_mask = separate_k_highest(self.k_features, roi_mask)
        roi_mask = roi_mask.astype(bool)

        print('SelectKHighest data reduction from: %s' % str(X.shape))
        X = X[..., roi_mask]
        print('SelectKHighest data reduction to: %s' % str(X.shape))

        self.feature_reduction_method = roi_path

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

    def _roi_mask_apply(self, X, roi_path):

        print('Applying mask: %s' % roi_path)

        roi_mask = masking.apply_mask(roi_path, self.mask_non_brain)
        roi_mask = roi_mask.astype(bool)

        print('ROI mask apply ROI mask size: %s' % str(roi_mask.sum()))

        X = X[..., roi_mask]

        self.feature_reduction_method = roi_path

        print('Masked data has shape: %s' % str(X.shape))

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

            if self.normalize:
                X, X_t = self._normalize(X, y, X_t)

            if self.nnadl:
                X, y = self.nnadl_prep(X, y) 
                X_t, y_t = self.nnadl_prep(X_t, y_t) 
            elif self.vectorize_target:
                # how many classes do we have?
                n_classes = len(self.contrast)
                # vectorize target (labels), aka ConvertToOneOfMany
                # e.g. [0, 1, 1] ==> [[1, 0], [0, 1], [0, 1]]
                y = self.vectorize(self.y, n_classes)
            else:
                pass

            if self.nnadl:
                return zip(X, y), zip(X_t, y_t), None
            else:
                return (X, y), (X_t, y_t), None


    def leave_one_run_out(self, runs, volumes, n_time):

        one_run = np.zeros(shape=(runs*volumes), dtype=bool)
        one_run[n_time*volumes:(n_time+1)*volumes] = 1
        rest_runs = -one_run

        one_run = one_run[self.condition_mask]
        rest_runs = rest_runs[self.condition_mask]

        # X, y - training dataset
        # X_t, y_t - test dataset

        # get from the original dataset training phase consisting of
        # 4 functional runs
        X = self.X_processed[rest_runs]
        y = self.y[rest_runs]

        # get from the original dataset testing phase consisting of
        # 1 functional run
        X_t = self.X_processed[one_run]
        y_t = self.y[one_run]

        if self.normalize:
            X, X_t = self._normalize(X, y, X_t)

        if self.nnadl:
            X, y = self.nnadl_prep(X, y) 
            X_t, y_t = self.nnadl_prep(X_t, y_t) 
        elif self.vectorize_target:
            # how many classes do we have?
            n_classes = len(self.contrast)
            # vectorize target (labels), aka ConvertToOneOfMany
            # e.g. [0, 1, 1] ==> [[1, 0], [0, 1], [0, 1]]
            y = self.vectorize(self.y, n_classes)
        else:
            pass

        self.training_data_max = X.max()
        self.training_data_min = X.min()

        if self.nnadl:
            return zip(X, y), zip(X_t, y_t), None
        else:
            return (X, y), (X_t, y_t), None


    def nnadl_prep(self, X, y):
        ''' Neural networks and deep learning tutorial data preparation
        '''

        # how many classes do we have?
        n_classes = len(self.contrast)
        # [0, 1, 1] ==> [[1, 0], [0, 1], [0, 1]]
        y = self.vectorize(y, n_classes)

        # reshape is needed for nnadl acceptable format
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        y = np.reshape(y, (y.shape[0], y.shape[1], 1))

        return X, y

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
