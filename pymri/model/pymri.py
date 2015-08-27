#!/usr/bin/env python

# TODO: submodules import (standard modules)
# import get_data_raw
import get_data


class Model(object):
    def __init__(self):
        self.directory_data_nifti = None
        self.directory_paradigm = None
        self.tr = None

        self.paradigm_seconds = None
        self.paradigm_volumens = None

        self.data_raw = None
        self.data_raw_category = None
        self.data_feature_extracted = None
        self.data_feature_selected = None

    '''
        transform data from nifti format to np.array using nibabel
    '''
    def nifti_to_array(self, nifti_files_list, tr, directory_paradigm):

        self.directory_paradigm = directory_paradigm
        self.tr = tr
        self.nifti_files_list = nifti_files_list

        paradigm_seconds = get_data.get_paradigm(directory_paradigm)

        paradigm_volumens = get_data.paradigm_sec_to_vol(
            paradigm_seconds,
            tr
            )

        self.data_raw, self.data_raw_category = get_data.get_raw_data(
            nifti_files_list,
            paradigm_volumens
            )

        self.paradigm_seconds = paradigm_seconds
        self.paradigm_volumens = paradigm_volumens

        return self.data_raw, self.data_raw_category

    def feature_extraction():
        pass

    def split_data():
        pass

    def feature_selection():
        pass
