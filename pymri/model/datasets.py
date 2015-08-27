from __future__ import print_function
import os
import csv
import nibabel as nib
import numpy as np
import math
import psutil


class RawDataset(object):
    def __init__(self, dir_paradigm=None, nifti_locations=None, tr=None):
        self.dir_paradigm = dir_paradigm
        self.nifti_locations = nifti_locations
        self.nifti_list = None
        self.tr = tr

        self.paradigm_seconds = None
        self.paradigm_volumens = None

        self.data_raw = None
        self.data_raw_category = None

    # read first cell of the txt file (to get info when condition begins)
    def read_first_cell(self, condition_absolute_path, delimiter='\t'):
        with open(condition_absolute_path, 'rb') as f:
            mycsv = csv.reader(f, delimiter=delimiter)
            return list(mycsv)[0][0]

    def get_paradigm(self):

        subdirectories = os.listdir(self.dir_paradigm)
        subdirectories.sort()

        conditions = os.listdir(self.dir_paradigm + '/' + subdirectories[0])
        conditions.sort()

        print(*subdirectories, sep='\n')
        print(*conditions, sep='\n')

        paradigm = []

        for subdir in subdirectories:
            paradigm.append([])
            for cond in conditions:
                paradigm[-1].append(
                    self.read_first_cell(
                        self.dir_paradigm + '/' + subdir + '/' + cond
                        )
                    )
                print(self.dir_paradigm + '/' + subdir + '/' + cond)

        return paradigm

    def paradigm_sec_to_vol(self):
        self.paradigm_seconds = self.get_paradigm()

        paradigm_volumens = []
        for task in self.paradigm_seconds:
            paradigm_volumens.append([])
            for cond in task:
                paradigm_volumens[-1].append(math.ceil(float(cond)/self.tr))

        return paradigm_volumens

    def get_raw_data(self):

        self.paradigm_volumens = self.paradigm_sec_to_vol()

        self.nifti_list = \
            [line.rstrip('\n') for line in open(self.nifti_locations)]

        # transform to array one nifti file to get data shape
        img_tmp = nib.load(self.nifti_list[0])
        single_img_shape = img_tmp.shape[::-1]
        data_all_shape = single_img_shape
        data_all_shape = data_all_shape[:0] + \
            (len(self.nifti_list),) + \
            data_all_shape[0:]
        print(data_all_shape)

        data_all_classes = np.zeros(data_all_shape)

        # create 5D array containing all runs
        for run in range(len(self.nifti_list)):
            img_tmp = nib.load(self.nifti_list[run])
            data_all_classes[run] = img_tmp.get_data().T

        # get only two classes from whole dataset (based on paradigm_volumens)
        # only 8 volumes per class per run are valid
        # that gives 16 volumens of 2 classes per run and 192 volumens per sub
        samples_num = len(data_all_classes)*2*8
        data_two_shape = single_img_shape[:0] + \
            (samples_num,) + \
            single_img_shape[1:]

        process = psutil.Process(os.getpid())
        print(process.get_memory_info()[0] / float(2 ** 20))
        print(data_two_shape)

        data_two_classes = np.zeros(data_two_shape)
        print(process.get_memory_info()[0] / float(2 ** 20))
        data_two_classes_category = np.zeros(samples_num)

        # TODO: loop taking data only from two categories (houses and faces)

        sample_cnt = 0

        for run in range(len(data_all_classes)):
            for img in (range(len(data_all_classes[run]))):
                if img >= self.paradigm_volumens[run][0] and \
                        img < self.paradigm_volumens[run][0]+8:
                            data_two_classes[sample_cnt] = \
                                data_all_classes[run][img]
                            data_two_classes_category[sample_cnt] = 0
                            sample_cnt += 1
                            # print(paradigm_volumens[run][0])
                if img >= self.paradigm_volumens[run][-1] and \
                        img < self.paradigm_volumens[run][-1]+8:
                            data_two_classes[sample_cnt] = \
                                data_all_classes[run][img]
                            data_two_classes_category[sample_cnt] = 1
                            sample_cnt += 1
                            # print(paradigm_volumens[run][-1])

        return data_two_classes, data_two_classes_category

    '''
        transform data from nifti format to np.array using nibabel
    '''
    def nifti_to_array(self):

        self.data_raw, self.data_raw_category = self.get_raw_data()

        # return self.data_raw, self.data_raw_category
