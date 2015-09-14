from __future__ import print_function
import os
import csv
import nibabel as nib
import numpy as np
import math
import psutil
from ram.ram_usage_proc import usage_print


class RawDataset(object):
    def __init__(self, dir_paradigm=None, nifti_locations=None, tr=None):
        self.tr = tr
        self.dir_paradigm = dir_paradigm
        if self.dir_paradigm is not None:
            self.paradigm_volumens = self.paradigm_sec_to_vol()
        else:
            self.paradigm_volumens = None

        self.nifti_locations = nifti_locations
        if nifti_locations is not None:
            self.nifti_list = \
                [line.rstrip('\n') for line in open(self.nifti_locations)]
        else:
            self.nifti_list = None

        self.paradigm_seconds = None

        self.single_run_shape = None
        self.single_vol_shape = None
        self.masked_vol_shape = None
        self.raw_data = None
        self.raw_data_category = None

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

        # print(*subdirectories, sep='\n')
        # print(*conditions, sep='\n')

        paradigm = []

        for subdir in subdirectories:
            paradigm.append([])
            for cond in conditions:
                paradigm[-1].append(
                    self.read_first_cell(
                        self.dir_paradigm + '/' + subdir + '/' + cond
                        )
                    )

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

        # transform to array one nifti file to get data shape
        img_tmp = nib.load(self.nifti_list[0])
        self.single_run_shape = img_tmp.shape[::-1]
        self.single_vol_shape = self.single_run_shape[1:]

        # how many samples will we get (number_of_runs*nb_classes*volumens)
        samples_num = len(self.nifti_list)*2*8

        usage_print()
        data_two_shape = self.single_run_shape[:0] + \
            (samples_num,) + \
            self.single_run_shape[1:]

        usage_print()
        data_two_classes = np.zeros(data_two_shape)
        data_two_classes_category = np.zeros(samples_num)

        sample_cnt = 0

        # create 5D array containing all runs
        for run in range(len(self.nifti_list)):
            print('run: %d/%d' % (run+1, len(self.nifti_list)))
            print('sample_counter: %d' % (sample_cnt))
            img_tmp = nib.load(self.nifti_list[run])
            img_tmp = img_tmp.get_data().T
            usage_print()

            # get only two classes from whole dataset (based on paradigm_volumens)
            # only 8 volumes per class per run are valid
            # that gives 16 volumens of 2 classes per run and 192 volumens per sub

            # TODO: loop taking data only from two categories (houses and faces)


            for img in (range(len(img_tmp))):
                # print('    img: %d/%d' % (img+1, len(img_tmp)))
                if img >= self.paradigm_volumens[run][0] and \
                        img < self.paradigm_volumens[run][0]+8:
                            data_two_classes[sample_cnt] = \
                                img_tmp[img]
                            data_two_classes_category[sample_cnt] = 0
                            sample_cnt += 1
                            # print(paradigm_volumens[run][0])
                if img >= self.paradigm_volumens[run][-1] and \
                        img < self.paradigm_volumens[run][-1]+8:
                            data_two_classes[sample_cnt] = \
                                img_tmp[img]
                            data_two_classes_category[sample_cnt] = 1
                            sample_cnt += 1
                            # print(paradigm_volumens[run][-1])

        return data_two_classes, data_two_classes_category
    
    def get_raw_data_masked(self, mask_nifti):

        mask = nib.load(mask_nifti).get_data()
        mask = mask.astype(bool)

        self.masked_vol_shape = np.sum(mask)

        # how many samples will we get (number_of_runs*nb_classes*volumens)
        samples_num = len(self.nifti_list)*2*8

        data_two_shape = (samples_num,) + (self.masked_vol_shape,)

        usage_print()
        data_two_classes = np.zeros(data_two_shape)
        data_two_classes_category = np.zeros(samples_num)
        usage_print()

        sample_cnt = 0

        # create 2D array containing all runs
        for run in range(len(self.nifti_list)):
            print('run: %d/%d' % (run+1, len(self.nifti_list)))
            print('sample_counter: %d' % (sample_cnt))
            func_data = nib.load(self.nifti_list[run]).get_data()
            X = func_data[mask].T
            usage_print()

            # 0 is houses, 7 is faces
            for i in (0, 7):
                vol_start = self.paradigm_volumens[run][i]
                data_two_classes[sample_cnt:sample_cnt+8] = \
                    X[vol_start:vol_start+8]
                if i== 0:
                    class_bin = 0
                else:
                    class_bin = 1
                data_two_classes_category[sample_cnt:sample_cnt+8] = \
                    class_bin
                print('%d:%d' % (sample_cnt, sample_cnt+8))
                sample_cnt += 8

        return data_two_classes, data_two_classes_category

    '''
        transform data from nifti format to np.array using nibabel
    '''
    def nifti_to_array_4D(self):

        self.raw_data, self.raw_data_category = self.get_raw_data()

        # return self.raw_data, self.raw_data_category

    def nifti_to_array_masked(self, nifti_mask):

        self.raw_data, self.raw_data_category = self.get_raw_data_masked(nifti_mask)

        # return self.raw_data, self.raw_data_category
