#!/usr/bin/env python

'''
    name:
        get_data.py
----------
    
    description:
        This module allows to get data from nifti files to numpy array data
        format. Then it creates samples of data with assigned class labels.
        Part of pymri package.
'''

from __future__ import print_function
import os
import csv
import nibabel as nib
import numpy as np
import math
import psutil


# read first cell of the txt file (to get info when condition begins)
def read_first_cell(condition_absolute_path, delimiter='\t'):
    with open(condition_absolute_path, 'rb') as f:
        mycsv = csv.reader(f, delimiter=delimiter)
        return list(mycsv)[0][0]


def get_paradigm(directory_paradigm):

    absolute_path = directory_paradigm

    subdirectories = os.listdir(absolute_path)
    subdirectories.sort()

    conditions = os.listdir(absolute_path + '/' + subdirectories[0])
    conditions.sort()

    print(*subdirectories, sep='\n')
    print(*conditions, sep='\n')

    paradigm = []

    for subdir in subdirectories:
        paradigm.append([])
        for cond in conditions:
            paradigm[-1].append(
                read_first_cell(absolute_path + '/' + subdir + '/' + cond)
                )
            print(absolute_path + '/' + subdir + '/' + cond)

    return paradigm


def paradigm_sec_to_vol(paradigm_seconds, tr):
    paradigm_volumens = []
    for task in paradigm_seconds:
        paradigm_volumens.append([])
        for cond in task:
            paradigm_volumens[-1].append(math.ceil(float(cond)/tr))

    return paradigm_volumens


def tuple_insert(tup, pos, ele):
    tup = tup[:pos]+(ele,)+tup[pos:]
    return tup


def get_raw_data(nifti_files_list, paradigm_volumens):
    nifti_files = [line.rstrip('\n') for line in open(nifti_files_list)]

    # transform to array one nifti file to get data shape
    img_tmp = nib.load(nifti_files[0])
    single_img_shape = img_tmp.shape[::-1]
    data_all_shape = single_img_shape
    data_all_shape = data_all_shape[:0]+(len(nifti_files),)+data_all_shape[0:]
    # data_all_shape = tuple_insert(data_all_shape, 0, len(nifti_files))
    print(data_all_shape)

    data_all_classes = np.zeros(data_all_shape)

    # create 5D array containing all runs
    for run in range(len(nifti_files)):
        img_tmp = nib.load(nifti_files[run])
        data_all_classes[run] = img_tmp.get_data().T

    # get only two classes from whole dataset (based on paradigm_volumens)
    # only 8 volumes per class per run are valid
    # that gives 16 volumens of 2 classes per run and 192 volumens per subject
    samples_num = len(data_all_classes)*2*8
    data_two_shape = single_img_shape[:0]+(samples_num,)+single_img_shape[1:]

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
            if img >= paradigm_volumens[run][0] and \
                    img < paradigm_volumens[run][0]+8:
                        data_two_classes[sample_cnt] = \
                            data_all_classes[run][img]
                        data_two_classes_category[sample_cnt] = 0
                        sample_cnt += 1
                        # print(paradigm_volumens[run][0])
            if img >= paradigm_volumens[run][-1] and \
                    img < paradigm_volumens[run][-1]+8:
                        data_two_classes[sample_cnt] = \
                            data_all_classes[run][img]
                        data_two_classes_category[sample_cnt] = 1
                        sample_cnt += 1
                        # print(paradigm_volumens[run][-1])

    return data_two_classes, data_two_classes_category
