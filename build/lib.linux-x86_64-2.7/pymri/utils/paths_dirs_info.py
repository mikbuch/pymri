import os


def get_subject_names(base_directory, subject_template):
    subjects_list = os.listdir(base_directory)

    subjects_list = [sub for sub in subjects_list if subject_template in sub]

    return subjects_list


def get_roi_standard(base_directory):
    roi_standard = os.listdir(base_directory)
    roi_standard = [base_directory + '/' + roi for roi in roi_standard]
    return roi_standard
