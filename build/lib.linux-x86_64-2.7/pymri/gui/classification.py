import Tkinter as tk
import tkFont
import tkFileDialog

import ttk
import os


# TODO: entire perform() function in subprocess to allow window to refresh
# TODO: create separate classes for each section to keep it clear

def classifier_choose(classifier_type):
    if 'FNN' in classifier_type:
        ANN_frame.grid(row=2, column=0, columnspan=10, padx=20, pady=10)
        if 'simple' in classifier_type:
            var_nnadl.set(1)
        else:
            var_nnadl.set(0)
    else:
        ANN_frame.grid_forget()
        var_nnadl.set(0)
    if 'SVC' in classifier_type:
        SVC_frame.grid(row=2, column=0, columnspan=10, padx=20, pady=10)
    else:
        SVC_frame.grid_forget()


def perform_choose(var_perform_type):
    if 'LPO' in var_perform_type:
        LPO_frame.grid(row=2, column=0, columnspan=10, padx=20, pady=10)
        var_n_times.set(LPO_n_times.get())
    else:
        LPO_frame.grid_forget()
    if 'LORO' in var_perform_type:
        LORO_frame.grid(row=2, column=0, columnspan=10, padx=20, pady=10)
        var_n_times.set(LORO_runs.get())
    else:
        LORO_frame.grid_forget()


def help_display(number):
    print number


def askfile(var_file):
    file_chosen = tkFileDialog.askopenfilename()
    var_file.set(file_chosen)


def askdir(var_dir):
    dir_chosen = tkFileDialog.askdirectory()
    var_dir.set(dir_chosen)


def check_frame(frame, variable, check, text, row=0, column=1, columnspan=10):
    if variable.get() == 1:
        frame.grid(
            row=row, column=column, columnspan=columnspan, sticky='W',
            padx=10, pady=10, ipadx=10, ipady=10
            )
        check['text'] = ""
    else:
        frame.grid_forget()
        check['text'] = text


def load_data(mvpa_directory):

    from pymri.dataset.datasets import DatasetManager

    print('Loading database from %s' % mvpa_directory)
    dataset = DatasetManager(
        mvpa_directory=mvpa_directory,
        # conditions has to be tuples
        contrast=(
            tuple(var_class_00.get().split(' ')),
            tuple(var_class_01.get().split(' '))
            )
        )
    return dataset


def create_classifier():
    '''
    For some classifiers testing phase is joined with training for convinence.
    '''

    if 'FNN' in var_classifier_type.get():

        from pymri.model import FNN

        if 'simple' in var_classifier_type.get():
            type = 'FNN simple'

        if 'theano' in var_classifier_type.get():
            type = 'FNN theano'

        # all parameters - for net architecture as well as for its training
        # has to be specified here (for reason see Classifier class)
        cls = FNN(
            type=type,
            input_layer_size=var_k_features.get(),
            hidden_layer_size=ANN_hn.get(),
            output_layer_size=var_classes_number.get(),
            epochs=ANN_epochs.get(),
            mini_batch_size=ANN_ms.get(),
            learning_rate=ANN_eta.get()
            )

    elif 'SVC' in var_classifier_type.get():

        from pymri.model import SVC

        cls = SVC(
            C=SVC_C.get(),
            kernel=SVC_kernel.get(),
            gamma=SVC_gamma.get(),
            degree=SVC_degree.get()
            )

    elif 'CNN' in var_classifier_type.get():

        from pymri.model import CNN
        import math

        cls = CNN(
            type='CNN',
            input=int(math.sqrt(var_k_features.get())),
            receptive=5,
            hidden_conv_layer=True,
            output_layer_size=var_classes_number.get(),
            epochs=ANN_epochs.get(),
            mini_batch_size=ANN_ms.get(),
            learning_rate=ANN_eta.get()
            )

    return cls


def feature_reduction(dataset, roi_path=None):

    if var_k_features_frame.get():
        reduction_method = var_reduction_method.get()
    else:
        reduction_method = None

    dataset.feature_reduction(
        k_features=784,
        reduction_method='SelectKBest (SKB)',
        normalize=True,
        nnadl=True
        )

    return dataset


def split_data(dataset, n_time=None):

    if 'LPO' in var_perform_type.get():
        training_data, test_data, valid_data = dataset.split_data(
            sizes=(1-LPO_p.get(), LPO_p.get())
            )
    elif 'LORO' in var_perform_type.get():
        training_data, test_data, validation_data = dataset.leave_one_run_out(
            runs=LORO_runs.get(),
            volumes=LORO_volumes.get(),
            n_time=n_time
            )

    return training_data, test_data


def train_and_test_classifier(cls, training_data, test_data):
    '''
    Training and testing are somtimes bound with each other.
    '''
    cls.train_and_test(training_data, test_data)

    return cls


def get_accuracy(cls):
    return cls.get_accuracy()


def visualise_results(results):

    import numpy as np
    import matplotlib.pyplot as plt

    mean_accuracies = np.zeros(shape=(LPO_n_times.get(),))
    for i in range(LPO_n_times.get()):
        mean_accuracies[i] = results[:i+1].mean()

    # plot best accuracy for particular validation
    plt.scatter(
        range(len(results)), results,
        marker='o', s=120, c='r', label='accuracy'
        )
    # to show wether overall average changes with number of validations
    plt.plot(
        range(len(mean_accuracies)), mean_accuracies, label='mean accurcy'
        )
    plt.ylim(-0.1, 1.1)
    plt.legend(loc=3)
    plt.show()


def perform_classification():

    import numpy as np

    # subs_num = 21
    # hands_num = 2
    # rois_num = 2
    # n_times_num = 100

    # get list of subjects (subjects list specified or get from the pattern)
    if '*' in subjects_names.get() and \
            len(subjects_names.get().split(' ')) == 1:
        subjects_pattern = subjects_names.get()

        from pymri.utils.paths_dirs_info import get_subject_names
        subjects_list = get_subject_names(
            var_base_dir.get(), subjects_pattern.replace('*', '')
            )
    else:
        subjects_list = subjects_names.get().split(' ')
    subs_num = len(subjects_list)

    # get list of hands (can be only one hand) or get the pattern
    if len(hands_names.get().split(' ')[0]) == 0:
        hands_list = ['Left', 'Right']
    else:
        hands_list = hands_names.get().split(' ')
    hands_num = len(hands_list)

    # get the information required from Load_data and Classifier/Performance
    # subs_num = subs_num.get()
    # hands_num = hands_num.get()
    # rois_num = rois_num.get()
    n_times_num = var_n_times.get()

    if var_rois_apply.get():
        # get list of rois
        rois_list = rois_names.get().split(' ')
        rois_num = len(rois_list)

        # create an array to store results of the classification performance
        results = np.zeros(shape=(subs_num, hands_num, rois_num, n_times_num))
        results_subjects = np.zeros(results.shape[:-1])
        proportions_test_dataset = np.zeros(shape=results.shape)
        proportions_mean = np.zeros(results.shape[:-2])
    else:
        # create an array to store results of the classification performance
        results = np.zeros(shape=(subs_num, hands_num, n_times_num))
        results_subjects = np.zeros(results.shape[:-1])
        proportions_test_dataset = np.zeros(shape=results.shape)
        proportions_mean = np.zeros(results.shape[:-1])

    # result's labels
    labels = []

    # which time of the cross validation is that
    var_n_time_current = tk.IntVar()

    # for subject in number of subjects, etc.
    for sub in range(subs_num):
        labels.append([])

        for hand in range(hands_num):
            labels[-1].append([])

            mvpa_directory = os.path.join(
                var_base_dir.get() +
                schema.get() % (subjects_list[sub], hands_list[hand])
                )

            print(mvpa_directory)
            # load dataset using variables from load_data frame (load_data tab)
            dataset = load_data(mvpa_directory)

            # if any rois to apply first do it, else classify once
            if var_rois_apply.get():
                rois_header = []
                for roi in range(rois_num):
                    # get the data from specified import ROIs
                    roi_path = os.path.join(
                        mvpa_directory + 'ROIs/' + rois_list[roi] + '.nii.gz'
                        )
                    rois_header.append(rois_list[roi])
                    dataset_reduced = feature_reduction(dataset, roi_path)

                    var_n_time_current.set('0')

                    for n_time in range(n_times_num):
                        # create Classifier specified in Classifier tab
                        cls = create_classifier()

                        # split dataset use Classifier/Performance settings
                        training_data, test_data = split_data(
                            dataset_reduced, var_n_time_current.get()
                            )
                        proportions_test_dataset[sub][hand][roi][n_time] = \
                            test_data[1].sum()/float(test_data[1].shape[0])

                        # train and test classifier
                        cls = train_and_test_classifier(
                            cls, training_data, test_data
                            )
                        accuracy = get_accuracy(cls)
                        del cls

                        var_n_time_current.set(var_n_time_current.get() + 1)

                        results[sub][hand][roi][n_time] = accuracy
                        print(
                            '%s, %s, %s, %d ==> %0.2f' % (
                                subjects_list[sub], hands_list[hand],
                                rois_list[roi], n_time, accuracy
                                )
                            )
                    print(
                        '%s, %s, %s <mean> ==> %0.2f' % (
                            subjects_list[sub], hands_list[hand],
                            rois_list[roi], results[sub][hand][roi].mean()
                            )
                        )
                    results_subjects[sub][hand][roi] = \
                        results[sub][hand][roi].mean()
                    proportions_mean[sub][hand] = \
                        proportions_test_dataset[sub][hand][roi].mean()
            else:
                dataset_reduced = feature_reduction(dataset)

                var_n_time_current.set('0')

                for n_time in range(n_times_num):
                    # create Classifier specified in Classifier tab
                    cls = create_classifier()

                    # split dataset use Classifier/Performance settings
                    training_data, test_data = split_data(
                        dataset_reduced, var_n_time_current.get()
                        )
                    proportions_test_dataset[sub][hand][n_time] = \
                        test_data[1].sum()/float(test_data[1].shape[0])

                    # train and test classifier
                    cls = train_and_test_classifier(
                        cls, training_data, test_data
                        )
                    accuracy = get_accuracy(cls)
                    del cls

                    var_n_time_current.set(var_n_time_current.get() + 1)

                    results[sub][hand][n_time] = accuracy
                    print(
                        '%s, %s, %d ==> %0.2f' % (
                            subjects_list[sub], hands_list[hand],
                            n_time, accuracy
                            )
                        )
                print(
                    '%s, %s <mean> ==> %0.2f' % (
                        subjects_list[sub], hands_list[hand],
                        results[sub][hand].mean()
                        )
                    )
                results_subjects[sub][hand] = \
                    results[sub][hand].mean()
                proportions_mean[sub][hand] = \
                    proportions_test_dataset[sub][hand].mean()

        # delimiter = ','
        # np.savetxt(
            # os.path.join(
                # var_output_dir.get() +
                # subjects_list[sub] + '_' + hands_list[hand] + '.txt'
                # ),
            # results[sub][hand][...][...].T,
            # delimiter=delimiter,
            # header=delimiter.join(rois_header)
            # )

    print('RESULTS MEAN: %f' % results.mean())
    if var_rois_apply.get():
        for i in range(rois_num):
            print(
                'ROI 00%s: %f (%s)' %
                (i, results[:, :, i].mean(), rois_list[i])
                )

    '''
    Statistical significance vs prior chance level
    '''
    from scipy import stats

    if var_rois_apply.get():
        for roi in range(len(rois_list)):
            print(
                '\n%s statistical difference vs prior chance' % rois_list[roi]
                )
            print(
                'p_value = %f' %
                stats.ttest_1samp(
                    results_subjects[..., roi], proportions_mean.mean()
                    )[1]
                )
        results_rois = np.array(
            [
                [
                    results_subjects[..., i].mean(),
                    stats.sem(results_subjects[..., i].flatten()),
                    results_subjects[..., i].std()
                ]
                for i in range(results_subjects.shape[-1])
            ]
            )
        print('results for particular ROIs: %s' % results_rois.T[0].flatten())
    else:
        print('\nstatistical difference vs prior chance')
        print(
            'p_value = %f' %
            stats.ttest_1samp(
                results_subjects.mean(), proportions_mean.mean()
                )[1]
            )

    import datetime
    results_output_filename = datetime.datetime.now().strftime("%Y%m%d%H%M")

    if not os.path.exists(var_output_dir.get()):
        os.makedirs(var_output_dir.get())

    np.save(
        os.path.join(
            var_output_dir.get(),
            results_output_filename + '_results'
            ),
        results
        )
    np.save(
        os.path.join(
            var_output_dir.get(),
            results_output_filename + '_results_subjects'
            ),
        results_subjects
        )
    np.save(
        os.path.join(
            var_output_dir.get(),
            results_output_filename + '_results_rois'
            ),
        results_rois
        )
    np.save(
        os.path.join(
            var_output_dir.get(),
            results_output_filename + '_proportions_mean'
            ),
        proportions_mean
        )

    # from pymri.visualization.percent_bars import plot_percent_bars
    # plot_percent_bars(percents=results_rois.flatten()*100)

    import ipdb
    ipdb.set_trace()
    return results


def go():
    # dataset = load_data()
    perform_classification()


def save_config():
    import ConfigParser

    config = ConfigParser.RawConfigParser()

    config.add_section('Load data')
    config.set('Load data', 'Base directory', var_base_dir.get())
    config.set('Load data', 'Subjects', subjects_names.get())
    config.set('Load data', 'Hands', hands_names.get())
    config.set('Load data', 'Files schema', schema.get())
    config.set('Load data', 'Class 00', var_class_00.get())
    config.set('Load data', 'Class 01', var_class_01.get())

    config.add_section('Classifier')
    config.set('Classifier', 'Type', var_classifier_type.get())
    config.set('Classifier', 'k features', var_k_features.get())
    config.set('Classifier', 'Class number', var_classes_number.get())
    config.set('Classifier', 'FNN epochs', ANN_epochs.get())
    config.set('Classifier', 'FNN hidden neurons', ANN_hn.get())
    config.set('Classifier', 'FNN mini-batch size', ANN_ms.get())
    config.set('Classifier', 'FNN learning rate', ANN_eta.get())
    config.set('Classifier', 'SVC C', SVC_C.get())
    config.set('Classifier', 'SVC kernel', SVC_kernel.get())
    config.set('Classifier', 'SVC degree', SVC_degree.get())
    config.set('Classifier', 'SVC gamma', SVC_gamma.get())

    config.add_section('Performance')
    config.set('Performance', 'Metrics method', var_perform_type.get())
    config.set('Performance', 'LPO p', LPO_p.get())
    config.set('Performance', 'LPO n_times', LPO_n_times.get())
    config.set('Performance', 'LORO runs', LORO_runs.get())
    config.set('Performance', 'LORO volumes', LORO_volumes.get())

    config.add_section('Feature reduction')
    config.set('Feature reduction', 'ROIs use', var_rois_apply.get())
    config.set('Feature reduction', 'ROIs names', rois_names.get())
    config.set('Feature reduction', 'k use', var_k_features_frame.get())
    config.set('Feature reduction', 'k features', var_k_features.get())
    config.set('Feature reduction', 'method', var_reduction_method.get())
    config.set('Feature reduction', 'Normalize', var_normalize.get())

    config.add_section('Output')
    config.set('Output', 'Output directory', var_output_dir.get())

    with open('pymri.cfg', 'wb') as configfile:
        config.write(configfile)

    print('Configuration successfully saved!')


def load_config():
    import ConfigParser

    config = ConfigParser.RawConfigParser()
    config.read('pymri.cfg')
    var_base_dir.set(config.get('Load data', 'Base directory'))
    subjects_names.set(config.get('Load data', 'Subjects'))
    hands_names.set(config.get('Load data', 'Hands'))
    schema.set(config.get('Load data', 'Files schema'))
    var_class_00.set(config.get('Load data', 'Class 00'))
    var_class_01.set(config.get('Load data', 'Class 01'))

    var_classifier_type.set(config.get('Classifier', 'Type'))
    var_k_features.set(config.get('Classifier', 'k features'))
    var_classes_number.set(config.get('Classifier', 'Class number'))
    ANN_epochs.set(config.get('Classifier', 'FNN epochs'))
    ANN_hn.set(config.get('Classifier', 'FNN hidden neurons'))
    ANN_ms.set(config.get('Classifier', 'FNN mini-batch size'))
    ANN_eta.set(config.get('Classifier', 'FNN learning rate'))
    SVC_C.set(config.get('Classifier', 'SVC C'))
    SVC_kernel.set(config.get('Classifier', 'SVC kernel'))
    SVC_degree.set(config.get('Classifier', 'SVC degree'))
    SVC_gamma.set(config.get('Classifier', 'SVC gamma'))

    var_perform_type.set(config.get('Performance', 'Metrics method'))
    LPO_p.set(config.get('Performance', 'LPO p'))
    LPO_n_times.set(config.get('Performance', 'LPO n_times'))
    LORO_runs.set(config.get('Performance', 'LORO runs'))
    LORO_volumes.set(config.get('Performance', 'LORO volumes'))

    var_rois_apply.set(config.get('Feature reduction', 'ROIs use'))
    rois_names.set(config.get('Feature reduction', 'ROIs names'))
    var_k_features_frame.set(config.get('Feature reduction', 'k use'))
    var_k_features.set(config.get('Feature reduction', 'k features'))
    var_reduction_method.set(config.get('Feature reduction', 'method'))
    var_normalize.set(config.get('Feature reduction', 'Normalize'))

    var_output_dir.set(config.get('Output', 'Output directory'))

    perform_choose(var_perform_type.get())
    classifier_choose(var_classifier_type.get())

    print('Configuration successfully loaded!')


def quit():
    root.destroy()


###########################################################################
#
#     CREATE MAIN FRAME
#
###########################################################################

root = tk.Tk()

font_size = 12
font_standard = tkFont.Font(size=font_size)
font_tab = tkFont.Font(size=font_size, weight='bold')
style = ttk.Style()

# tab text padding
style.theme_create(
    "tab_style", parent="alt", settings={
        "TNotebook.Tab": {
            "configure": {
                "padding": [50, 10]
            },
            "expand": [("selected", [1, 1, 1, 0])]
        }
    }
    )

style.theme_use("tab_style")
style.configure('.', font=font_tab)

root.wm_title('PyMRI main window')

note = ttk.Notebook(root)

getFld = tk.IntVar()
var_ev = tk.IntVar()

var_bold = tk.StringVar()
var_attr = tk.StringVar()
var_attr_lit = tk.StringVar()
var_mask_brain = tk.StringVar()


###########################################################################
#
#     LOAD DATA FRAME
#
###########################################################################

load_data_frame = tk.Frame(note)

note.add(load_data_frame, text='Load data', padding=20)
note.grid(
    row=0, column=0, columnspan=10, sticky='WE',
    padx=20, pady=10, ipadx=10, ipady=10
    )

# Number of subjects and hands to test the classifier on ######################

# Base directory (Project main directory) ###
var_base_dir = tk.StringVar()

base_dir_lbl = tk.Label(
    load_data_frame, text="Base directory:", font=font_standard
    )
base_dir_lbl.grid(
    row=0, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

base_dir_txt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=var_base_dir
    )
base_dir_txt.grid(row=0, column=2, columnspan=10, sticky="WE", pady=5)


# Specify subjects (list, or schema path) ###

# subjects_number variable is set in load_data() function
subjects_number = tk.IntVar()
subjects_names = tk.StringVar()

subjects_lbl = tk.Label(
    load_data_frame, text="Subjects:", font=font_standard
    )
subjects_lbl.grid(
    row=1, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

subjects_txt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=subjects_names
    )
subjects_txt.grid(row=1, column=2, columnspan=10, sticky="WE", pady=5)


# Specify hands (list them, if nothing set get both hands) ###

# hands_number variable is set in load_data() function
hands_number = tk.IntVar()
hands_names = tk.StringVar()

hands_lbl = tk.Label(
    load_data_frame, text="Hands:", font=font_standard
    )
hands_lbl.grid(
    row=2, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

hands_txt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=hands_names
    )
hands_txt.grid(row=2, column=2, columnspan=10, sticky="WE", pady=5)


# Files schema ###

schema = tk.StringVar()

schema_lbl = tk.Label(
    load_data_frame, text="Files schema:", font=font_standard
    )
schema_lbl.grid(
    row=3, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

schema_txt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=schema
    )
schema_txt.grid(row=3, column=2, columnspan=10, sticky="WE", pady=5)

sep_paths_classes = ttk.Separator(load_data_frame, orient=tk.HORIZONTAL)
sep_paths_classes.grid(row=4, column=1, columnspan=9, sticky='EW', pady=10)


# classes (contrasts) get ###
# class_00
var_class_00 = tk.StringVar()

constrast_00_lbl = tk.Label(
    load_data_frame, text="Class 00:", font=font_standard
    )
constrast_00_lbl.grid(
    row=5, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

constast_00_txt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=var_class_00
    )
constast_00_txt.grid(row=5, column=2, columnspan=10, sticky="WE", pady=5)

var_class_01 = tk.StringVar()

# class_01
var_class_01 = tk.StringVar()
constrast_01_lbl = tk.Label(
    load_data_frame, text="Class 01:", font=font_standard
    )
constrast_01_lbl.grid(
    row=6, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

constast_01_txt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=var_class_01
    )
constast_01_txt.grid(row=6, column=2, columnspan=10, sticky="WE", pady=5)


var_classes_number = tk.IntVar()
var_classes_number.set(2)


###############################################################################
#
#     FEATURE REDUCTION (SELECTION/EXTRACTION) FRAME
#
###############################################################################

# Feature frame Setup #########################################################
feature_frame = tk.Frame(note)
feature_frame.grid(
    row=1, column=0, columnspan=10, sticky='WE',
    padx=20, pady=10, ipadx=10, ipady=10
    )

note.add(feature_frame, text='Feature reduction', padding=20)
note.grid(
    row=0, column=0, columnspan=10, sticky='WE',
    padx=20, pady=10, ipadx=10, ipady=10
    )


# Fill feature frame ##########################################################

# ROIs frame ###############################

var_rois_apply = tk.IntVar()
rois_text = "Regions of Intrest (ROIs)"

rois_check = tk.Checkbutton(
    feature_frame, text=rois_text, onvalue=1, offvalue=0,
    command=lambda: check_frame(
        rois_frame, var_rois_apply, rois_check, rois_text
        ),
    variable=var_rois_apply, font=font_standard
    )
rois_check.grid(row=0, column=0, columnspan=1, padx=10, pady=10, sticky='W')

rois_frame = tk.LabelFrame(
    feature_frame, borderwidth=font_size/6,
    text=rois_text, font=font_standard
    )


# ROIs names
rois_names = tk.StringVar()

rois_lbl = tk.Label(
    rois_frame, text="ROIs names:", font=font_standard
    )
rois_lbl.grid(
    row=0, column=0, columnspan=1, sticky='E', padx=5, pady=5
    )

rois_entry = tk.Entry(
    rois_frame, width=40, font=font_standard, textvariable=rois_names
    )
rois_entry.grid(row=0, column=2, columnspan=10, sticky="WE", pady=5)


# Feature reduction frame ##################

# Reduction method

var_k_features_frame = tk.IntVar()
k_features_text = "Feature reduction"

k_features_check = tk.Checkbutton(
    feature_frame, text=k_features_text, onvalue=1, offvalue=0,
    command=lambda: check_frame(
        k_features_frame, var_k_features_frame,
        k_features_check, k_features_text, row=1
        ),
    variable=var_k_features_frame, font=font_standard
    )
k_features_check.grid(
    row=1, column=0, columnspan=1, padx=10, pady=10, sticky='W'
    )

k_features_frame = tk.LabelFrame(
    feature_frame, borderwidth=font_size/6,
    text=k_features_text, font=font_standard
    )

var_k_features = tk.IntVar()

k_label = tk.Label(
    k_features_frame,
    text="Reduct to (extract/select) n features:", font=font_standard
    )
k_label.grid(row=0, column=0, sticky='W')

k_features_entry = tk.Entry(
    k_features_frame, width=4, textvariable=var_k_features, font=font_standard
    )
k_features_entry.grid(row=0, column=1)

# Feature reduction method
var_reduction_method = tk.StringVar()
var_reduction_method.set("SelectKHighest from mask (SKH)")

feature_option = tk.OptionMenu(
    k_features_frame, var_reduction_method,
    "SelectKHighest from mask (SKH)",
    "SelectKBest (SKB)",
    # "Principal Component Analysis (PCA)",
    # "Restricted Boltzmann Machine (RBM)",
    )
feature_option.grid(row=1, stick='W', padx='20')
feature_option.configure(font=font_standard)

# Normalize data ###
var_normalize = tk.IntVar()

normalize_check = tk.Checkbutton(
    feature_frame,
    text="Normalize data", onvalue=1, offvalue=0,
    variable=var_normalize,
    font=font_standard
    )
normalize_check.grid(
    row=3, column=0, columnspan=3, padx=10, pady=10, sticky='W'
    )

'''
# PCA feature extraction method ###
PCA_frame = tk.Frame(feature_frame)

PCA_whiten = tk.BooleanVar()
PCA_chk_whiten = tk.Checkbutton(
    PCA_frame, text="whiten", variable=PCA_whiten, font=font_standard)
PCA_chk_whiten.grid(row=3, column=1)


# RBM feature extraction method ###
RBM_frame = tk.Frame(feature_frame)

# learning rate
RBM_lr = tk.StringVar()
RBM_lr.set('0.1')
RBM_lr_label = tk.Label(
    RBM_frame, text="Learning rate:", font=font_standard
    )
RBM_lr_label.grid(row=0, column=0, padx=5, pady=2)
RBM_lr_entry = tk.Entry(
    RBM_frame, width=5, font=font_standard, textvariable=RBM_lr
    )
RBM_lr_entry.grid(row=0, column=1, pady=2)

# batch size
RBM_bs = tk.StringVar()
RBM_bs.set('10')
RBM_bs_label = tk.Label(
    RBM_frame, text="Batch size:", font=font_standard
    )
RBM_bs_label.grid(row=0, column=2, padx=5, pady=2)
RBM_bs_entry = tk.Entry(
    RBM_frame, width=5, font=font_standard, textvariable=RBM_bs
    )
RBM_bs_entry.grid(row=0, column=3, pady=2)

# n_iter
RBM_ni = tk.StringVar()
RBM_ni.set('10')
RBM_ni_label = tk.Label(
    RBM_frame, text="n_iter:", font=font_standard
    )
RBM_ni_label.grid(row=0, column=4, padx=5, pady=2)
RBM_ni_entry = tk.Entry(
    RBM_frame, width=4, font=font_standard, textvariable=RBM_ni
    )
RBM_ni_entry.grid(row=0, column=5, pady=2)

# verbose
RBM_ver = tk.StringVar()
RBM_ver.set('0')
RBM_ver_label = tk.Label(
    RBM_frame, text="verbose:", font=font_standard
    )
RBM_ver_label.grid(row=1, column=0, sticky='E', padx=5, pady=2)
RBM_ver_entry = tk.Entry(
    RBM_frame, width=5, font=font_standard, textvariable=RBM_ver
    )
RBM_ver_entry.grid(row=1, column=1, pady=2)

# random state
RBM_rs = tk.StringVar()
RBM_rs.set('None')
RBM_rs_label = tk.Label(
    RBM_frame, text="n_iter:", font=font_standard
    )
RBM_rs_label.grid(row=1, column=2, padx=5, pady=2)
RBM_rs_entry = tk.Entry(
    RBM_frame, width=5, font=font_standard, textvariable=RBM_rs
    )
RBM_rs_entry.grid(row=1, column=3, pady=2)
'''

###########################################################################
#
#     CLASSIFIER FRAME
#
###########################################################################

# Frame Setup #############################################################
classifier_frame = tk.Frame(root)

note.add(classifier_frame, text='Classifier', padding=20)
note.grid(
    row=0, column=0, columnspan=10, sticky='WE',
    padx=20, pady=10, ipadx=10, ipady=10
    )

# Fill frame ##############################################################
var_classifier_type = tk.StringVar()
var_classifier_type.set(
    "Feedforward Neural Network - simple python script (FNN simple)"
    )

classifier_option = tk.OptionMenu(
    classifier_frame, var_classifier_type,
    "Feedforward Neural Network - simple python script (FNN simple)",
    "Feedforward Neural Network - theano version (FNN theano)",
    "Convolutional Neural Network (CNN)",
    "Supported Vector Classifier (SVC)",
    "Linear Discriminant Analysis (LDA)",
    "Quadratic Discriminant Analysis (QDA)",
    command=classifier_choose
    )
classifier_option.grid(row=1, stick='WS', padx='20')
classifier_option.configure(font=font_standard)


# ANN parameters ###
ANN_frame = tk.Frame(classifier_frame)

# nnadl for FNN simple
var_nnadl = tk.IntVar()

# Input neurons
ANN_in = tk.IntVar()
# ANN_in.set(var_k_features.get())
ANN_in_label = tk.Label(
    ANN_frame, text="Input neurons:", font=font_standard
    )
ANN_in_label.grid(row=0, column=0, padx=5, pady=2)
ANN_in_entry = tk.Entry(
    ANN_frame, width=5, font=font_standard, textvariable=var_k_features
    )
ANN_in_entry.grid(row=0, column=1, pady=2)

# Hidden neurons
ANN_hn = tk.IntVar()
ANN_hn.set(30)
ANN_hn_label = tk.Label(
    ANN_frame, text="Hidden neurons:", font=font_standard
    )
ANN_hn_label.grid(row=0, column=2, padx=5, pady=2)
ANN_hn_entry = tk.Entry(
    ANN_frame, width=5, font=font_standard, textvariable=ANN_hn
    )
ANN_hn_entry.grid(row=0, column=3, pady=2)

# Output neurons
ANN_out = tk.IntVar()
ANN_out.set(2)
ANN_out_label = tk.Label(
    ANN_frame, text="Output neurons:", font=font_standard
    )
ANN_out_label.grid(row=0, column=4, padx=5, pady=2)
ANN_out_entry = tk.Entry(
    ANN_frame, width=5, font=font_standard, textvariable=ANN_out
    )
ANN_out_entry.grid(row=0, column=5, pady=2)

# Epochs
ANN_epochs = tk.IntVar()
ANN_epochs.set(100)
ANN_epochs_label = tk.Label(
    ANN_frame, text="Epochs:", font=font_standard
    )
ANN_epochs_label.grid(row=1, column=0, padx=5, pady=2)
ANN_epochs_entry = tk.Entry(
    ANN_frame, width=5, font=font_standard, textvariable=ANN_epochs
    )
ANN_epochs_entry.grid(row=1, column=1, pady=2)


# eta learning rate
ANN_eta = tk.DoubleVar()
ANN_eta.set(3.0)
ANN_eta_label = tk.Label(
    ANN_frame, text="learning rate (eta):", font=font_standard
    )
ANN_eta_label.grid(row=1, column=2, padx=5, pady=2)
ANN_eta_entry = tk.Entry(
    ANN_frame, width=5, font=font_standard, textvariable=ANN_eta
    )
ANN_eta_entry.grid(row=1, column=3, pady=2)

# Mini-batch size
ANN_ms = tk.IntVar()
ANN_ms.set(10)
ANN_ms_label = tk.Label(
    ANN_frame, text="mini-batch size:", font=font_standard
    )
ANN_ms_label.grid(row=1, column=4, padx=5, pady=2)
ANN_ms_entry = tk.Entry(
    ANN_frame, width=4, font=font_standard, textvariable=ANN_ms
    )
ANN_ms_entry.grid(row=1, column=5, pady=2)


# SVC parameters ###
SVC_frame = tk.Frame(classifier_frame)

# C
SVC_C = tk.DoubleVar()
SVC_C.set(1.0)
SVC_C_label = tk.Label(
    SVC_frame, text="C:", font=font_standard
    )
SVC_C_label.grid(row=0, column=0, padx=5, pady=2)
SVC_C_entry = tk.Entry(
    SVC_frame, width=5, font=font_standard, textvariable=SVC_C
    )
SVC_C_entry.grid(row=0, column=1, pady=2)

# kernel
SVC_kernel = tk.StringVar()
SVC_kernel.set('linear')
SVC_kernel_label = tk.Label(
    SVC_frame, text="kernel:", font=font_standard
    )
SVC_kernel_label.grid(row=0, column=2, padx=5, pady=2)
SVC_kernel_entry = tk.Entry(
    SVC_frame, width=5, font=font_standard, textvariable=SVC_kernel
    )
SVC_kernel_entry.grid(row=0, column=3, pady=2)

# degree
SVC_degree = tk.IntVar()
SVC_degree.set(3)
SVC_degree_label = tk.Label(
    SVC_frame, text="degree:", font=font_standard
    )
SVC_degree_label.grid(row=0, column=4, padx=5, pady=2)
SVC_degree_entry = tk.Entry(
    SVC_frame, width=5, font=font_standard, textvariable=SVC_degree
    )
SVC_degree_entry.grid(row=0, column=5, pady=2)

# gamma
SVC_gamma = tk.StringVar()
SVC_gamma.set('auto')
SVC_gamma_label = tk.Label(
    SVC_frame, text="gamma:", font=font_standard
    )
SVC_gamma_label.grid(row=0, column=6, padx=5, pady=2)
SVC_gamma_entry = tk.Entry(
    SVC_frame, width=5, font=font_standard, textvariable=SVC_gamma
    )
SVC_gamma_entry.grid(row=0, column=7, pady=2)


# ### Cross validate ##########################################################
perform_frame = tk.LabelFrame(
    classifier_frame, text='Performance metrics', font=font_standard
    )
perform_frame.grid(
    row=3, column=0, sticky='W',
    padx='20', pady='20', ipady='10'
    )

perform_options = [
    'LeavePOut (LPO randomly)',
    'LeaveOneRunOut (LORO)',
    'Shuffle Split',
    'KFold',
    'Train Test Split'
    ]

var_perform_type = tk.StringVar()
var_perform_type.set(perform_options[0])

perform_options_menu = tk.OptionMenu(
    perform_frame, var_perform_type,
    *perform_options,
    command=perform_choose
    )
perform_options_menu.grid(row=1, column=0, stick='W', padx='20')
perform_options_menu.configure(font=font_standard)

var_n_times = tk.IntVar()


# LPO parameters ###
LPO_frame = tk.Frame(perform_frame)

# leave p
LPO_p = tk.DoubleVar()
LPO_p.set('0.25')
LPO_p_label = tk.Label(
    LPO_frame, text="p (proportion or number):", font=font_standard
    )
LPO_p_label.grid(row=0, column=0, padx=5, pady=2)
LPO_p_entry = tk.Entry(
    LPO_frame, width=5, font=font_standard, textvariable=LPO_p
    )
LPO_p_entry.grid(row=0, column=1, pady=2)

# n_times perform
LPO_n_times = tk.IntVar()
LPO_n_times.set('6')
LPO_n_times_label = tk.Label(
    LPO_frame, text="LeavePOut k times:", font=font_standard
    )
LPO_n_times_label.grid(row=0, column=2, padx=5, pady=2)
LPO_n_times_entry = tk.Entry(
    LPO_frame, width=5, font=font_standard, textvariable=LPO_n_times
    )
LPO_n_times_entry.grid(row=0, column=3, pady=2)

# LORO parameters ###
LORO_frame = tk.Frame(perform_frame)

var_n_time_current = tk.IntVar(None)

# leave p
LORO_runs = tk.IntVar()
LORO_runs.set(5)
LORO_runs_label = tk.Label(
    LORO_frame, text="runs:", font=font_standard
    )
LORO_runs_label.grid(row=0, column=0, padx=5, pady=2)
LORO_runs_entry = tk.Entry(
    LORO_frame, width=5, font=font_standard, textvariable=LORO_runs
    )
LORO_runs_entry.grid(row=0, column=1, pady=2)

# volumes perform
LORO_volumes = tk.IntVar()
LORO_volumes.set(145)
LORO_volumes_label = tk.Label(
    LORO_frame, text="volumes:", font=font_standard
    )
LORO_volumes_label.grid(row=0, column=2, padx=5, pady=2)
LORO_volumes_entry = tk.Entry(
    LORO_frame, width=5, font=font_standard, textvariable=LORO_volumes
    )
LORO_volumes_entry.grid(row=0, column=3, pady=2)

###########################################################################
#
#     OUTPUT FRAME
#
###########################################################################

var_output_dir = tk.StringVar()

output_frame = tk.Frame(note)

note.add(output_frame, text='Output and logs', padding=20)
note.grid(
    row=0, column=0, columnspan=10, sticky='WE',
    padx=20, pady=10, ipadx=10, ipady=10
    )

# Output ###################################################################
output_lbl = tk.Label(
    output_frame, text="Output directory:", font=font_standard
    )
output_lbl.grid(row=0, column=0, columnspan=2, sticky='E', padx=5, pady=5)

output_entry = tk.Entry(
    output_frame, width=40, font=font_standard, textvariable=var_output_dir
    )
output_entry.grid(row=0, column=2, columnspan=8, sticky="WE", pady=5)

output_button = tk.Button(
    output_frame,
    command=lambda: askdir(var_output_dir),
    text="Browse ...", font=font_standard
    )
output_button.grid(row=0, column=10, sticky='W', padx=5, pady=5)


###########################################################################
#
#     MENU
#
###########################################################################

options = tk.Frame(
    root, borderwidth=font_size/6
    )

options.grid(
    row=4, column=0, columnspan=10, sticky='WE',
    padx=10, pady=10, ipadx=10, ipady=10
    )

go = tk.Button(
    options,
    command=go,
    text=" Go ", font=font_standard
    )
go.grid(row=4, column=0, padx=10, pady=10)

save = tk.Button(
    options,
    command=save_config,
    text="Save", font=font_standard
    )
save.grid(row=4, column=1, padx=10, pady=10)

load = tk.Button(
    options,
    command=load_config,
    text="Load", font=font_standard
    )
load.grid(row=4, column=2, padx=10, pady=10)

help = tk.Button(
    options,
    command=lambda: help_display(8),
    text="Help", font=font_standard
    )
help.grid(row=4, column=3, padx=10, pady=10)

quit = tk.Button(
    options,
    command=quit,
    text="Quit", font=font_standard
    )
quit.grid(row=4, column=4, padx=10, pady=10)


perform_choose(var_perform_type.get())
classifier_choose(var_classifier_type.get())
options.mainloop()
