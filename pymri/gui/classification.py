import Tkinter as tk
import tkFont
import tkFileDialog

import ttk


# TODO: entire perform() function in subprocess to allow window to refresh
# TODO: create separate classes for each section to keep it clear

def classifier_choose(var_classifier_type):
    if 'fnn' in var_classifier_type:
        fANN_frame.grid(row=2, column=0, columnspan=10, padx=20, pady=10)
    else:
        fANN_frame.grid_forget()


def perform_choose(var_perform_type):
    if var_perform_type == 'LeavePOut (LPO randomly)':
        LPO_frame.grid(row=2, column=0, columnspan=10, padx=20, pady=10)
    else:
        LPO_frame.grid_forget()


def help_display(number):
    print number


def askfile(var):
    file_chosen = tkFileDialog.askopenfilename()
    var.set(file_chosen)


def askdir_output():
    dir_chosen = tkFileDialog.askdirectory()
    var_output.set(dir_chosen)


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


def perform():
    # TODO: it is better to create a list of commands
    # because program can go through all code in search for errors

    # Load data ###

    from pymri.dataset.datasets import DatasetManager
    # dataset settings

    # in case we use fann as classifier we have to reshape data
    if 'fANN' in var_classifier_type.get():
        nnadl = True
    else:
        nnadl = False

    ds = DatasetManager(
        # path_input='/home/jesmasta/amu/master/nifti/bold/',
        path_bold=var_bold.get(),
        path_attr=var_attr.get(),
        path_attr_lit=var_attr_lit.get(),
        path_mask_brain=var_mask_brain.get(),
        path_output=var_output.get(),
        # conditions has to be tuples
        contrast=(
            tuple(var_class_00.get().split(' ')),
            tuple(var_class_01.get().split(' '))
            ),
        nnadl = nnadl
        )
    # load data
    ds.load_data()

    # TODO: more universal function needed

    if 'SKB' in var_feature_method.get():
        feature_arguments = None
    elif 'ROI' in var_feature_method.get():
        # print var_mask_roi.get()
        feature_arguments = var_mask_roi.get()
        # print(feature_arguments)

    # select feature reduction method
    ds.feature_reduction(
        roi_selection=var_feature_method.get(),
        k_features=var_k_features.get(),
        normalize = var_normalize.get(),
        feature_arguments=feature_arguments
        )

    ###########################################################################
    #
    #        CREATE MODEL
    #
    ###########################################################################

    print(var_perform_type.get())
    if 'LPO' in var_perform_type.get():
        import numpy as np
        accuracies = np.zeros(shape=(LPO_n_times.get(),))
        for i in range(LPO_n_times.get()):
            # get training, validation and test datasets for specified roi
            # training_data, validation_data, test_data = ds.split_data()
            training_data, test_data, valid_data = ds.split_data(
                sizes=(1-LPO_p.get(), LPO_p.get())
                )
            if 'fANN' in var_classifier_type.get():
                # artificial neural network
                from pymri.model import fnn

                net = fnn.Network([ds.k_features, fANN_hn.get(), 2])
                # train and test network
                net.SGD(
                    training_data, fANN_epochs.get(),
                    fANN_ms.get(), fANN_eta.get(), test_data=test_data
                    )

                # record the best result
                accuracies[i] = net.best_score/float(len(test_data))

            if 'theano fnn' in var_classifier_type.get():
                print('--- THEANO FNN ---')
                

                import pymri.model.network3 as network3
                from pymri.model.network3 import Network

                training_data, validation_data, test_data = \
                    network3.load_data_shared()
                mini_batch_size = 10
                net = Network([
                    network3.FullyConnectedLayer(n_in=784, n_out=100),
                    network3.SoftmaxLayer(n_in=100, n_out=2)], mini_batch_size)
                net.SGD(training_data, 300, mini_batch_size, 0.1, 
                        validation_data, test_data)
                accuracies[i] = net.best_validation_accuracy

    mean_accuracy = accuracies.mean()
    print('\n\nmean accuracy: %f' % mean_accuracy)

    ###########################################################################
    #
    #   VISUALIZE
    #
    ###########################################################################

def load_data(mvpa_directory):

    from pymri.dataset.datasets import DatasetManager

    print('Loading database from %s' % mvpa_directory)
    dataset = DatasetManager(
        mvpa_directory = mvpa_directory,
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

    if 'fnn' in var_classifier_type.get():

        from pymri.model import FNN

        if 'simple' in var_classifier_type.get():
            type = 'fnn simple'

        if 'theano' in var_classifier_type.get():
            type = 'fnn theano'

        # all parameters - for net architecture as well as for its training
        # has to be specified here (for reason see Classifier class)
        cls = FNN(
            type=type,
            input_layer_size=var_k_features.get(),
            hidden_layer_size=fANN_hn.get(),
            output_layer_size=var_classes_number.get(),
            epochs=fANN_epochs.get(),
            minibatch_size=fANN_ms.get(),
            learning_rate=fANN_eta.get()
            )
    return cls


def feature_reduction(dataset, roi_path=None):

    dataset.feature_reduction(
        roi_path=roi_path,
        k_features=var_k_features.get(),
        reduction_method=var_reduction_method.get(),
        normalize = var_normalize.get(),
        nnadl=True
        )

    return dataset

def split_data(dataset):

    training_data, test_data, valid_data = dataset.split_data(
        sizes=(1-LPO_p.get(), LPO_p.get())
        )
    return training_data, test_data


def train_and_test_classifier(cls, training_data, test_data):
    '''
    Training and testing are somtimes bound with each other.
    '''
    cls.train_and_test(training_data, test_data)

    return cls

def get_best_accuracy(cls):
    return cls.get_best_accuracy()

def visualise_results(results):

    import numpy as np
    import matplotlib.pyplot as plt

    mean_accuracies = np.zeros(shape=(LPO_n_times.get(),))
    for i in range(LPO_n_times.get()):
        mean_accuracies[i] = accuracies[:i+1].mean()

    # plot best accuracy for particular validation
    plt.scatter(
        range(len(accuracies)), accuracies,
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


    # get list of subjects (can be one subject specifiedz or get the pattern
    if '%' in subjects_names.get() and len(subjects.get().split(' ')) == 1:
        subjects_pattern = subjects_names.get()
    else:
        subjects_lists = subjects_names.get().split(' ')
        subs_num = len(subjects_lists)

    # get list of hands (can be only one hand) or get the pattern
    if len(hands_names.get().split(' ')[0]) == 0:
        hands_list = ['Left', 'Right']
    else:
        hands_list = hands_names.get().split(' ')
    hands_num = len(hands_list)
    

    # get list of rois
    rois_list = rois_names.get().split(' ')
    rois_num = len(rois_list)

    n_times_num = LPO_n_times.get()
    

    # get the information required from Load_data and Classifier/Performance
    # subs_num = subs_num.get()
    # hands_num = hands_num.get()
    # rois_num = rois_num.get()
    # n_times_num = n_times_num.get()

    # create an array to store results of the classification performance
    results = np.zeros(shape=(subs_num, hands_num, rois_num, n_times_num))

    # for subject in number of subjects, etc.

    for sub in range(subs_num):

        for hand in range(hands_num):

            mvpa_directory = \
                schema.get() % (subjects_lists[sub], hands_list[hand])

            print(mvpa_directory)
            # load dataset using variables from load_data frame (load_data tab))
            dataset = load_data(mvpa_directory)

            for roi in range(rois_num):
                # get the data from specified import ROIs
                roi_path = mvpa_directory + rois_list[roi] + '.nii.gz'
                dataset_reduced = feature_reduction(dataset, roi_path)

                for n_time in range(n_times_num):
                    # create Classifier specified in Classifier tab
                    cls = create_classifier()

                    # split dataset use Classifier/Performance settings
                    training_data, test_data = split_data(dataset_reduced)

                    # train and test classifier
                    cls = train_and_test_classifier(
                        cls, training_data, test_data
                        )
                    best_accuracy = get_best_accuracy(cls)
                    del cls


                    results[sub][hand][roi][n_time] = \
                        best_accuracy / float(len(test_data))

    import pdb
    pdb.set_trace()
    return results

def go():
    # dataset = load_data()
    perform_classification()

def save_config():
    import ConfigParser

    config = ConfigParser.RawConfigParser()

    config.add_section('Load data')
    config.set('Load data', 'Subjects', subjects_names.get())
    config.set('Load data', 'Hands', hands_names.get())
    config.set('Load data', 'Files schema', schema.get())
    config.set('Load data', 'Class 00', var_class_00.get())
    config.set('Load data', 'Class 01', var_class_01.get())

    config.add_section('Classifier')
    config.set('Classifier', 'Type', var_classifier_type.get())
    config.set('Classifier', 'k features', var_k_features.get())
    config.set('Classifier', 'Class number', var_classes_number.get())
    config.set('Classifier', 'fnn epochs', fANN_epochs.get())
    config.set('Classifier', 'fnn hidden neurons', fANN_hn.get())
    config.set('Classifier', 'fnn minibatch size', fANN_ms.get())
    config.set('Classifier', 'fnn learning rate', fANN_eta.get())

    config.add_section('Performance')
    config.set('Performance', 'Metrics method', var_perform_type.get())
    config.set('Performance', 'LeavePOut', LPO_p.get())
    config.set('Performance', 'n_times', LPO_n_times.get())

    config.add_section('Feature reduction')
    config.set('Feature reduction', 'ROIs use', var_rois_frame.get())
    config.set('Feature reduction', 'ROIs names', rois_names.get())
    config.set('Feature reduction', 'k use', var_k_features_frame.get())
    config.set('Feature reduction', 'k features', var_k_features.get())
    config.set('Feature reduction', 'Normalize', var_normalize.get())

    with open('pymri.cfg', 'wb') as configfile:
        config.write(configfile)

    print('Configuration successfully saved!')

def load_config():
    import ConfigParser

    config = ConfigParser.RawConfigParser()
    config.read('pymri.cfg')
    subjects_names.set(config.get('Load data', 'Subjects'))
    hands_names.set(config.get('Load data', 'Hands'))
    schema.set(config.get('Load data', 'Files schema'))
    var_class_00.set(config.get('Load data', 'Class 00'))
    var_class_01.set(config.get('Load data', 'Class 01'))

    var_classifier_type.set(config.get('Classifier', 'Type'))
    var_k_features.set(config.get('Classifier', 'k features'))
    var_classes_number.set(config.get('Classifier', 'Class number'))
    fANN_epochs.set(config.get('Classifier', 'fnn epochs'))
    fANN_hn.set(config.get('Classifier', 'fnn hidden neurons'))
    fANN_ms.set(config.get('Classifier', 'fnn minibatch size'))
    fANN_eta.set(config.get('Classifier', 'fnn learning rate'))

    var_perform_type.set(config.get('Performance', 'Metrics method'))
    LPO_p.set(config.get('Performance', 'LeavePOut'))
    LPO_n_times.set(config.get('Performance', 'n_times'))

    var_rois_frame.set(config.get('Feature reduction', 'ROIs use'))
    rois_names.set(config.get('Feature reduction', 'ROIs names'))
    var_k_features_frame.set(config.get('Feature reduction', 'k use'))
    var_k_features.set(config.get('Feature reduction', 'k features'))
    var_normalize.set(config.get('Feature reduction', 'Normalize'))


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

# Specify subjects (list, or schema path) ###

# subjects_number variable is set in load_data() function
subjects_number = tk.IntVar()
subjects_names = tk.StringVar()

subjects_lbl = tk.Label(
    load_data_frame, text="Subjects:", font=font_standard
    )
subjects_lbl.grid(
    row=0, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

subjects_txt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=subjects_names
    )
subjects_txt.grid(row=0, column=2, columnspan=10, sticky="WE", pady=5)


# Specify hands (list them, if nothing set get both hands) ###

# hands_number variable is set in load_data() function
hands_number = tk.IntVar()
hands_names = tk.StringVar()

hands_lbl = tk.Label(
    load_data_frame, text="Hands:", font=font_standard
    )
hands_lbl.grid(
    row=1, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

hands_txt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=hands_names
    )
hands_txt.grid(row=1, column=2, columnspan=10, sticky="WE", pady=5)


# Files schema ###

schema = tk.StringVar()

schema_lbl = tk.Label(
    load_data_frame, text="Files schema:", font=font_standard
    )
schema_lbl.grid(
    row=2, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

schema_txt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=schema
    )
schema_txt.grid(row=2, column=2, columnspan=10, sticky="WE", pady=5)

sep_paths_classes = ttk.Separator(load_data_frame, orient=tk.HORIZONTAL)
sep_paths_classes.grid(row=3, column=1, columnspan=9, sticky='EW', pady=10)


# classes (contrasts) get ###
# class_00
var_class_00 = tk.StringVar()

constrast_00_lbl = tk.Label(
    load_data_frame, text="Class 00:", font=font_standard
    )
constrast_00_lbl.grid(
    row=4, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

constast_00_txt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=var_class_00
    )
constast_00_txt.grid(row=4, column=2, columnspan=10, sticky="WE", pady=5)

var_class_01 = tk.StringVar()

# class_01
var_class_01 = tk.StringVar()
constrast_01_lbl = tk.Label(
    load_data_frame, text="Class 01:", font=font_standard
    )
constrast_01_lbl.grid(
    row=5, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

constast_01_txt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=var_class_01
    )
constast_01_txt.grid(row=5, column=2, columnspan=10, sticky="WE", pady=5)


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

var_rois_frame = tk.IntVar()
rois_text = "Regions of Intrest (ROIs)"

rois_check = tk.Checkbutton(
    feature_frame, text=rois_text, onvalue=1, offvalue=0,
    command=lambda: check_frame(
        rois_frame, var_rois_frame, rois_check, rois_text
        ),
    variable=var_rois_frame, font=font_standard
    )
rois_check.grid(row=0, column=0, columnspan=1, padx=10, pady=10, sticky='W')

rois_frame = tk.LabelFrame(
    feature_frame, borderwidth = font_size/6,
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
var_reduction_method = tk.StringVar()
var_reduction_method.set('SelectKBest (SKB)')


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
k_features_check.grid(row=1, column=0, columnspan=1, padx=10, pady=10, sticky='W')

k_features_frame = tk.LabelFrame(
    feature_frame, borderwidth = font_size/6,
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




# # Feature Selection/Extraction method ###
# var_feature_method = tk.StringVar()
# var_feature_method.set("SelectKBest (SKB)")
# feature_arguments=None

# feature_option = tk.OptionMenu(
    # feature_frame, var_feature_method,
    # "SelectKBest (SKB)",
    # "ROI mask",
    # "Principal Component Analysis (PCA)",
    # "Restricted Boltzmann Machine (RBM)",
    # command=feature_choose
    # )
# feature_option.grid(row=2, stick='W', padx='20')
# feature_option.configure(font=font_standard)

# # ROI mask feature selection method ###
# ROI_frame = tk.Frame(feature_frame)

# var_mask_roi = tk.StringVar()

# ROI_label = tk.Label(
    # ROI_frame, text="Mask file:", font=font_standard
    # )
# ROI_label.grid(row=0, column=0, sticky='E', padx=5, pady=5)

# ROI_entry = tk.Entry(
    # ROI_frame, width=40, font=font_standard, textvariable=var_mask_roi
    # )
# ROI_entry.grid(row=0, column=1, columnspan=8, sticky="W", padx=5, pady=5)

# ROI_button = tk.Button(
    # ROI_frame,
    # command=lambda: askfile(var_mask_roi),
    # text="Browse ...", font=font_standard
    # )
# ROI_button.grid(row=0, column=10, sticky='W', padx=5, pady=5)

# # PCA feature extraction method ###
# PCA_frame = tk.Frame(feature_frame)

# PCA_whiten = tk.BooleanVar()
# PCA_chk_whiten = tk.Checkbutton(
    # PCA_frame, text="whiten", variable=PCA_whiten, font=font_standard)
# PCA_chk_whiten.grid(row=3, column=1)


# # RBM feature extraction method ###
# RBM_frame = tk.Frame(feature_frame)

# # learning rate
# RBM_lr = tk.StringVar()
# RBM_lr.set('0.1')
# RBM_lr_label = tk.Label(
    # RBM_frame, text="Learning rate:", font=font_standard
    # )
# RBM_lr_label.grid(row=0, column=0, padx=5, pady=2)
# RBM_lr_entry = tk.Entry(
    # RBM_frame, width=5, font=font_standard, textvariable=RBM_lr
    # )
# RBM_lr_entry.grid(row=0, column=1, pady=2)

# # batch size
# RBM_bs = tk.StringVar()
# RBM_bs.set('10')
# RBM_bs_label = tk.Label(
    # RBM_frame, text="Batch size:", font=font_standard
    # )
# RBM_bs_label.grid(row=0, column=2, padx=5, pady=2)
# RBM_bs_entry = tk.Entry(
    # RBM_frame, width=5, font=font_standard, textvariable=RBM_bs
    # )
# RBM_bs_entry.grid(row=0, column=3, pady=2)

# # n_iter
# RBM_ni = tk.StringVar()
# RBM_ni.set('10')
# RBM_ni_label = tk.Label(
    # RBM_frame, text="n_iter:", font=font_standard
    # )
# RBM_ni_label.grid(row=0, column=4, padx=5, pady=2)
# RBM_ni_entry = tk.Entry(
    # RBM_frame, width=4, font=font_standard, textvariable=RBM_ni
    # )
# RBM_ni_entry.grid(row=0, column=5, pady=2)

# # verbose
# RBM_ver = tk.StringVar()
# RBM_ver.set('0')
# RBM_ver_label = tk.Label(
    # RBM_frame, text="verbose:", font=font_standard
    # )
# RBM_ver_label.grid(row=1, column=0, sticky='E', padx=5, pady=2)
# RBM_ver_entry = tk.Entry(
    # RBM_frame, width=5, font=font_standard, textvariable=RBM_ver
    # )
# RBM_ver_entry.grid(row=1, column=1, pady=2)

# # random state
# RBM_rs = tk.StringVar()
# RBM_rs.set('None')
# RBM_rs_label = tk.Label(
    # RBM_frame, text="n_iter:", font=font_standard
    # )
# RBM_rs_label.grid(row=1, column=2, padx=5, pady=2)
# RBM_rs_entry = tk.Entry(
    # RBM_frame, width=5, font=font_standard, textvariable=RBM_rs
    # )
# RBM_rs_entry.grid(row=1, column=3, pady=2)


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
var_classifier_type.set("Feedforward Neural Network - simple python script (fnn simple)")

classifier_option = tk.OptionMenu(
    classifier_frame, var_classifier_type,
    "Feedforward Neural Network - simple python script (fnn simple)",
    "Feedforward Neural Network - theano version (fnn theano)",
    "Convolutional Neural Network (CNN)",
    "Supported Vector Classifier (SVC)",
    "Linear Discriminant Analysis (LDA)",
    "Quadratic Discriminant Analysis (QDA)",
    command=classifier_choose
    )
classifier_option.grid(row=1, stick='WS', padx='20')
classifier_option.configure(font=font_standard)


# fANN parameters ###
fANN_frame = tk.Frame(classifier_frame)

# hidden neurons
fANN_epochs = tk.IntVar()
fANN_epochs.set(100)
fANN_epochs_label = tk.Label(
    fANN_frame, text="Epochs:", font=font_standard
    )
fANN_epochs_label.grid(row=0, column=0, padx=5, pady=2)
fANN_epochs_entry = tk.Entry(
    fANN_frame, width=5, font=font_standard, textvariable=fANN_epochs
    )
fANN_epochs_entry.grid(row=0, column=1, pady=2)

# hidden neurons
fANN_hn = tk.IntVar()
fANN_hn.set(30)
fANN_hn_label = tk.Label(
    fANN_frame, text="Hidden neurons:", font=font_standard
    )
fANN_hn_label.grid(row=0, column=2, padx=5, pady=2)
fANN_hn_entry = tk.Entry(
    fANN_frame, width=5, font=font_standard, textvariable=fANN_hn
    )
fANN_hn_entry.grid(row=0, column=3, pady=2)

# eta learning rate
fANN_eta = tk.DoubleVar()
fANN_eta.set(3.0)
fANN_eta_label = tk.Label(
    fANN_frame, text="learning rate (eta):", font=font_standard
    )
fANN_eta_label.grid(row=0, column=4, padx=5, pady=2)
fANN_eta_entry = tk.Entry(
    fANN_frame, width=5, font=font_standard, textvariable=fANN_eta
    )
fANN_eta_entry.grid(row=0, column=5, pady=2)

# minibatch size
fANN_ms = tk.IntVar()
fANN_ms.set(10)
fANN_ms_label = tk.Label(
    fANN_frame, text="minibatch size:", font=font_standard
    )
fANN_ms_label.grid(row=0, column=6, padx=5, pady=2)
fANN_ms_entry = tk.Entry(
    fANN_frame, width=4, font=font_standard, textvariable=fANN_ms
    )
fANN_ms_entry.grid(row=0, column=7, pady=2)

# ### Cross validate
perform_frame = tk.LabelFrame(
    classifier_frame, text='Performance metrics', font=font_standard
    )
perform_frame.grid(
    row=3, column=0, sticky='W',
    padx='20', pady='20', ipady='10'
    )

perform_options = [
    'LeavePOut (LPO randomly)', 'Shuffle Split',
    'KFold', 'Train Test Split'
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


###########################################################################
#
#     OUTPUT FRAME
#
###########################################################################

var_output = tk.StringVar()

output_frame = tk.Frame(note)

note.add(output_frame, text='Output and logs', padding=20)
note.grid(
    row=0, column=0, columnspan=10, sticky='WE',
    padx=20, pady=10, ipadx=10, ipady=10
    )

# Output ###################################################################
outFileLbl = tk.Label(
    output_frame, text="Output directory:", font=font_standard
    )
outFileLbl.grid(row=0, column=0, columnspan=2, sticky='E', padx=5, pady=5)

outFileTxt = tk.Entry(
    output_frame, width=40, font=font_standard, textvariable=var_output
    )
outFileTxt.grid(row=0, column=2, columnspan=8, sticky="WE", pady=5)

outFileBtn = tk.Button(
    output_frame,
    command=askdir_output,
    text="Browse ...", font=font_standard
    )
outFileBtn.grid(row=0, column=10, sticky='W', padx=5, pady=5)


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
