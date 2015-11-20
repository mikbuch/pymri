import Tkinter as tk
import tkFont
import tkFileDialog

import ttk

# TODO: Create working area frame, menu under it
# TODO: Scrolling left and right in the entry
# TODO: Wider entry
# TODO: Ctrl+a to select all in entry


def feature_choose(sel_ext_method):
    if sel_ext_method == 'ROI mask':
        ROI_frame.grid(row=3, column=0, columnspan=10, padx=20, pady=10)
    else:
        ROI_frame.grid_forget()

    if sel_ext_method == 'Principal Component Analysis':
        PCA_frame.grid(row=3, column=0, columnspan=10, padx=20, pady=10)
    else:
        PCA_frame.grid_forget()

    if sel_ext_method == 'Restricted Boltzmann Machine':
        RBM_frame.grid(row=3, column=0, columnspan=10, padx=20, pady=10)
    else:
        RBM_frame.grid_forget()


def classifier_choose(var_classifier_type):
    if var_classifier_type == 'feedforward Artificial Neural Network (fANN)':
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


def perform():
    # TODO: it is better to create a list of commands
    # because program can go through all code in search for errors

    # Load data ###

    from pymri.dataset.datasets import DatasetManager
    # dataset settings

    # in case we use fann as classifier we have to reshape data
    if 'fANN' in var_classifier_type.get():
        nnadl = True

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

    # select feature reduction method
    ds.feature_reduction(
        roi_selection=var_mask_roi.get(),
        k_features=var_k_features.get(),
        normalize = var_normalize.get()
        )
    # ds.feature_reduction(roi_selection=\
    # '/amu/master/nifti/bold/roi_mask_plan.nii.gz')

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

    mean_accuracy = accuracies.mean()
    print('\n\nmean accuracy: %f' % mean_accuracy)

    ###########################################################################
    #
    #   VISUALIZE
    #
    ###########################################################################
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


def save():
    import csv
    with open('settings.log', 'wb') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Bold image', var_bold.get()])
        writer.writerow(['Attributes text file', var_attr.get()])
        writer.writerow(['Attributes literal text file', var_attr_lit.get()])
        writer.writerow(['Mask image', var_mask_brain.get()])
        writer.writerow(['Condition (class) 00', var_class_00.get()])
        writer.writerow(['Condition (class) 01', var_class_01.get()])

        # FAETURE SELECTION/EXTRACTION VARIABLES
        writer.writerow(['k features to extract/select', var_k_features.get()])
        writer.writerow(['Normalize data', var_normalize.get()])
        writer.writerow(['Feature sel/ext method', var_feature_method.get()])

        writer.writerow(['ROI mask path', var_mask_roi.get()])


        # CLASSIFIER VARIABLES
        writer.writerow(['Classifier type', var_classifier_type.get()])

        writer.writerow(['fANN epochs', fANN_epochs.get()])
        writer.writerow(['fANN hidden neurons', fANN_hn.get()])
        writer.writerow(['fANN eta', fANN_eta.get()])
        writer.writerow(['fANN minibatch size', fANN_ms.get()])

        # PERFORMANCE VARIABLES
        writer.writerow(['Performance measure', var_perform_type.get()])

        writer.writerow(['LPO p', LPO_p.get()])
        writer.writerow(['LPO n times', LPO_n_times.get()])

        # OUTPUT VARIABLES
        writer.writerow(['Output', var_output.get()])

    print('Successfully saved!')


def load():
    import csv
    read_data = []
    with open('settings.log', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            read_data.append(row)

    var_bold.set(read_data[0][1])
    var_attr.set(read_data[1][1])
    var_attr_lit.set(read_data[2][1])
    var_mask_brain.set(read_data[3][1])
    var_class_00.set(read_data[4][1])
    var_class_01.set(read_data[5][1])

    # FAETURE SELECTION/REDUCTION VARIABLES
    var_k_features.set(read_data[6][1])
    var_normalize.set(read_data[7][1])
    var_feature_method.set(read_data[8][1])

    var_mask_roi.set(read_data[9][1])

    # CLASSIFIER VARIABLES
    var_classifier_type.set(read_data[10][1])

    fANN_epochs.set(read_data[11][1])
    fANN_hn.set(read_data[12][1])
    fANN_eta.set(read_data[13][1])
    fANN_ms.set(read_data[14][1])

    # PERFORMANCE VARIABLES
    var_perform_type.set(read_data[15][1])

    LPO_p.set(read_data[16][1])
    LPO_n_times.set(read_data[17][1])

    var_output.set(read_data[18][1])

    feature_choose(var_feature_method.get())
    classifier_choose(var_classifier_type.get())
    perform_choose(var_perform_type.get())

    print('Successfully loaded!')


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

# Input ###################################################################

# file ####
inFileLbl = tk.Label(
    load_data_frame, text="Input bold file:", font=font_standard
    )
inFileLbl.grid(row=0, column=0, columnspan=2, sticky='E', padx=5, pady=5)

inFileTxt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=var_bold
    )
inFileTxt.grid(row=0, column=2, columnspan=8, sticky="WE", pady=5)

inFileBtn = tk.Button(
    load_data_frame,
    command=lambda: askfile(var_bold),
    text="Browse ...", font=font_standard
    )
inFileBtn.grid(row=0, column=10, sticky='W', padx=5, pady=5)

# attributes ####
inAttrLbl = tk.Label(
    load_data_frame, text="Attributs file:", font=font_standard
    )
inAttrLbl.grid(row=1, column=0, columnspan=2, sticky='E', padx=5, pady=5)

inAttrTxt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=var_attr
    )
inAttrTxt.grid(row=1, column=2, columnspan=8, sticky="WE", pady=5)

inAttrBtn = tk.Button(
    load_data_frame,
    command=lambda: askfile(var_attr),
    text="Browse ...", font=font_standard
    )
inAttrBtn.grid(row=1, column=10, sticky='W', padx=5, pady=5)

# attributes literal ####
inLiterLbl = tk.Label(
    load_data_frame, text="Attributes literal file:", font=font_standard
    )
inLiterLbl.grid(row=2, column=0, columnspan=2, sticky='E', padx=5, pady=5)

inLiterTxt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=var_attr_lit
    )
inLiterTxt.grid(row=2, column=2, columnspan=8, sticky="WE", pady=5)

inLiterBtn = tk.Button(
    load_data_frame,
    command=lambda: askfile(var_attr_lit),
    text="Browse ...", font=font_standard
    )
inLiterBtn.grid(row=2, column=10, sticky='W', padx=5, pady=5)

# mask brain regions ####
inMaskBLbl = tk.Label(
    load_data_frame, text="Mask brain region:", font=font_standard
    )
inMaskBLbl.grid(row=3, column=0, columnspan=2, sticky='E', padx=5, pady=5)

inMaskBTxt = tk.Entry(
    load_data_frame, width=40, font=font_standard, textvariable=var_mask_brain
    )
inMaskBTxt.grid(row=3, column=2, columnspan=8, sticky="WE", pady=5)

inMaskBBtn = tk.Button(
    load_data_frame,
    command=lambda: askfile(var_mask_brain),
    text="Browse ...", font=font_standard
    )
inMaskBBtn.grid(row=3, column=10, sticky='W', padx=5, pady=5)

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



###########################################################################
#
#     FEATURE SELECTION/EXTRACTION FRAME
#
###########################################################################

# Frame Setup #############################################################
feature_frame = tk.Frame(note)
feature_frame.grid(
    row=1, column=0, columnspan=10, sticky='WE',
    padx=20, pady=10, ipadx=10, ipady=10
    )

note.add(feature_frame, text='Selection/reduction', padding=20)
note.grid(
    row=0, column=0, columnspan=10, sticky='WE',
    padx=20, pady=10, ipadx=10, ipady=10
    )


# Fill frame ##############################################################

var_k_features = tk.IntVar()

k_features_frame = tk.Frame(
    feature_frame, borderwidth=font_size/6
    )
k_features_frame.grid(
    padx=20, pady=20, ipadx=10, ipady=10
    )

k_features_label = tk.Label(
    k_features_frame, text="Extract/select n features:", font=font_standard
    )
k_features_label.grid(row=0, column=0, sticky='W')

k_features_entry = tk.Entry(
    k_features_frame, width=4, textvariable=var_k_features, font=font_standard
    )
k_features_entry.grid(row=0, column=1)

# Normalize data ###
var_normalize = tk.IntVar()

normalize_chk = tk.Checkbutton(
    feature_frame,
    text="Normalize data", onvalue=1, offvalue=0,
    variable=var_normalize,
    font=font_standard
    )
normalize_chk.grid(
    row=1, column=0, padx=10, pady=10, sticky='W'
    )

# Feature Selection/Extraction method ###
var_feature_method = tk.StringVar()
var_feature_method.set("SelectKBest")

feature_option = tk.OptionMenu(
    feature_frame, var_feature_method,
    "SelectKBest",
    "ROI mask",
    "Principal Component Analysis",
    "Restricted Boltzmann Machine",
    command=feature_choose
    )
feature_option.grid(row=2, stick='W', padx='20')
feature_option.configure(font=font_standard)

# ROI mask feature selection method ###
ROI_frame = tk.Frame(feature_frame)

var_mask_roi = tk.StringVar()

ROI_label = tk.Label(
    ROI_frame, text="Mask file:", font=font_standard
    )
ROI_label.grid(row=0, column=0, sticky='E', padx=5, pady=5)

ROI_entry = tk.Entry(
    ROI_frame, width=40, font=font_standard, textvariable=var_mask_roi
    )
ROI_entry.grid(row=0, column=1, columnspan=8, sticky="W", padx=5, pady=5)

ROI_button = tk.Button(
    ROI_frame,
    command=lambda: askfile(var_mask_roi),
    text="Browse ...", font=font_standard
    )
ROI_button.grid(row=0, column=10, sticky='W', padx=5, pady=5)

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

# learning_rate=0.1, batch_size=10, n_iter=10, verbose=0, random_state=None


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
var_classifier_type.set("feedforward Artificial Neural Network (fANN)")

classifier_option = tk.OptionMenu(
    classifier_frame, var_classifier_type,
    "feedforward Artificial Neural Network (fANN)",
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
    command=perform,
    text=" Go ", font=font_standard
    )
go.grid(row=4, column=0, padx=10, pady=10)

save = tk.Button(
    options,
    command=save,
    text="Save", font=font_standard
    )
save.grid(row=4, column=1, padx=10, pady=10)

load = tk.Button(
    options,
    command=load,
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