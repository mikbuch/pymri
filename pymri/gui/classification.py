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
        ROI_frame.grid(row=2, column=0, columnspan=10, padx=20, pady=10)
    else:
        ROI_frame.grid_forget()

    if sel_ext_method == 'Principal Component Analysis':
        PCA_frame.grid(row=2, column=0, columnspan=10, padx=20, pady=10)
    else:
        PCA_frame.grid_forget()

    if sel_ext_method == 'Restricted Boltzmann Machine':
        RBM_frame.grid(row=2, column=0, columnspan=10, padx=20, pady=10)
    else:
        RBM_frame.grid_forget()


def classifier_choose(classifier_type):
    if classifier_type == 'feedforward Artificial Neural Network (fANN)':
        fANN_frame.grid(row=2, column=0, columnspan=10, padx=20, pady=10)
    else:
        fANN_frame.grid_forget()

def perform_choose(perform_type):
    if perform_type == 'LeavePOut (randomly)':
        LPO_frame.grid(row=2, column=0, columnspan=10, padx=20, pady=10)
    else:
        LPO_frame.grid_forget()

def cv_choose(cv_type):
    pass


def askfile_bold():
    file_chosen = tkFileDialog.askopenfilename()
    var_bold.set(file_chosen)

def askfile_attr():
    file_chosen = tkFileDialog.askopenfilename()
    var_attr.set(file_chosen)

def askfile_attr_lit():
    file_chosen = tkFileDialog.askopenfilename()
    var_attr_lit.set(file_chosen)

def askfile_mask_brain():
    file_chosen = tkFileDialog.askopenfilename()
    var_mask_brain.set(file_chosen)

def askdir_output():
    dir_chosen = tkFileDialog.askdirectory()
    var_output.set(dir_chosen)

def perform():
    # TODO: it is better to create a list of commands
    # because program can go through all code in search for errors

    # Load data ###
    
    from pymri.dataset.datasets import DatasetManager2
    # dataset settings
    ds = DatasetManager2(
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
        normalize = var_norm_load.get(),
        nnadl = True,
        )
    # load data
    ds.load_data()

    import pdb
    pdb.set_trace()

    # TODO: more universal function needed
    # select feature reduction method
    ds.feature_reduction(roi_selection='SelectKBest', k_features=k_features)
    # ds.feature_reduction(roi_selection='/amu/master/nifti/bold/roi_mask_plan.nii.gz')
    # TODO: this ifi var_features is not set,
    # TODO: else take at most var_features from mask
    k_features = ds.X_processed.shape[1]


    # Create classifier ###

    # Perform classification ###

    #


def save():
    import csv
    with open('pymri_settings.log', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['Bold image'])
        spamwriter.writerow([var_bold.get()])
        spamwriter.writerow(['Attributes text file'])
        spamwriter.writerow([var_attr.get()])
        spamwriter.writerow(['Attributes literal text file'])
        spamwriter.writerow([var_attr_lit.get()])
        spamwriter.writerow(['Mask image'])
        spamwriter.writerow([var_mask_brain.get()])
        spamwriter.writerow(['Condition (class) 00'])
        spamwriter.writerow([var_class_00.get()])
        spamwriter.writerow(['Condition (class) 01'])
        spamwriter.writerow([var_class_01.get()])
        spamwriter.writerow(['Normalize load'])
        spamwriter.writerow([var_norm_load.get()])
        spamwriter.writerow(['n features to extract/select'])
        spamwriter.writerow([var_k_features.get()])
        spamwriter.writerow(['Output'])
        spamwriter.writerow([var_output.get()])
    print('Successfully saved!')

def load():
    import csv
    read_data = []
    with open('pymri_settings.log', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            read_data.append(row)
    var_bold.set(read_data[1][0])
    var_attr.set(read_data[3][0])
    var_attr_lit.set(read_data[5][0])
    var_mask_brain.set(read_data[7][0])
    var_class_00.set(read_data[9][0])
    var_class_01.set(read_data[11][0])
    var_norm_load.set(read_data[13][0])

    var_k_features.set(read_data[15][0])

    var_output.set(read_data[-1][0])
    print('Successfully loaded!')

def quit():
    root.destroy()


root = tk.Tk()
font_size=12
standard_font = tkFont.Font(size=font_size)

getFld = tk.IntVar()
var_ev = tk.IntVar() 

var_bold = tk.StringVar(None)
var_attr = tk.StringVar(None)
var_attr_lit = tk.StringVar(None)
var_mask_brain = tk.StringVar(None)

root.wm_title('PyMRI main window')


###########################################################################
#
#     LOAD DATA FRAME
#
###########################################################################

load_data = tk.LabelFrame(
    root, borderwidth = font_size/6,
    text=" Load data: ", font=standard_font,
    )
load_data.grid(
    row=0, column=0, columnspan=10, sticky='WE',
    padx=20, pady=10, ipadx=10, ipady=10
    )

# Input ###################################################################

# file ####
inFileLbl = tk.Label(
        load_data, text="Input bold file:", font=standard_font
    )
inFileLbl.grid(row=0, column=0, columnspan=2, sticky='E', padx=5, pady=5)

inFileTxt = tk.Entry(
    load_data, width=40, font=standard_font, textvariable=var_bold
    )
inFileTxt.grid(row=0, column=2, columnspan=8, sticky="WE", pady=5)

inFileBtn = tk.Button(
    load_data,
    command=askfile_bold,
    text="Browse ...", font=standard_font
    )
inFileBtn.grid(row=0, column=10, sticky='W', padx=5, pady=5)

# attributes ####
inAttrLbl = tk.Label(
    load_data, text="Attributs file:", font=standard_font
    )
inAttrLbl.grid(row=1, column=0, columnspan=2, sticky='E', padx=5, pady=5)

inAttrTxt = tk.Entry(
    load_data, width=40, font=standard_font, textvariable=var_attr
    )
inAttrTxt.grid(row=1, column=2, columnspan=8, sticky="WE", pady=5)

inAttrBtn = tk.Button(
    load_data,
    command=askfile_attr,
    text="Browse ...", font=standard_font
    )
inAttrBtn.grid(row=1, column=10, sticky='W', padx=5, pady=5)

# attributes literal ####
inLiterLbl = tk.Label(
    load_data, text="Attributes literal file:", font=standard_font
    )
inLiterLbl.grid(row=2, column=0, columnspan=2, sticky='E', padx=5, pady=5)

inLiterTxt = tk.Entry(
    load_data, width=40, font=standard_font, textvariable=var_attr_lit
    )
inLiterTxt.grid(row=2, column=2, columnspan=8, sticky="WE", pady=5)

inLiterBtn = tk.Button(
    load_data,
    command=askfile_attr_lit,
    text="Browse ...", font=standard_font
    )
inLiterBtn.grid(row=2, column=10, sticky='W', padx=5, pady=5)

# mask brain regions ####
inMaskBLbl = tk.Label(
    load_data, text="Mask brain region:", font=standard_font
    )
inMaskBLbl.grid(row=3, column=0, columnspan=2, sticky='E', padx=5, pady=5)

inMaskBTxt = tk.Entry(
    load_data, width=40, font=standard_font, textvariable=var_mask_brain
    )
inMaskBTxt.grid(row=3, column=2, columnspan=8, sticky="WE", pady=5)

inMaskBBtn = tk.Button(
    load_data,
    command=askfile_mask_brain,
    text="Browse ...", font=standard_font
    )
inMaskBBtn.grid(row=3, column=10, sticky='W', padx=5, pady=5)

sep_paths_classes = ttk.Separator(load_data, orient=tk.HORIZONTAL)
sep_paths_classes.grid(row=4, column=1, columnspan=9, sticky='EW', pady=10) 


# classes (contrasts) get ###
# class_00
var_class_00 = tk.StringVar()

constrast_00_lbl = tk.Label(
    load_data, text="Class 00:", font=standard_font
    )
constrast_00_lbl.grid(
    row=5, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

constast_00_txt = tk.Entry(
    load_data, width=40, font=standard_font, textvariable=var_class_00
    )
constast_00_txt.grid(row=5, column=2, columnspan=10, sticky="WE", pady=5)

var_class_01 = tk.StringVar()

# class_01
var_class_01 = tk.StringVar()
constrast_01_lbl = tk.Label(
    load_data, text="Class 01:", font=standard_font
    )
constrast_01_lbl.grid(
    row=6, column=0, columnspan=2, sticky='E', padx=5, pady=5
    )

constast_01_txt = tk.Entry(
    load_data, width=40, font=standard_font, textvariable=var_class_01
    )
constast_01_txt.grid(row=6, column=2, columnspan=10, sticky="WE", pady=5)

# Normalize data ###
var_norm_load = tk.IntVar() 

norm_load_chk = tk.Checkbutton(
           load_data,
           text="Normalize data", onvalue=1, offvalue=0,
           variable=var_norm_load,
           font=standard_font
           )
norm_load_chk.grid(
    row=7, column=0, padx=10, pady=10, sticky='W'
    )


###########################################################################
#
#     FEATURE SELECTION/EXTRACTION FRAME
#
###########################################################################

# Frame Setup #############################################################
feature = tk.LabelFrame(
    root, borderwidth = font_size/6,
    text=" Feature selection/extraction: ", font=standard_font,
    )
feature.grid(
    row=1, column=0, columnspan=10, sticky='WE',
    padx=20, pady=10, ipadx=10, ipady=10
    )

# Fill frame ##############################################################

var_k_features = tk.IntVar()

k_features_frame = tk.Frame(
    feature, borderwidth = font_size/6
    )
k_features_frame.grid(
    padx=20, pady=20, ipadx=10, ipady=10
    )

k_features_label = tk.Label(
    k_features_frame, text="Extract/select n features:", font=standard_font
    )
k_features_label.grid(row=0, column=0, sticky='W')

k_features_entry = tk.Entry(
    k_features_frame, width=4, textvariable=var_k_features, font=standard_font
    )
k_features_entry.grid(row=0, column=1)


feature_method = tk.StringVar(feature)
feature_method.set("SelectKBest")

feature_option = tk.OptionMenu(
     feature, feature_method,
     "SelectKBest",
     "ROI mask",
     "Principal Component Analysis",
     "Restricted Boltzmann Machine",
     command=feature_choose
     )
feature_option.grid(row=1, stick='W', padx='20')
feature_option.configure(font=standard_font) 

# ROI mask feature selection method ###
ROI_frame = tk.Frame(feature)

var_roi = tk.StringVar()

ROI_label = tk.Label(
    ROI_frame, text="Mask file:", font=standard_font
    )
ROI_label.grid(row=0, column=0, sticky='E', padx=5, pady=5)

ROI_entry = tk.Entry(
    ROI_frame, width=40, font=standard_font, textvariable=var_roi
    )
ROI_entry.grid(row=0, column=1, columnspan=8, sticky="W", padx=5, pady=5)

ROI_button = tk.Button(
    ROI_frame,
    command=askfile_bold,
    text="Browse ...", font=standard_font
    )
ROI_button.grid(row=0, column=10, sticky='W', padx=5, pady=5)

# PCA feature extraction method ###
PCA_frame = tk.Frame(feature)

PCA_whiten = tk.BooleanVar()
PCA_chk_whiten = tk.Checkbutton(
    PCA_frame, text="whiten", variable=PCA_whiten, font=standard_font)
PCA_chk_whiten.grid(row=3, column=1)


# RBM feature extraction method ###
RBM_frame = tk.Frame(feature)

# learning rate
RBM_lr = tk.StringVar()
RBM_lr.set('0.1')
RBM_lr_label = tk.Label(
    RBM_frame, text="Learning rate:", font=standard_font
    )
RBM_lr_label.grid(row=0, column=0, padx=5, pady=2)
RBM_lr_entry = tk.Entry(
    RBM_frame, width=5, font=standard_font, textvariable=RBM_lr
    )
RBM_lr_entry.grid(row=0, column=1, pady=2)

# batch size
RBM_bs = tk.StringVar()
RBM_bs.set('10')
RBM_bs_label = tk.Label(
    RBM_frame, text="Batch size:", font=standard_font
    )
RBM_bs_label.grid(row=0, column=2, padx=5, pady=2)
RBM_bs_entry = tk.Entry(
    RBM_frame, width=5, font=standard_font, textvariable=RBM_bs
    )
RBM_bs_entry.grid(row=0, column=3, pady=2)

# n_iter
RBM_ni = tk.StringVar()
RBM_ni.set('10')
RBM_ni_label = tk.Label(
    RBM_frame, text="n_iter:", font=standard_font
    )
RBM_ni_label.grid(row=0, column=4, padx=5, pady=2)
RBM_ni_entry = tk.Entry(
    RBM_frame, width=4, font=standard_font, textvariable=RBM_ni
    )
RBM_ni_entry.grid(row=0, column=5, pady=2)

# verbose
RBM_ver = tk.StringVar()
RBM_ver.set('0')
RBM_ver_label = tk.Label(
    RBM_frame, text="verbose:", font=standard_font
    )
RBM_ver_label.grid(row=1, column=0, sticky='E', padx=5, pady=2)
RBM_ver_entry = tk.Entry(
    RBM_frame, width=5, font=standard_font, textvariable=RBM_ver
    )
RBM_ver_entry.grid(row=1, column=1, pady=2)

# random state
RBM_rs = tk.StringVar()
RBM_rs.set('None')
RBM_rs_label = tk.Label(
    RBM_frame, text="n_iter:", font=standard_font
    )
RBM_rs_label.grid(row=1, column=2, padx=5, pady=2)
RBM_rs_entry = tk.Entry(
    RBM_frame, width=5, font=standard_font, textvariable=RBM_rs
    )
RBM_rs_entry.grid(row=1, column=3, pady=2)

# learning_rate=0.1, batch_size=10, n_iter=10, verbose=0, random_state=None


###########################################################################
#
#     CLASSIFIER FRAME
#
###########################################################################

# Frame Setup #############################################################
classifier_frame = tk.LabelFrame(
    root, borderwidth = font_size/6,
    text=" Classifier settings: ", font=standard_font,
    )
classifier_frame.grid(
    row=2, column=0, columnspan=10, sticky='WE',
    padx=20, pady=10, ipadx=10, ipady=10
    )

# Fill frame ##############################################################
classifier_type = tk.StringVar()
classifier_type.set("feedforward Artificial Neural Network (fANN)")

classifier_option = tk.OptionMenu(
     classifier_frame, classifier_type,
     "feedforward Artificial Neural Network (fANN)",
     "Convolutional Neural Network (CNN)",
     "Supported Vector Classifier (SVC)",
     "Linear Discriminant Analysis (LDA)",
     "Quadratic Discriminant Analysis (QDA)",
     command=classifier_choose
     )
classifier_option.grid(row=1, stick='WS', padx='20')
classifier_option.configure(font=standard_font) 


# fANN parameters ###
fANN_frame = tk.Frame(classifier_frame)

# hidden neurons
fANN_hn = tk.StringVar()
fANN_hn.set('30')
fANN_hn_label = tk.Label(
    fANN_frame, text="Hidden neurons:", font=standard_font
    )
fANN_hn_label.grid(row=0, column=0, padx=5, pady=2)
fANN_hn_entry = tk.Entry(
    fANN_frame, width=5, font=standard_font, textvariable=fANN_hn
    )
fANN_hn_entry.grid(row=0, column=1, pady=2)

# eta learning rate
fANN_eta = tk.StringVar()
fANN_eta.set('3')
fANN_eta_label = tk.Label(
    fANN_frame, text="learning rate (eta):", font=standard_font
    )
fANN_eta_label.grid(row=0, column=2, padx=5, pady=2)
fANN_eta_entry = tk.Entry(
    fANN_frame, width=5, font=standard_font, textvariable=fANN_eta
    )
fANN_eta_entry.grid(row=0, column=3, pady=2)

# minibatch size
fANN_ms = tk.StringVar()
fANN_ms.set('10')
fANN_ms_label = tk.Label(
    fANN_frame, text="minibatch size:", font=standard_font
    )
fANN_ms_label.grid(row=0, column=4, padx=5, pady=2)
fANN_ms_entry = tk.Entry(
    fANN_frame, width=4, font=standard_font, textvariable=fANN_ms
    )
fANN_ms_entry.grid(row=0, column=5, pady=2)

# ### Cross validate
perform_frame = tk.LabelFrame(
    classifier_frame, text='Performance metrics', font=standard_font
    )
perform_frame.grid(
    row=3, column=0, sticky='W',
    padx='20', pady='20', ipady='10'
    )

perform_options = [
    'LeavePOut (randomly)', 'Shuffle Split',
    'KFold', 'Train Test Split'
    ]

perform_type = tk.StringVar(feature)
perform_type.set(perform_options[0])

perform_options_menu = tk.OptionMenu(
     perform_frame, perform_type,
     *perform_options,
     command=perform_choose
     )
perform_options_menu.grid(row=1, column=0, stick='W', padx='20')
perform_options_menu.configure(font=standard_font) 

# LPO parameters ###
LPO_frame = tk.Frame(perform_frame)

# leave p
LPO_p = tk.StringVar()
LPO_p.set('0.25')
LPO_p_label = tk.Label(
    LPO_frame, text="p (proportion or number):", font=standard_font
    )
LPO_p_label.grid(row=0, column=0, padx=5, pady=2)
LPO_p_entry = tk.Entry(
    LPO_frame, width=5, font=standard_font, textvariable=LPO_p
    )
LPO_p_entry.grid(row=0, column=1, pady=2)

# n_times perform
LPO_n_times = tk.StringVar()
LPO_n_times.set('6')
LPO_n_times_label = tk.Label(
        LPO_frame, text="LeavePOut k times:", font=standard_font
    )
LPO_n_times_label.grid(row=0, column=2, padx=5, pady=2)
LPO_n_times_entry = tk.Entry(
    LPO_frame, width=5, font=standard_font, textvariable=LPO_n_times
    )
LPO_n_times_entry.grid(row=0, column=3, pady=2)


###########################################################################
#
#     OUTPUT FRAME
#
###########################################################################

var_output = tk.StringVar(None)

output = tk.LabelFrame(
    root, borderwidth = font_size/6,
    text=" Output: ", font=standard_font,
    )
output.grid(
    row=3, column=0, columnspan=10, sticky='WE',
    padx=20, pady=10, ipadx=10, ipady=10
    )

# Output ###################################################################
outFileLbl = tk.Label(
    output, text="Output directory:", font=standard_font
    )
outFileLbl.grid(row=0, column=0, columnspan=2, sticky='E', padx=5, pady=5)

outFileTxt = tk.Entry(
    output, width=40, font=standard_font, textvariable=var_output
    )
outFileTxt.grid(row=0, column=2, columnspan=8, sticky="WE", pady=5)

outFileBtn = tk.Button(
    output,
    command=askdir_output,
    text="Browse ...", font=standard_font
    )
outFileBtn.grid(row=0, column=10, sticky='W', padx=5, pady=5)



###########################################################################
#
#     MENU
#
###########################################################################

options = tk.Frame(
    root, borderwidth = font_size/6
    )

options.grid(
    row=4, column=0, columnspan=10, sticky='WE',
    padx=10, pady=10, ipadx=10, ipady=10
    )

go = tk.Button(
    options,
    command=perform,
    text=" Go ", font=standard_font
    )
go.grid(row=4, column=0, padx=10, pady=10)

save = tk.Button(
    options,
    command=save,
    text="Save", font=standard_font
    )
save.grid(row=4, column=1, padx=10, pady=10)

load = tk.Button(
    options,
    command=load,
    text="Load", font=standard_font
    )
load.grid(row=4, column=2, padx=10, pady=10)

help = tk.Button(
    options,
    command=askdir_output,
    text="Help", font=standard_font
    )
help.grid(row=4, column=3, padx=10, pady=10)

quit = tk.Button(
    options,
    command=quit,
    text="Quit", font=standard_font
    )
quit.grid(row=4, column=4, padx=10, pady=10)



perform_choose(perform_type.get())
classifier_choose(classifier_type.get())
options.mainloop()
