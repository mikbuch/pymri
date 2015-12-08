'''
name: workflow_flirt.py
type: script

FSL FLIRT apply transform to series of files.
Use DataGrabber and iterables.
'''


###############################################################################
#
#      VARIABLES, PATHS
#
###############################################################################

# base_directory = '/Users/AClab/Documents/mikbuch/Maestro_Project1'
base_directory = '/tmp/Maestro_Project1'

subject_template = 'GK'


###############################################################################
#
#      SUBJECTS NAMES - FUNCTION
#
###############################################################################

import os


def get_subject_names(base_directory, subject_template):
    subjects_list = os.listdir(base_directory)

    subjects_list = [sub for sub in subjects_list if subject_template in sub]

    return subjects_list


###############################################################################
#
#      CREATE MAIN WORKFLOW
#
###############################################################################

from nipype.pipeline import Workflow, Node
import nipype.interfaces.utility as util

flirt_apply_all_subs = Workflow(name='flirt_apply_all_subs')

inputsub = Node(
    interface=util.IdentityInterface(
        fields=['sub']
        ),
    name='inputsub'
    )
# inputsub.inputs.sub = ['GK011RZJA', 'GK012OHPA']
# inputsub.iterables = ('sub', ['GK011RZJA', 'GK012OHPA'])
inputsub.iterables = (
    'sub', get_subject_names(base_directory, subject_template)
    )

inputhand = Node(
    interface=util.IdentityInterface(
        fields=['hand']
        ),
    name='inputhand'
    )
inputhand.iterables = ('hand', ['Left', 'Right'])


# input_roi_standard = Node(
    # interface=util.IdentityInterface(
        # fields=['roi_standard']
        # ),
    # name='input_roi_standard'
    # )
# input_roi_standard.iterables = ('hand', ['Left', 'Right'])


inputnode = Node(
    interface=util.IdentityInterface(
        fields=['in_sub', 'in_hand']
        ),
    name='inputspec'
    )
# inputnode.inputs.in_sub = 'GK011RZJA'



###############################################################################
#
#     DATA GRABBER NODES
#
###############################################################################

from nipype.interfaces.io import DataGrabber
from os.path import abspath as opap


# ### ROIs IN STANDARD SPACE (the ones to be transformed, flirt's 'in_file') ##
dg_roi_standard = Node(DataGrabber(
    infields=['subject_id', 'hand'],
    outfields=['roi_stanard']),
    name='roi_standard_source'
    )
dg_roi_standard.inputs.base_directory = opap(base_directory)
dg_roi_standard.inputs.template = 'mvpa/ROIs_standrad_space/*.nii*'
dg_roi_standard.inputs.sort_filelist = True


# # ### REFERENCE IMAGES (size - dimensions) ####################################
# dg_reference = Node(DataGrabber(
    # infields=['subject_id', 'hand'], outfields=['func']), name='reference'
    # )
# dg_reference.inputs.base_directory = opap(base_directory)
# dg_reference.inputs.template = 'mvpa/ROIs_standrad_space/*.nii*'
# dg_reference.inputs.sort_filelist = True


# ### MOLOCH ####################################
dg_moloch = Node(DataGrabber(
    infields=['subject_id', 'hand'],
    outfields=['reference', 'matrix']),
    name='moloch'
    )
dg_moloch.inputs.base_directory = opap(base_directory)
dg_moloch.inputs.template = dict(
    reference=\
        '%s/Analyzed_data/MainExp_%sHand_Run-1.feat/reg/example_func.nii.gz',
    matrix=''
        '%s/Analyzed_data/MainExp_%sHand_Run-1.feat/reg/standard2example*',
    )
dg_moloch.inputs.template_args = dict(
    reference=[['subject_id', 'hand']],
    metrix=[['subject_id', 'hand']]
    )
dg_moloch.inputs.sort_filelist = True


flirt_apply_all_subs.connect(
    inputsub, 'sub',
    dg_roi_standard, 'subject_id'
    )
flirt_apply_all_subs.connect(
    inputhand, 'hand',
    dg_roi_standard, 'hand'
    )

flirt_apply_all_subs.connect(
    inputsub, 'sub',
    dg_moloch, 'subject_id'
    )
flirt_apply_all_subs.connect(
    inputhand, 'hand',
    dg_moloch, 'hand'
    )

# ###############################################################################
# #
# #     FLIRT APPLY NODE
# #
# ###############################################################################

# from nipype.interfaces import fsl

# flt = Node(fsl.FLIRT(), name='flirt')
# # flt.inputs.in_file = 'thresh_zstat1.nii.gz'
# # flt.inputs.reference = 'example_func_lh.nii.gz'
# flt.inputs.output_type = 'NIFTI_GZ'

# flt.inputs.interp = 'nearestneighbour'

# # flt.inputs.in_matrix_file = 'standard2example_func.mat'
# flt.inputs.apply_xfm = True

# # flt.inputs.out_file = 'bh_pC-cT_flirted.nii.gz'



# ###############################################################################
# #
# #     DATA SINK NODE
# #
# ###############################################################################

# def add_two_strings(subject, hand):
    # return subject + '/' + hand + '_Hand/'

# from nipype.interfaces.utility import Function
# add_two_strings_node = Node(
    # interface=Function(
        # input_names=["subject", "hand"],
        # output_names=["sub_hand_name"],
        # function=add_two_strings
        # ),
    # name='ats'
    # )

# from nipype.interfaces.io import DataSink

# datasink = Node(interface=DataSink(), name='datasink')
# datasink.inputs.base_directory = opap('/tmp/sinks')
# datasink.inputs.parameterization = False

# '''
# If iterating trought hands will be available
# '''
# meta.connect(
    # inputsub, 'sub',
    # add_two_strings_node, 'subject'
    # )
# meta.connect(
    # inputhand, 'hand',
    # add_two_strings_node, 'hand'
    # )
# meta.connect(
    # add_two_strings_node, 'sub_hand_name',
    # datasink, 'container'
    # )


# meta.connect(
    # featreg_merge, 'merge.merged_file',
    # # datasink, ds.inputs.subject_id + '/' + ds.inputs.hand + '_Hand/mvpa'
    # datasink, 'mvpa'
    # )
# # meta.connect(inputsub, 'sub', datasink, 'container')


###############################################################################
#
#     BASE_DIR, GRAPH, AND RUN
#
###############################################################################

meta.base_dir = '/tmp/working_dir'
meta.write_graph("graph.dot")

# Uncomment the last line to run workflow.
# featreg_merge.run()
