'''
sub_hand_preproc_merged.py
script

Perform Feat preprocessing on given data files and then merge ouputs.
Inputs are taken using DataGrabber interface.

Specify base_direcory and pattern (template) for the subject.

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

meta = Workflow(name='meta')

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


featreg_merge = Workflow(name='featreg_merge')

inputnode = Node(
    interface=util.IdentityInterface(
        fields=['in_sub', 'in_hand']
        ),
    name='inputspec'
    )
# inputnode.inputs.in_sub = 'GK011RZJA'


###############################################################################
#
#     DATA GRABBER NODE
#
###############################################################################

from nipype.interfaces.io import DataGrabber
from os.path import abspath as opap


ds = Node(DataGrabber(
    infields=['subject_id', 'hand'], outfields=['func']), name='datasource'
    )
ds.inputs.base_directory = opap(base_directory)
ds.inputs.template = '%s/%s_Hand/*.nii*'
ds.inputs.sort_filelist = True
# ds.inputs.subject_id = 'GK011RZJA'
# ds.inputs.hand = 'Left'

featreg_merge.connect(inputnode, 'in_hand', ds, 'hand')
featreg_merge.connect(inputnode, 'in_sub', ds, 'subject_id')

'''
    To print the list of files being taken uncomment the following lines.
'''
#  functional_input = ds.run().outputs
#  input_files = functional_input.get()['func']
#  print input_files

meta.connect(inputsub, 'sub', featreg_merge, 'inputspec.in_sub')
meta.connect(inputhand, 'hand', featreg_merge, 'inputspec.in_hand')

###############################################################################
#
#     CREATE FEAT REGISTRATION WORKFLOW NODE
#
###############################################################################

from nipype.workflows.fmri.fsl import create_featreg_preproc
import nipype.interfaces.fsl as fsl

preproc = create_featreg_preproc(highpass=True, whichvol='mean')
preproc.inputs.inputspec.fwhm = 0
preproc.inputs.inputspec.highpass = 128./(2*2.5)

featreg_merge.connect(ds, 'func', preproc, 'inputspec.func')


###############################################################################
#
#     MERGE NODE
#
###############################################################################

merge = Node(
    interface=fsl.utils.Merge(
        dimension='t',
        output_type='NIFTI_GZ',
        merged_file='bold.nii.gz'
        ),
    name='merge'
    )
featreg_merge.connect(
    preproc, 'outputspec.highpassed_files', merge, 'in_files'
    )


###############################################################################
#
#     DATA SINK NODE
#
###############################################################################

def add_two_strings(subject, hand):
    return subject + '/' + hand + '_Hand/'

from nipype.interfaces.utility import Function
add_two_strings_node = Node(
    interface=Function(
        input_names=["subject", "hand"],
        output_names=["sub_hand_name"],
        function=add_two_strings
        ),
    name='ats'
    )

from nipype.interfaces.io import DataSink

datasink = Node(interface=DataSink(), name='datasink')
datasink.inputs.base_directory = opap('/tmp/sinks')
datasink.inputs.parameterization = False

'''
If iterating trought hands will be available
'''
meta.connect(
    inputsub, 'sub',
    add_two_strings_node, 'subject'
    )
meta.connect(
    inputhand, 'hand',
    add_two_strings_node, 'hand'
    )
meta.connect(
    add_two_strings_node, 'sub_hand_name',
    datasink, 'container'
    )


meta.connect(
    featreg_merge, 'merge.merged_file',
    # datasink, ds.inputs.subject_id + '/' + ds.inputs.hand + '_Hand/mvpa'
    datasink, 'mvpa'
    )
# meta.connect(inputsub, 'sub', datasink, 'container')


###############################################################################
#
#     BASE_DIR, GRAPH, AND RUN
#
###############################################################################

meta.base_dir = '/tmp/working_dir'
meta.write_graph("graph.dot")

# Uncomment the last line to run workflow.
# featreg_merge.run()
