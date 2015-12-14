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
#      CREATE MAIN WORKFLOW
#
###############################################################################

from nipype.pipeline import Workflow, Node
import nipype.interfaces.utility as util

from pymri.utils.paths_dirs_info import get_subject_names

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


inputnode = Node(
    interface=util.IdentityInterface(
        fields=['in_sub', 'in_hand']
        ),
    name='inputspec'
    )
# inputnode.inputs.in_sub = 'GK011RZJA'


###############################################################################
#
#     INPUTNODE - STANDARD SPACE ROIs
#
###############################################################################

from pymri.utils.paths_dirs_info import get_roi_standard

input_roi_standard = Node(
    interface=util.IdentityInterface(
        fields=['roi_standard']
        ),
    name='input_roi_standard'
    )
input_roi_standard.iterables = (
    'roi_standard',
    get_roi_standard(base_directory + '/mvpa/ROIs_standard_space')
    )


###############################################################################
#
#     DATA GRABBER NODES
#
###############################################################################

from nipype.interfaces.io import DataGrabber
from os.path import abspath as opap


# ### MOLOCH ####################################
dg_moloch = Node(DataGrabber(
    infields=['subject_id', 'hand'],
    outfields=['reference', 'matrix']),
    name='moloch'
    )
dg_moloch.inputs.base_directory = opap(base_directory)
dg_moloch.inputs.template = '*'
reg_dir_template = '%s/Analyzed_Data/MainExp_%sHand_Run-1.feat/reg/'
dg_moloch.inputs.field_template = dict(
    reference=reg_dir_template + 'example_func.nii.gz',
    matrix=reg_dir_template + 'standard2example*'
    )
dg_moloch.inputs.template_args = dict(
    reference=[['subject_id', 'hand']],
    matrix=[['subject_id', 'hand']]
    )
dg_moloch.inputs.sort_filelist = True


flirt_apply_all_subs.connect(
    inputsub, 'sub',
    dg_moloch, 'subject_id'
    )
flirt_apply_all_subs.connect(
    inputhand, 'hand',
    dg_moloch, 'hand'
    )

###############################################################################
#
#     FLIRT APPLY NODE
#
###############################################################################

from nipype.interfaces import fsl

flt = Node(fsl.FLIRT(), name='flirt')
flt.inputs.output_type = 'NIFTI_GZ'

flt.inputs.interp = 'nearestneighbour'
flt.inputs.apply_xfm = True


flirt_apply_all_subs.connect(
    input_roi_standard, 'roi_standard',
    flt, 'in_file'
    )


flirt_apply_all_subs.connect(
    dg_moloch, 'reference',
    flt, 'reference'
    )
flirt_apply_all_subs.connect(
    dg_moloch, 'matrix',
    flt, 'in_matrix_file'
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
datasink.inputs.base_directory = opap(base_directory)
# datasink.inputs.base_directory = opap('/tmp/sinks')
datasink.inputs.parameterization = False
datasink.inputs.substitutions = [('', ''), ('_flirt', '')]

'''
If iterating trought hands will be available
'''
flirt_apply_all_subs.connect(
    inputsub, 'sub',
    add_two_strings_node, 'subject'
    )
flirt_apply_all_subs.connect(
    inputhand, 'hand',
    add_two_strings_node, 'hand'
    )
flirt_apply_all_subs.connect(
    add_two_strings_node, 'sub_hand_name',
    datasink, 'container'
    )


flirt_apply_all_subs.connect(
    flt, 'out_file',
    datasink, 'mvpa/ROIs'
    )


###############################################################################
#
#     BASE_DIR, GRAPH, AND RUN
#
###############################################################################

flirt_apply_all_subs.base_dir = '/tmp/working_dir'
flirt_apply_all_subs.write_graph("graph.dot")

# Uncomment the last line to run workflow.
flirt_apply_all_subs.run()
