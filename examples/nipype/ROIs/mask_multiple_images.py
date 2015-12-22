import os


###############################################################################
#
#      VARIABLES, PATHS
#
###############################################################################

'''
From where to take files to preprocess.
'''
# base_directory = \
#     '/Users/AClab/Documents/mikbuch/Maestro_Project1/mvpa/ROIs_standard_space'
# base_directory = '/tmp/Maestro_Project1/mvpa/ROIs_standard_space/'
base_directory = '/home/jesmasta/downloads/CEREBRAL_CORTEX_ROIs'

'''
Where to put workflow outputs that we need.
Desired path is the one from which we have taken files initially.
Specifying here other directory is considered debugging operation.
'''
# datasink_directory = base_directory
datasink_directory = '/tmp/sinks'

'''
Place where all files created and required by workflow will be stored.
'''
workflow_base_directory = '/tmp/ttt'

'''
Workflow-specific (mask_multiple) variable. Where to store processed files.
'''
processed_masks_directory = 'MNI152_masked'
# Get mask from FSLDIR location
mask_file = \
    os.environ['FSLDIR'] + '/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'


###############################################################################
#
#      MASK MULTIPLE WORKFLOW
#
###############################################################################

from nipype.pipeline import Workflow, Node
import nipype.interfaces.io as nio
from os.path import abspath as opap

mask_multiple = Workflow(name='mask_multiple')

grabber = nio.DataGrabber()
grabber.inputs.base_directory = opap(base_directory)
grabber.inputs.template = '*.nii*'
grabber.inputs.sort_filelist = True

grabbed = grabber.run()
rois_filelist = grabbed.outputs.outfiles

###############################################################################
#
#      MASK SINGLE WORKFLOW
#
###############################################################################
apply_mask_multiple = Workflow(name='apply_mask_multiple')

import nipype.interfaces.utility as util
inputnode = Node(
    interface=util.IdentityInterface(
        fields=['roi']
        ),
    name='inputspec'
    )
inputnode.iterables = ('roi', rois_filelist)

import nipype.interfaces.fsl as fsl

masknode = Node(fsl.maths.ApplyMask(), name='masknode')
masknode.inputs.mask_file = mask_file
apply_mask_multiple.connect(
    inputnode, 'roi',
    masknode, 'in_file',
    )

sinker = Node(nio.DataSink(), name='sinker')
sinker.inputs.base_directory = opap(datasink_directory)
sinker.inputs.parameterization = False
sinker.inputs.substitutions = [('', ''), ('_masked', '')]
apply_mask_multiple.connect(
    masknode, 'out_file',
    sinker, processed_masks_directory
    )

apply_mask_multiple.base_dir = workflow_base_directory
apply_mask_multiple.run()
