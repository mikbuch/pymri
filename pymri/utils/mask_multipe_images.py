###############################################################################
#
#      VARIABLES, PATHS
#
###############################################################################

'''
From where to take files to preprocess.
'''
# base_directory = '/Users/AClab/Documents/mikbuch/Maestro_Project1'
base_directory = '/tmp/Maestro_Project1'

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
workflow_base_directory = '/tmp/working_dir'

'''
Workflow-specific (mask_multipe) variable. Where to store processed files.
'''
processed_masks_directory = 'MNI152_masked'


###############################################################################
#
#      MASK MULTIPE WORKFLOW
#
###############################################################################

from nipype.pipeline import Workflow, Node
import nipype.interfaces.io as nio
mask_multipe = Workflow(name='mask_multipe')

datagrabber = Node(nio.DataGrabber(), name='datagrabber')

import nipype.interdaces.fsl as fsl
masknode = Node(fsl.maths.ApplyMask(), name='masknode')
mask_multipe.connect(
    datagrabber, 'out_file',
    masknode, 'in_file'
    )

datasink = Node(nio.DataSink(), name='datasink')
datasink.inputs.base_directory = opap(datasink_directory)
mask_multipe.connect(
    masknode, 'out_file',
    datasink, processed_masks_directory
    )
