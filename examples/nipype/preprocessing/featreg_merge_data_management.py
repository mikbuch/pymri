'''
featreg_merge_data_grabber.py
script

Perform Feat preprocessing on given data files and then merge ouputs.
Inputs are taken using DataGrabber interface.

'''

###############################################################################
#
#      CREATE MAIN WORKFLOW
#
###############################################################################

from nipype.pipeline import Workflow, Node

featreg_merge = Workflow(name='featreg_merge')


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


###############################################################################
#
#     DATA GRABBER NODE
#
###############################################################################

from nipype.interfaces.io import DataGrabber
from os.path import abspath as opap

base_directory = '/Users/AClab/Documents/mikbuch/Maestro_Project1'

ds = Node(DataGrabber(
    infields=['subject_id', 'hand'], outfields=['func']), name='datasource'
    )
ds.inputs.base_directory = opap(base_directory)
ds.inputs.template = '%s/%s_Hand/*.nii'
ds.inputs.sort_filelist = True
ds.inputs.subject_id = 'GK011RZJA'
ds.inputs.hand = 'Left'

'''
    To print the list of files being taken uncomment the following lines.
'''
#  functional_input = ds.run().outputs
#  input_files = functional_input.get()['func']
#  print input_files


###############################################################################
#
#     MERGE NODE
#
###############################################################################

featreg_merge.connect(ds, 'func', preproc, 'inputspec.func')

merge = Node(
    interface=fsl.utils.Merge(
        dimension='t',
        output_type='NIFTI_GZ',
        merged_file='merged.nii.gz'
        ),
    name='merge'
    )
featreg_merge.connect(
    preproc, 'outputspec.highpassed_files', merge, 'in_files'
    )

###############################################################################
#
#     DATA SINKER NODE
#
###############################################################################

from nipype.interfaces.io import DataSinker

datasink = Node(interface=DataSink(), name='datasink')
datasink.inputs.base_directory = opap('/tmp/sinks')

featreg_merge.connect(merge, 'merged_file', datasink, 'bold')


###############################################################################
#
#     BASE_DIR, GRAPH, AND RUN
#
###############################################################################

featreg_merge.base_dir = '/tmp/working_dir'
featreg_merge.write_graph("graph.dot")

featreg_merge.run()
