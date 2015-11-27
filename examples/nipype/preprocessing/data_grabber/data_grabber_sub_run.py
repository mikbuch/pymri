fiom nipype.interfaces.io import DataGrabber
from nipype.pipeline import Node
from os.path import abspath as opap

ds = Node(DataGrabber(
    infields=['subject_id', 'run'], outfields=['func']), name='datasource'
    )
ds.inputs.base_directory = opap('ds105')
ds.inputs.template = '%s/BOLD/task001_run%03d/bold.nii.gz'
ds.inputs.sort_filelist = True
ds.inputs.subject_id = 'sub001'
ds.inputs.run = 1

functional_input = ds.run().outputs
input_files = functional_input.get()['func']

print input_files
