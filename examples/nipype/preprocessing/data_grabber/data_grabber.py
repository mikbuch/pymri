from nipype.interfaces.io import DataGrabber
from nipype.pipeline import Node
from os.path import abspath as opap

ds = Node(DataGrabber(
    infields=['subject_id'], outfields=['func']), name='datasource'
    )
ds.inputs.base_directory = opap('ds105')
ds.inputs.template = '%s/BOLD/task001*/bold.nii.gz'
ds.inputs.sort_filelist = True
ds.inputs.subject_id = 'sub001'

functional_input = ds.run().outputs
input_files = functional_input.get()['func']

print input_files
