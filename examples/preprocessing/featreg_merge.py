'''
featreg_merge.py
script

Perform Feat preprocessing on given data files and then merge ouputs.

'''

from nipype.workflows.fmri.fsl import create_featreg_preproc
import nipype.interfaces.fsl as fsl
from nipype.pipeline import Workflow, Node

# get filelist from file
nifti_filelist = open('nifti_filelist.txt').read().splitlines()

featreg_merge = Workflow(name='featreg_merge')

preproc = create_featreg_preproc(highpass=True, whichvol='mean')
preproc.inputs.inputspec.func = nifti_filelist
preproc.inputs.inputspec.fwhm = 0
preproc.inputs.inputspec.highpass = 128./(2*2.5)
# preproc.base_dir = '/tmp/pre/working_dir'
# preproc.run() 


merge = Node(
    interface=fsl.utils.Merge(
        dimension='t',
        output_type='NIFTI_GZ',
        merged_file='merged.nii.gz'
        ),
    name='merge'
    )
featreg_merge.connect(preproc, 'outputspec.highpassed_files', merge, 'in_files')

featreg_merge.base_dir='/tmp/pre/working_dir'

featreg_merge.write_graph("graph.dot")

featreg_merge.run()
