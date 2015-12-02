'''
name:
simple_flirt.py

type:
script

FSL FLIRT apply transform (simple example)
'''

from nipype.interfaces import fsl

flt = fsl.FLIRT()
flt.inputs.in_file = 'thresh_zstat1.nii.gz'
flt.inputs.reference = 'example_func_lh.nii.gz'
flt.inputs.output_type = 'NIFTI_GZ'

 flt.inputs.interp = 'nearestneighbour'

flt.inputs.in_matrix_file = 'standard2example_func.mat'
flt.inputs.apply_xfm = True

flt.inputs.out_file = 'bh_pC-cT_flirted.nii.gz'

print(flt.cmdline)
flt.run()

'''
check the results
fslview example_func_lh.nii.gz bh_flirted_tri.nii.gz -l Red-Yellow -b 0.0,4.0
'''
