import subprocess as sp


def merge_nifti(nifti_locations, output_name='output_merge.nii.gz'):
    cmd = 'fslmerge -t '
    cmd += output_name + ' '

    nifti_list = \
        [line.rstrip('\n') for line in open(nifti_locations)]

    cmd += ' '.join(nifti_list) + ' '

    print(cmd)
    process = sp.Popen(cmd, stdout=sp.PIPE, shell=True)
    output = process.communicate()[0]
    print(output)
