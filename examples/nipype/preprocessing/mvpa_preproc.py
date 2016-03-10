'''
name:
mvpa_preproc.py

type:
script

Perform Feat preprocessing on given data files and then merge ouputs.
Inputs are taken using DataGrabber interface.

Specify base_directory and pattern (template) for the subject.

NOTE:
Reference image for motion correction is common across all runs of particular
subject's hand. Available options:
run number => (mean, first, middle, last volume)
'''

# TODO: run number is fixed now, how to pass multipe arguments at connections

import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

from nipype.workflows.fmri.fsl import create_featreg_preproc
from nipype.workflows.fmri.fsl.preprocess import pickfirst, pickvol

from nipype.pipeline import Workflow, Node, MapNode
import nipype.interfaces.utility as util

from nipype.interfaces.io import DataGrabber
from os.path import abspath as opap

import os


###########################################################################
#
#      SUBJECTS NAMES - FUNCTION
#
###########################################################################

def get_subject_names(base_directory, subject_template):
    subjects_list = os.listdir(base_directory)

    subjects_list = [sub for sub in subjects_list if subject_template in sub]

    return subjects_list


def add_two_strings(subject, hand):
    return subject + '/Analyzed_data/MainExp_' + hand + 'Hand.mvpa/'


def pickrun(run_files, run_num=3):
    for run_name in run_files:
        if 'Run'+str(run_num) in run_name:
            ref_run = run_name
    return ref_run


def create_realign_reference(run, whichvol_glob, name='realignref'):
    """
    run (int): run's number

    whichvol_glob: 'first' or 'middle' or 'last' or
        'mean', if 'mean' was chosed for run then whichvol_glob does't matter
    """
    realignref = Workflow(name=name)

    inputnode = Node(
        interface=util.IdentityInterface(
            fields=['in_sub', 'in_hand']),
        name='inputspec'
        )

    ds = Node(DataGrabber(
        infields=['subject_id', 'hand'], outfields=['func']), name='datasource'
        )
    ds.inputs.base_directory = opap(base_directory)
    ds.inputs.template = '%s/%s_Hand/*.nii*'
    ds.inputs.sort_filelist = True
    # ds.inputs.subject_id = 'GK011RZJA'
    # ds.inputs.hand = 'Left'

    realignref.connect(inputnode, 'in_hand', ds, 'hand')
    realignref.connect(inputnode, 'in_sub', ds, 'subject_id')

    img2float = MapNode(
        interface=fsl.ImageMaths(
            out_data_type='float',
            op_string='',
            suffix='_dtype'
            ),
        iterfield=['in_file'],
        name='img2float'
        )

    realignref.connect(ds, ('func', pickrun), img2float, 'in_file')
    # realignref.connect(inputnode, 'in_files', img2float, 'in_file')

    if whichvol_glob != 'mean':
        extract_ref = Node(
            interface=fsl.ExtractROI(t_size=1),
            iterfield=['in_file'],
            name='extractref'
            )

        realignref.connect(
            img2float, ('out_file', pickfirst), extract_ref, 'in_file'
            )
        realignref.connect(
            img2float, ('out_file', pickvol, 0, whichvol_glob),
            extract_ref, 't_min'
            )

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=['ref_vol']),
        name='outputnode'
        )

    if whichvol_glob != 'mean':
        realignref.connect(
            extract_ref, 'roi_file', outputnode, 'ref_vol'
            )
    else:
        meanfunc = pe.Node(
            interface=fsl.ImageMaths(op_string='-Tmean', suffix='_mean'),
            name='meanfunc'
            )
        realignref.connect(
            img2float, ('out_file', pickfirst), meanfunc, 'in_file'
            )
        realignref.connect(
            meanfunc, 'out_file', outputnode, 'ref_vol'
            )

    return realignref


def create_featreg_merge(run, whichvol_glob, name='featregmerge'):
    ###########################################################################
    #
    #     FEATREG_MERGE WORKFLOW
    #
    ###########################################################################

    featregmerge = Workflow(name=name)

    inputnode = Node(
        interface=util.IdentityInterface(
            fields=['in_sub', 'in_hand', 'run', 'whichvol_glob']),
        name='inputspec'
        )
    # inputnode.inputs.in_sub = 'GK011RZJA'

    ###########################################################################
    #
    #     DATA GRABBER NODE
    #
    ###########################################################################

    ds = Node(DataGrabber(
        infields=['subject_id', 'hand'], outfields=['func']), name='datasource'
        )
    ds.inputs.base_directory = opap(base_directory)
    ds.inputs.template = '%s/%s_Hand/*.nii*'
    ds.inputs.sort_filelist = True
    # ds.inputs.subject_id = 'GK011RZJA'
    # ds.inputs.hand = 'Left'

    featregmerge.connect(inputnode, 'in_hand', ds, 'hand')
    featregmerge.connect(inputnode, 'in_sub', ds, 'subject_id')

    '''
        To print the list of files being taken uncomment the following lines.
    '''
    #  functional_input = ds.run().outputs
    #  input_files = functional_input.get()['func']
    #  print input_files

    ###########################################################################
    #
    #     CREATE FEAT REGISTRATION WORKFLOW NODE
    #
    ###########################################################################

    preproc = create_featreg_preproc(highpass=True, whichvol='first')
    preproc.inputs.inputspec.fwhm = 0
    preproc.inputs.inputspec.highpass = 128./(2*2.5)

    # remove_nodes takes list as an argument
    preproc.remove_nodes([preproc.get_node('extractref')])
    '''
    preproc.disconnect(
        preproc.get_node('extractref'), 'roi_file',
        preproc.get_node('realign'), 'ref_file'
        )
    '''
    featregmerge.connect(ds, 'func', preproc, 'inputspec.func')

    ###########################################################################
    #
    #     MERGE NODE
    #
    ###########################################################################

    merge = Node(
        interface=fsl.utils.Merge(
            dimension='t',
            output_type='NIFTI_GZ',
            merged_file='bold.nii.gz'
            ),
        name='merge'
        )
    featregmerge.connect(
        preproc, 'outputspec.highpassed_files',
        merge, 'in_files'
        )

    masksnode = Node(
        interface=fsl.utils.Merge(
            dimension='t',
            output_type='NIFTI_GZ',
            merged_file='masks_merged.nii.gz'
            ),
        name='masksnode'
        )
    featregmerge.connect(
        preproc, 'outputspec.mask', masksnode, 'in_files'
        )

    # ### SPLIT MERGED MASKS ##################################################

    splitnode = Node(
        interface=fsl.utils.Split(
            dimension='t',
            output_type='NIFTI_GZ'
            ),
        name='splitnode'
        )
    featregmerge.connect(
        masksnode, 'merged_file',
        splitnode, 'in_file'
        )

    return featregmerge

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
Directory where all files created and required by workflow will be stored.
'''
# workflow_base_directory = \
#     '/Users/AClab/Documents/mikbuch/Maestro_Project1/mvpa/preprocessing'
workflow_base_directory = '/home/jesmasta/amu/master/working_dir'


subject_template = 'GK'


###############################################################################
#
#      CREATE MAIN WORKFLOW
#
###############################################################################

def create_mvpa_preproc(
        base_directory, datasink_directory,
        workflow_base_directory, subject_template,
        name='mvpapreproc', run=1, whichvol_glob='mean'
        ):

    mvpapreproc = pe.Workflow(name=name)

    inputsub = Node(
        interface=util.IdentityInterface(fields=['sub']),
        name='inputsub'
        )
    # inputsub.inputs.sub = ['GK011RZJA', 'GK012OHPA']
    # inputsub.iterables = ('sub', ['GK011RZJA', 'GK012OHPA'])
    inputsub.iterables = (
        'sub',
        get_subject_names(base_directory, subject_template)
        )

    inputhand = Node(
        interface=util.IdentityInterface(fields=['hand']),
        name='inputhand'
        )
    inputhand.iterables = ('hand', ['Left', 'Right'])

    # ### REFERENCE EXTRACTION ################################################
    reference = create_realign_reference(run=run, whichvol_glob=whichvol_glob)
    mvpapreproc.connect(inputsub, 'sub', reference, 'inputspec.in_sub')
    mvpapreproc.connect(inputhand, 'hand', reference, 'inputspec.in_hand')

    # featreg_merge workflow
    merge = create_featreg_merge(run=run, whichvol_glob=whichvol_glob)

    mvpapreproc.connect(inputsub, 'sub', merge, 'inputspec.in_sub')
    mvpapreproc.connect(inputhand, 'hand', merge, 'inputspec.in_hand')

    mvpapreproc.connect(
        reference, 'outputnode.ref_vol',
        merge, 'featpreproc.realign.ref_file'
        )

    ###########################################################################
    #
    #     DATA SINK NODE
    #
    ###########################################################################

    from nipype.interfaces.utility import Function
    add_two_strings_node = Node(
        interface=Function(
            input_names=['subject', 'hand'],
            output_names=['sub_hand_name'],
            function=add_two_strings
            ),
        name='ats'
        )

    '''
    If iterating trought hands will be available
    '''
    mvpapreproc.connect(
        inputsub, 'sub',
        add_two_strings_node, 'subject'
        )
    mvpapreproc.connect(
        inputhand, 'hand',
        add_two_strings_node, 'hand'
        )

    from nipype.interfaces.io import DataSink

    datasink = Node(interface=DataSink(), name='datasink')
    datasink.inputs.base_directory = opap(datasink_directory)
    datasink.inputs.parameterization = False

    mvpapreproc.connect(
        add_two_strings_node, 'sub_hand_name',
        datasink, 'container'
        )

    mvpapreproc.connect(
        merge, 'merge.merged_file',
        # datasink, ds.inputs.subject_id + '/' + ds.inputs.hand + '_Hand/mvpa'
        datasink, 'preprocessed'
        )
    # mvpa_preproc.connect(inputsub, 'sub', datasink, 'container')

    datasink_masks = Node(interface=DataSink(), name='datasink_masks')
    datasink_masks.inputs.base_directory = opap(datasink_directory)
    datasink_masks.inputs.parameterization = False
    datasink_masks.inputs.substitutions = [('', ''), ('vol0000', 'mask')]

    mvpapreproc.connect(
        add_two_strings_node, 'sub_hand_name',
        datasink_masks, 'container'
        )

    mvpapreproc.connect(
        merge, ('splitnode.out_files', pickfirst),
        datasink_masks, 'preprocessed'
        )

    ###########################################################################
    #
    #     BASE_DIR, GRAPH, AND RUN
    #
    ###########################################################################

    mvpapreproc.base_dir = workflow_base_directory
    # mvpa_preproc.base_dir = \
    #    '/Users/AClab/Documents/mikbuch/Maestro_Project1/mvpa/preprocessing/'

    return mvpapreproc


mvpa_preprocessing = create_mvpa_preproc(
    base_directory=base_directory,
    datasink_directory=datasink_directory,
    workflow_base_directory=workflow_base_directory,
    subject_template=subject_template
    )
mvpa_preprocessing.write_graph("graph.dot")
# mvpa_preprocessing.run()
