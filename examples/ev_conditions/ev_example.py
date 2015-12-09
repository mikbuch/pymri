'''
name: ev_example.py
type: script

Use this script to generate all 'attributes.txt' and 'attributes_literal.txt'
files. They contain information about conditions in particular volumens.
'''

from pymri.ev_conditions import get_attributes

base_directory = '/tmp/Maestro_Project1/'
subject_template = 'GK'
ev_template = base_directory + '%s/EVs/S_%s_%s/'
mvpa_template = base_directory + '%s/%s_Hand/mvpa/'

from pymri.utils.paths_dirs_info import get_subject_names

subjects = get_subject_names(base_directory, subject_template)
hands = ['Left', 'Right']

for sub in subjects:
    for hand in hands:
        get_attributes(
            input_dir=ev_template % (sub, sub[2:5], hand[0]),
            tr=2,
            output_dir=mvpa_template % (sub, hand)
            )
