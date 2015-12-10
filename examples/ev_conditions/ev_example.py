'''
name: ev_example.py
type: script

Use this script to generate all 'attributes.txt' and 'attributes_literal.txt'
files. They contain information about conditions in particular volumens.
'''

import os

from pymri.ev_conditions import get_attributes

base_directory = '/tmp/Maestro_Project1/'
# base_directory = '/Users/AClab/Documents/mikbuch/Maestro_Project1/'
subject_template = 'GK'
ev_template = base_directory + '%s/EVs/S_%s_%s/'
hand_template = base_directory + '%s/%s_Hand/'
mvpa_template = base_directory + '%s/%s_Hand/mvpa/'

from pymri.utils.paths_dirs_info import get_subject_names

subjects = get_subject_names(base_directory, subject_template)
hands = ['Left', 'Right']

for sub in subjects:
    for hand in hands:
        ev_dir = ev_template % (sub, sub[2:5], hand[0])
        hand_dir = hand_template % (sub, hand)
        mvpa_dir = mvpa_template % (sub, hand)
        # check if proper subject_hand directory exists (is properly pointed)
        if os.path.exists(hand_dir):
            # create mvpa directory (for sub_hand) if it doesn't exist
            if not os.path.exists(mvpa_dir):
                os.makedirs(mvpa_dir)
        get_attributes(
            input_dir=ev_dir,
            tr=2,
            output_dir=mvpa_dir,
            quiet=False
            )
