'''
name: ev_multipe_subjcts.py
type: script

Use this script to generate all 'attributes.txt' and 'attributes_literal.txt'
files. They contain information about conditions in particular volumens.
'''

import os

from pymri.ev_conditions import get_attributes
from pymri.utils.paths_dirs_info import get_subject_names

base_directory = '/tmp/Maestro_Project1/'
# base_directory = '/Users/AClab/Documents/mikbuch/Maestro_Project1/'
subject_template = 'GK'
ev_template = base_directory + '%s/EVs/S_%s_%s/'
hand_template = base_directory + \
    '%s/Analyzed_data/MainExp_%sHand.mvpa/preprocessed'
preprocessed_template = base_directory + \
    '%s/Analyzed_data/MainExp_%sHand.mvpa/preprocessed'

subjects = get_subject_names(base_directory, subject_template)
hands = ['Left', 'Right']

for sub in subjects:
    for hand in hands:
        ev_dir = ev_template % (sub, sub[2:5], hand[0])
        hand_dir = hand_template % (sub, hand)
        preprocessed_dir = preprocessed_template % (sub, hand)
        # check if proper subject_hand directory exists (is properly pointed)
        if os.path.exists(hand_dir):
            # create mvpa directory (for sub_hand) if it doesn't exist
            if not os.path.exists(preprocessed_dir):
                os.makedirs(preprocessed_dir)
        get_attributes(
            input_dir=ev_dir,
            tr=2,
            output_dir=preprocessed_dir,
            quiet=False
            )
