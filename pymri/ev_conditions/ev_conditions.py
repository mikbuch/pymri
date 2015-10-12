"""
name: ev_conditions.py

Plan:
1. Get all conditions for all runs to one pyhon list.
2*. Seconds to volumens transformation.
3. Sort by run, then by volumens or seconds (depends on logical value of vol).
4. Split lists to x sublists (where x is number of the runs).
5*. Fill unlabeled volumens basing on the previous volumen label.
6. Write sublists to separate files.

Note: the way the volumens files are written allows easy masking nifti driven
arrays.

* Steps marked with aterix are performed if vol=True
"""

import glob
import os.path


def merge_ev_files(
    path,
    tr,
    output='out_ev.txt',
    format='mac',
    vol=True,
    vol_num=None
    ):
    """Merge Explanatory Variable files.

    Create one EV (conditions) file from multipe files (all concerning the same
    file).
    You can decide whether to save output in second or as volumen format.

    Parameters
    ----------
    path : input directory path
        Directory containing all EV files for particular run. 

    output : output file path
        
    format : mac, lin or win
        Necessary for determining linebreak character.

    vol : True or False
        If is set then file is saved in volume format, else in second.

    Returns
    -------
    out : output file path

    Examples
    --------
    >>> from pymri.ev_conditions import merge_ev_files
    >>> merge_ev_files('./dir_containing_EVs/', 2)
    """

    # store all time points with corresponding conditions
    evs = []

    # use glob.glob in order to ignore all hidden files
    # source: http://stackoverflow.com/a/7099342
    paths = glob.glob(os.path.join(path, '*'))

    import csv
    import re

    for path_absolute in paths:
        # get only the filename, not the absolute path
        ev_filename = re.sub(path, '', path_absolute)

        with open(path_absolute, 'rU') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                # some lines at the end may be blank
                if row:
                    # extract run number and condition name from filename
                    run_index = ev_filename.find('_Run_') + len('_Run_')
                    run = int(ev_filename[run_index])
                    cond = ev_filename[run_index+6:-4]
                    evs.append([float(row[0]), run, cond, ev_filename])

    # sort list basing on 1st ([1]) index 
    # 3. Sort by run, then by volumens or seconds.
    import operator
    evs = sorted(evs, key=operator.itemgetter(1, 0))

    # if paramter is not changed transform seconds to volumens
    if vol:
        for ev in evs:
            ev[0] = int(round(ev[0]/float(tr)))

    # 4. Split lists to x sublists (where x is number of the runs).
    runs = [[]]
    for ev in range(len(evs)):
        runs[-1].append(evs[ev])
        if ev < len(evs)-1:
            if evs[ev][1] < evs[ev+1][1]:
                runs.append([])
        
    ###########################################################################
    # Create list to store information about the particular volumen in the 
    # particular run. Then fill it basing on information from runs list (when
    # the conditions switches).
    ###########################################################################

    # list to store info about volumens
    volumen_condition = []
    for run in runs:
        volumen_condition.append([])
        count_cond = 0
        cond_tmp = run[count_cond]
        for vol in range(145):
            if count_cond < len(run)-2:
                # if the condition doesn't switches for the next volumen, use
                # the same condition as before
                if run[count_cond+1][0] < vol + 2:
                    cond_tmp = run[count_cond+1]
                    count_cond += 1
            # take information about condition and run
            volumen_condition[-1].append([cond_tmp[2], cond_tmp[1]-1])

    evs = runs
    
    # write to file
    with open('attributes_literal.txt', 'wb') as outfile:
        csv_writer = csv.writer(outfile, delimiter=' ')
        for run in volumen_condition:
            csv_writer.writerows(run)

    # specify the conditions' integer representation to further encoding
    conditions = {'Rest':0,
        'PlanTool_0':1, 'PlanTool_5':2, 'PlanCtrl_0':3, 'PlanCtrl_5':4,
        'ExeTool_0':5, 'ExeTool_5':6, 'ExeCtrl_0':7, 'ExeCtrl_5':8}

    # change string values to specific integer code
    attributes = list(volumen_condition)
    for run in range(len(volumen_condition)):
        for cond in conditions:
            attributes[run] = [
                map(lambda x:x if x!= cond else conditions[cond],
                attributes[run][i])
                for i in range(len(volumen_condition[run]))
                ]

    # Write encoded conditions.
    # First number (column) is condition code, second column is run number.
    with open('attributes.txt', 'wb') as outfile:
        csv_writer = csv.writer(outfile, delimiter=' ')
        for run in attributes:
            csv_writer.writerows(run)

    # return for debugging
    return attributes, volumen_condition

# example function usage:
# a, v = merge_ev_files('/home/jesmasta/amu/master/M_Buchwald/EVs/S_011_L/', 2)
