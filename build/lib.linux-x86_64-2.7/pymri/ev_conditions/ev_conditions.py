"""
name: ev_condtitions.py
type: module

Plan:
1. Get all conditions for all runs to one pyhon list.
2. Seconds to volumens transformation.
3. Sort by run, then by volumens or seconds (depends on logical value of vol).
4. Split lists to x sublists (where x is number of the runs).
5. Fill unlabeled volumens basing on the previous volumen label.
6. Write sublists to separate files.

Note: the way the volumens files are written allows easy masking nifti driven
arrays.


Notes:
If the files has been prepared manually. Sometimes there are human errors.
The following precautions has been taken to prevent compiler errors:
    * Problem: in files sometimes there are blank lines at the end.
      Solution: check if there is anything in the line before processing it.
    * Problem: sometimes there are other files than EVs in EV directory.
      Solution: template for EV files (default *.txt) - template system is
      provided by the author and is simple and crude but efficient.

"""

import glob
import os.path

def get_attributes(
        input_dir,
        tr,
        output_dir='./',
        template='_Run_',
        format='mac',
        quiet=True
        ):
    """Get Explanatory Variables attributes from standard EV files.

    Create one EV (conditions) file from multipe files (all concerning the same
    file).
    You can decide whether to save output in second or as volumen format.

    Parameters
    ----------
    input_dir : string
        Directory containing all EV files for particular run.

    tr : int or float
        How often does the slice was being aquired.

    output_dir : string, optional
        Directory where attributes*.txt files will be created. Default is
        current directory.

    template : string, optional
        EV file template (if case there are some other files in EV directory.

    format : string, optional
        mac, lin or win
        Necessary for determining linebreak character.

    quiet: bool, optional
        Wheter print informations (feedback) during processing or not.


    Examples
    --------
    >>> from pymri.ev_conditions import get_attributes
    >>> get_attributes('./dir_containing_EVs/', 2)
    """

    # store all time points with corresponding conditions
    evs = []

    # use glob.glob in order to ignore all hidden files
    # source: http://stackoverflow.com/a/7099342
    paths = glob.glob(os.path.join(input_dir, '*'))

    # take only files that match template
    paths = [path for path in paths if template.replace('*', '') in path]

    # if user forgot to specify dir
    if output_dir[-1] != '/':
        output_dir += '/'

    import csv
    import re

    for path_absolute in paths:

        # get only the filename, not the absolute path
        ev_filename = re.sub(input_dir, '', path_absolute)

        with open(path_absolute, 'rU') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                # some lines at the end may be blank
                if row:
                    if len(row[0]) > 0:
                        # extract run number and condition name from filename
                        run_index = ev_filename.find('_Run_') + len('_Run_')
                        run = int(ev_filename[run_index])
                        # TODO: create better function for finding condition
                        if ev_filename[run_index+6].isalpha():
                            cond = ev_filename[run_index+6:-4]
                        else:
                            cond = ev_filename[run_index+7:-4]
                        evs.append([float(row[0]), run, cond, ev_filename])

    # sort list basing on 1st ([1]) index
    # 3. Sort by run, then by volumens or seconds.
    import operator
    evs = sorted(evs, key=operator.itemgetter(1, 0))

    # transform seconds to volumens
    for ev in evs:
        ev[0] = ev[0]/float(tr)

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
    attributes_literal_output = output_dir + 'attributes_literal.txt'
    with open(attributes_literal_output, 'wb') as outfile:
        csv_writer = csv.writer(outfile, delimiter=' ')
        for run in volumen_condition:
            csv_writer.writerows(run)
    if not quiet:
        print('Successfully generated file: %s' % attributes_literal_output)

    # specify the conditions' integer representation to further encoding
    conditions = {
        'Rest': 0,
        'PlanTool_0': 1, 'PlanTool_5': 2, 'PlanCtrl_0': 3, 'PlanCtrl_5': 4,
        'ExeTool_0': 5, 'ExeTool_5': 6, 'ExeCtrl_0': 7, 'ExeCtrl_5': 8
        }

    # lowering conditions strings is required in case someone mistypes ev
    # filename (it happens)
    for key in conditions:
        conditions[key.lower()] = conditions.pop(key)

    # change string values to specific integer code
    attributes = list(volumen_condition)
    for run in range(len(volumen_condition)):
        for vol in range(len(volumen_condition[run])):
            # get the condition code (being string) and lower it to prevent
            # errors if someone mistypes condition name
            condition_code = volumen_condition[run][vol][0].lower()
            attributes[run][vol] = \
                [conditions[condition_code], run]

    # Write encoded conditions.
    # First number (column) is condition code, second column is run number.
    attributes_output = output_dir + 'attributes.txt'
    with open(attributes_output, 'wb') as outfile:
        csv_writer = csv.writer(outfile, delimiter=' ')
        for run in attributes:
            csv_writer.writerows(run)
    if not quiet:
        print('Successfully generated file: %s' % attributes_output)
