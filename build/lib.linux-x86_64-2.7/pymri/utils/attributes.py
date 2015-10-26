'''
attributes.py
module

For Haxby's dataset: get the attributes of all merged runs for one subject.
1452 volumens have all merged runs for one subject.

Generate file containing 1452 lines - one for each volumen.
'''

import numpy as np


###############################################################################
#
#     attributes.txt generation
#
###############################################################################

def get_conditions(directory):
    runs = 12
    conds = 8
    # conditions: 12 runs, 8 conditions each
    conditions = np.zeros(shape=(runs, conds))
    for run in range(runs):
        for cond in range(conds):
            file_cond = '/task001_run' + '{0:03}'.format(run+1) +\
                '/cond' + '{0:03}'.format(cond+1) + '.txt'
            info = np.genfromtxt(directory + file_cond, delimiter='\t')[0][0]
            conditions[run][cond] = info
    conditions = conditions.argsort()
    return conditions

a = np.array([5, 9, 5, 9, 6, 9, 5, 9, 5, 9, 6, 9, 5, 9, 6, 9, 5])
b = np.array(
    [12, 24, 12, 24, 12, 24, 12, 24, 12, 24, 12, 24, 12, 24, 12, 24, 12]
    )


def debug_volumens_seconds(vols, secs):
    for i in range(len(a)):
        if i % 2 != 0:
            vols_sum = vols[:i].sum()
            secs_sum = secs[:i].sum()
            print(str(secs_sum) + ' ' + str(secs_sum+24))
            print(str(vols_sum*2.5) + ' ' + str(vols_sum*2.5+2.5*9))
            print('')


attributes = np.zeros(shape=(1452, 2))

conditions = np.zeros(shape=(6, 12))

run_cnt = 0
for vol in range(len(attributes)):
    if vol % 121 == 0:
        run_cnt += 1
    attributes[vol][1] = run_cnt - 1

conditions = get_conditions('/media/e0b555e6-cfa3-41fa-abd8-17ea6e249dc2/\
ds105/sub005/model/model001/onsets')

add_one = np.ones(shape=conditions.shape)
conditions = np.add(conditions, add_one)

paradigm = np.zeros(shape=(conditions.shape[0], conditions.shape[1]*2+1))
for i in range(len(paradigm)):
    for j in range(len(paradigm[i])):
        if j % 2 != 0:
            paradigm[i][j] = conditions[i][j/2.0-0.5]

cnt = 0

for vol in range(len(attributes)):
    vol_in_run = vol % 121
    run = vol / 121
    for i in range(len(a)):
        cnt += a[i]
        if cnt >= vol_in_run:
            attributes[vol][0] = int(paradigm[run][i])
            cnt = 0
            break

np.savetxt('attributes.txt', attributes, fmt='%i')


###############################################################################
#
#     attributes_literal.txt generation
#
###############################################################################

literal = \
    ['rest', 'house', 'scrambledpix', 'cat', 'shoe',
     'bottle', 'scissors', 'chair', 'face']

attributes_literal = attributes.astype(int).T.tolist()

for i in range(len(literal)):
    attributes_literal[0] = map(
        lambda x: x if x != i else literal[i], attributes_literal[0]
        )

attributes_literal = np.array(attributes_literal).T
np.savetxt('attributes_literal.txt', attributes_literal, fmt='%s')
