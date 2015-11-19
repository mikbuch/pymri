

# ######## DatastManager initialization #######################################

from pymri.dataset import DatasetManager

k_features = 784
hidden_neurons = 46

###############################################################################
#
#        LOAD DATA
#
###############################################################################
from pymri.dataset import DatasetManager
# dataset settings
ds = DatasetManager(
    path_input='/home/jesmasta/amu/master/nifti/bold/',
    contrast=(('PlanTool_0', 'PlanTool_5'), ('PlanCtrl_0', 'PlanCtrl_5')),
    k_features = k_features,
    normalize = True,
    nnadl = True,
    sizes=(0.75, 0.25)
    )
# load data
ds.load_data()

###############################################################################
#
#        CHOOSE ROIs
#
###############################################################################

# select feature reduction method
ds.feature_reduction(roi_selection='SelectKBest')
# ds.feature_reduction(roi_selection='/amu/master/nifti/bold/roi_mask_plan.nii.gz')

k_features = ds.X_processed.shape[1]
print(k_features)

# get training, validation and test datasets for specified roi
training_data, test_data, vd = ds.split_data()

# set up network
from pymri.model import fnn
decision = 'y'
while decision == 'y':
    net = fnn.Network([k_features, hidden_neurons, 2])
    net.SGD(training_data, 100, 11, 2.961, test_data=test_data)
    print('\nBest score: %d' % (net.best_score))
    print('Continue training? (y/n + enter)'),
    decision = raw_input()

net.biases = net.best_biases
net.weights = net.best_weights


##############################################################################
#
#   FITNESS FUCNTION
#
##############################################################################
from pymri.genetic_algorithm import get_prob_class
# the goal ('fitness') function to be maximized


def feedforward_network(individual):
    return get_prob_class(
        net, numpy.reshape(individual, (individual.shape[0], 1)), 0
        ),

# ######## DEAP genetic algorithm #############################################

import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

IND_SIZE = k_features

toolbox.register("attr_bool", random.uniform, ds.training_data_min, ds.training_data_max)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2
    
    
toolbox.register("evaluate", feedforward_network)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    
    pop = toolbox.population(n=300)
    
    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=numpy.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats,
                        halloffame=hof)


    return pop, stats, hof

if __name__ == "__main__":
    main()
