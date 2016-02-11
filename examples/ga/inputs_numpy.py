
###############################################################################
#
#        LOAD DATA
#
###############################################################################

from pymri.dataset.datasets import DatasetManager

mvpa_directory = '/tmp/Maestro_Project1/GK011RZJA/Right_Hand/mvpa/'

print('Loading database from %s' % mvpa_directory)
dataset = DatasetManager(
    mvpa_directory=mvpa_directory,
    # conditions has to be tuples
    contrast=(('PlanTool_0', 'PlanTool_5'), ('PlanCtrl_0', 'PlanCtrl_5')),
    )

dataset_reduced = dataset.feature_reduction(
    k_features=784,
    reduction_method='SelectKBest (SKB)',
    normalize=True,
    nnadl=True
    )

from pymri.model import FNN

# create Classifier
cls = FNN(
    type='FNN simple',
    input_layer_size=784,
    hidden_layer_size=46,
    output_layer_size=2,
    epochs=100,
    mini_batch_size=11,
    learning_rate=3.0,
    verbose=True
    )

# split dataset
training_data, test_data, valid_data = dataset.split_data(
    sizes=(0.75, 0.25)
    )

# train and test classifier
cls.train_and_test(training_data, test_data)
accuracy = cls.get_accuracy()

print('accuracy = %0.2f' % accuracy)


##############################################################################
#
#   FITNESS FUCNTION
#
##############################################################################
from pymri.genetic_algorithm import get_prob_class
# the goal ('fitness') function to be maximized


def feedforward_network(individual):
    return get_prob_class(
        cls.net, numpy.reshape(individual, (individual.shape[0], 1)), 0
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

toolbox.register(
    "attr_bool", random.uniform,
    dataset.training_data_min, dataset.training_data_max
    )
toolbox.register(
    "individual", tools.initRepeat,
    creator.Individual, toolbox.attr_bool, n=IND_SIZE
    )
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
    else:
        # Swap the two cx points
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


    import ipdb
    ipdb.set_trace()
    return pop, stats, hof

if __name__ == "__main__":
    main()
