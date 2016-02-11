
def random_min_max():
    return random.uniform(dataset.training_data_min, dataset.training_data_max)

k_features = 784


###############################################################################
#
#        LOAD DATA
#
###############################################################################

from pymri.dataset.datasets import DatasetManager

# mvpa_directory = '/tmp/Maestro_Project1/GK011RZJA/Right_Hand/mvpa/'
mvpa_directory = \
    '/amu/master/Maestro_Project1.preprocessed/GK011RZJA/Right_Hand/mvpa/'

print('Loading database from %s' % mvpa_directory)
dataset = DatasetManager(
    mvpa_directory=mvpa_directory,
    # conditions has to be tuples
    contrast=(('PlanTool_0', 'PlanTool_5'), ('PlanCtrl_0', 'PlanCtrl_5')),
    )


dataset_reduced = dataset.feature_reduction(
    k_features=k_features,
    reduction_method='SelectKBest (SKB)',
    normalize=True,
    nnadl=True
    )

from pymri.model import FNN

# create Classifier
cls = FNN(
    type='FNN simple',
    input_layer_size=k_features,
    hidden_layer_size=46,
    output_layer_size=2,
    epochs=100,
    # epochs=10,
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
#   GENETIC ALGORITHM SETUP
#
##############################################################################

import random
import numpy as np

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

IND_SIZE = k_features


toolbox = base.Toolbox()
toolbox.register(
    "attr_float", random.uniform,
    dataset.training_data_min,
    dataset.training_data_max
    )
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)


# define the population to be a list of 'individual's
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

##############################################################################
#
#   FITNESS FUCNTION
#
##############################################################################
from pymri.genetic_algorithm import get_prob_class
# the goal ('fitness') function to be maximized


def activate_network(individual):
    return get_prob_class(
        cls.net, np.reshape(individual, (individual.shape[0], 1)), 0
        ),

# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate", activate_network)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
# toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selRoulette)

# ----------


# random.seed(64)

# create an initial population of 50 individuals (where
# each individual is a list of integers)
pop = toolbox.population(n=10)

# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
#
# NGEN  is the number of generations for which the
#       evolution runs
CXPB, MUTPB, NGEN = 0.1, 0.001, 1000

mean_log = np.zeros(shape=(NGEN,))
min_log = np.zeros(shape=(NGEN,))
max_log = np.zeros(shape=(NGEN,))
std_log = np.zeros(shape=(NGEN,))

print("Start of evolution")

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

print("  Evaluated %i individuals" % len(pop))

# Begin the evolution
for g in range(NGEN):
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        # cross two individuals with probability CXPB
        if random.random() < CXPB:
            toolbox.mate(child1, child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:

        # mutate an individual with probability MUTPB
        if random.random() < MUTPB:

            mutant[random.randint(0, len(mutant)-1)] = random_min_max()
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # print("  Evaluated %i individuals" % len(invalid_ind))

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5

    # print("  Min %s" % min(fits))
    # print("  Max %s" % max(fits))
    # print("  Avg %s" % mean)
    # print("  Std %s" % std)
    min_log[g] = min(fits)
    max_log[g] = max(fits)
    mean_log[g] = mean
    std_log[g] = std

print("\n\n\n")
print("####################################")
print("-- End of (successful) evolution --")

print(
    '\nPOP: %d, CXPB: %0.3f, MUTPB: %0.3f, NGEN: %d' %
    (len(pop), CXPB, MUTPB, NGEN)
    )

# # print("\nBest individual is %s" % best_ind)
best_ind = tools.selBest(pop, 1)[0]

best_ind_with_class = np.zeros(shape=(k_features, 1,))
for i in range(len(best_ind)):
    best_ind_with_class[i][0] = best_ind[i]

best_activation = cls.net.feedforward(best_ind_with_class)
print(
    "\nOutput layer values for best individual are:" +
    "\nx_1 = %f and x_2 = %f" %
    (best_activation[0], best_activation[1])
    )
print(
    "\nFitness value of the best individual is: %s" %
    best_ind.fitness.values[0]
    )

print(
    "\nIndividuals were vectors of floats in range < %f, %f >" %
    (dataset.training_data_min, dataset.training_data_max)
    )
print("####################################")
print("\n\n\n")

# visualize evolution
import matplotlib.pyplot as plt

plt.subplot(211)
plt.plot(max_log, linewidth=2.0, c='r', label='max fitness')
plt.plot(mean_log, linewidth=2.0, label='mean fitness')
plt.xlim(0,max_log.shape[0])
plt.ylim(-0.1, 1.1)
plt.ylabel('fitness')
plt.xlabel('number of the generation (G)')
plt.legend(loc=4)

plt.subplot(212)
plt.plot(max_log, linewidth=2.0, c='r', label='max fitness')
plt.plot(mean_log, linewidth=2.0, label='mean fitness')
plt.ylim(
    mean_log.min()-0.01*mean_log.min(), max_log.max() + 0.01*max_log.max()
    )
plt.ylabel('fitness')
plt.xlabel('number of the generation (G)')
plt.legend(loc=4)

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 32}
plt.rcParams.update({'font.size': 32})
plt.rc('font', **font)

plt.show()


############################################
#
#   SAVE ACTIVATION PATTERN AS NIFTI
#
############################################
fluctuation = np.array(best_ind)
activation = np.array(fluctuation)
deactivation = np.array(activation)
deactivation[activation > 0] = 0
deactivation = -deactivation
activation[activation < 0] = 0

# save activation to nifti file
dataset.save_as_nifti(fluctuation, 'prototype_fluctuation')
dataset.save_as_nifti(activation, 'prototype_activation')
dataset.save_as_nifti(deactivation, 'prototype_deactivation')
