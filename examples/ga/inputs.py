
import random
import numpy as np

from deap import base
from deap import creator
from deap import tools

from pymri.dataset.nnadl_dataset import load_nifti
from pymri.genetic_algorithm.fitness_functions import test_network


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

IND_SIZE=50

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)


# define the population to be a list of 'individual's
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# TODO: adjust that or maybe no need with float vector
# def decode_bin(bin_list):
    # return int(''.join(map(str, bin_list)), 2)
 

from pymri.dataset import load_nnadl_dataset
path = '/home/jesmasta/amu/master/nifti/bold/'
training_data, validation_data, test_data = load_nnadl_dataset(
    path,
    (('ExeTool_0', 'ExeTool_5'), ('ExeCtrl_0', 'ExeCtrl_5')),
    k_features = 50,
    normalize = True,
    scale_0_1 = True,
    sizes=(0.75, 0.25)
    )

from pymri.model import fnn
decision = 'y'
while decision == 'y':
    net = fnn.Network([50, 25, 2])
    net.SGD(training_data, 100, 11, 2.961, test_data=test_data)
    print('\nBest score: %d' % (net.best_score))
    print('Continue training? (y/n + enter)'),
    decision = raw_input()
    
net.biases = net.best_biases
net.weights = net.best_weights

# TODO: number of features (reduced with PCA or selectkbest)

from pymri.genetic_algorithm import get_prob_class
# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    return get_prob_class(
        net, np.reshape(individual, (individual.shape[0], 1)), 0
        ),

# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
# toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selRoulette)

# ----------


def main():
    random.seed(64)


    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=800)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #
    # NGEN  is the number of generations for which the
    #       evolution runs
    CXPB, MUTPB, NGEN = 0.5, 0.8, 1000

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

                mutant[random.randint(0,len(mutant)-1)] = random.random()
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

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    import matplotlib.pyplot as plt
    plt.plot(min_log)
    plt.plot(max_log)
    # plt.errorbar(np.arange(len(mean_log)), mean_log, yerr=std_log*3)
    plt.show()


if __name__ == "__main__":
    main()

a = toolbox.individual()
z = evalOneMax(a)
