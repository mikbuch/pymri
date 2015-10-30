
import random
import numpy as np

from deap import base
from deap import creator
from deap import tools

from pymri.dataset.nnadl_dataset import load_nifti
from pymri.genetic_algorithm.fitness_functions import test_network


# length of binary chain
# here it is 27 (see README_ga)
L = 27

# load dataset form nifti
path = '/home/jesmasta/amu/master/nifti/bold/'
X, y = load_nifti(path)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator: define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers: define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, L
    )

# define the population to be a list of 'individual's
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decode_bin(bin_list):
    return int(''.join(map(str, bin_list)), 2)

# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    # get three arguments from binary chain (in binary represenation)
    hidden = individual[:10]
    eta = individual[10:22]
    minibatch= individual[22:]

    # transform binary reprezentation into ints and floats
    hidden = decode_bin(hidden)
    eta = decode_bin(eta)
    minibatch = decode_bin(minibatch)

    return test_network(X, y, 2, hidden, eta*0.001, minibatch+10),

# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

# ----------


def main():
    random.seed(64)


    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=6)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #
    # NGEN  is the number of generations for which the
    #       evolution runs
    CXPB, MUTPB, NGEN = 0.5, 0.2, 6

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
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
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
    plt.errorbar(np.arange(len(mean_log)), mean_log, yerr=std_log*3)
    plt.show()

if __name__ == "__main__":
    main()
