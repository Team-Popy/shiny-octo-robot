""" FIRST ATTEMPT TO USE DEAP AND DEMO TOGETHER, ONLY SOME PARTS CHANGED, MORE FROM DEMO HAS TO BE REMOVED"""

import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
from math import fabs, sqrt
import glob, os
from deap import base, creator
import random
from deap import tools


# evaluation CODE FROM DEMO
# todo: ask if we can use it
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


# todo: understand DEAP parameters and how to use toolbox
""" DEAP config starts - I DIDN'T SET ANYTHING SO WE NEED TO WORK ON THAT STILL IT'S DEFAULT FOR NOW """
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

IND_SIZE = 10

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)
""" DEAP config ends """

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'test_deap1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

""" if you change the number of hidden neurons, the number of sensors changes as well, if we want
to use perceptron we have to use only 1 hidden layer so that it works """
n_hidden_neurons = 1

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[3],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment
env.state_to_log()  # checks environment state

# Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###
ini = time.time()  # sets time marker

# genetic algorithm params
run_mode = 'train'  # train or test

#todo: understand this formula why are there these numbers

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

""" ATTENTION - everytime you change anything (besides gens), delete evoman_solstate file that is used here """
dom_u = 1
dom_l = -1
npop = 20
gens = 10
mutation = 0.2
last_best = 0


#todo: ask if we can use it

# runs simulation CODE FROM DEMO
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

#todo: ask if we can use it

# normalizes CODE FROM DEMO
def norm(x, pfit_pop):
    if (max(pfit_pop) - min(pfit_pop)) > 0:
        x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


# CODE FROM DEMO
init_population = np.random.uniform(dom_l, dom_u, (npop, n_vars))


#todo: create our own way of choosing parents

# FROM DEMO - WE HAVE TO CHANGE IT LATER
def tournament(population, population_fitness):
    random_value_1 = np.random.randint(0, population.shape[0], 1)
    random_value_2 = np.random.randint(0, population.shape[0], 1)

    if population_fitness[random_value_1] > population_fitness[random_value_2]:
        return population[random_value_1][0]
    else:
        return population[random_value_2][0]



#todo: ask if we can use it

# limits FROM DEMO
def limits(x):
    if x > dom_u:
        return dom_u
    elif x < dom_l:
        return dom_l
    else:
        return x


""" DEAP IMPLEMENTATION PARTIALLY """
crossover_threshold, mutation_threshold = 0.5, 0.2


def crossover(population_data):
    total_offspring = np.zeros((0, n_vars))  # tuple shape (0, num_of_sensors)

    # this loop is from DEMO
    for p in range(0, population_data.shape[0], 2):
        # todo: here I made a mix of tournament and deap we should change it later
        parent_1 = tournament(population_data, fit_pop)[::2]
        parent_2 = tournament(population_data, fit_pop)[1::2]

        """ OUR PART STARTS """
        """ crossover """
        offspring = toolbox.mate(parent_1, parent_2)  # results in two new children in a tuple
        offspring_1 = offspring[0]
        offspring_2 = offspring[1]

        """ combine them together?"""
        one_offspring = np.hstack((offspring_1, offspring_2))

        """ mutation """
        if random.random() < mutation_threshold:
            mutant_prior = toolbox.clone(one_offspring)
            mutated_offspring = toolbox.mutate(one_offspring)[0]
            total_offspring = np.vstack((total_offspring, mutated_offspring))

        else:
            total_offspring = np.vstack((total_offspring, one_offspring))

    """ OUR PART ENDS """
    return total_offspring  # has to be (x, 31) shape


# todo: create our own way of replacing worst gonomes (or just remove everything and take all new offspring,
#  we can check what's better)

# FROM DEMO CODE
# kills the worst genomes, and replace with new best/random solutions
def doomsday(pop, fit_pop):
    worst = int(npop / 4)  # a quarter of the population
    order = np.argsort(fit_pop)
    orderasc = order[0:worst]

    for o in orderasc:
        for j in range(0, n_vars):
            pro = np.random.uniform(0, 1)
            if np.random.uniform(0, 1) <= pro:
                pop[o][j] = np.random.uniform(dom_l, dom_u)  # random dna, uniform dist.
            else:
                pop[o][j] = pop[order[-1:]][0][j]  # dna from best

        fit_pop[o] = evaluate([pop[o]])

    return pop, fit_pop


# todo: I think we can have it? but we can ask
# loads file with the best solution for testing
if run_mode == 'test':
    bsol = np.loadtxt(experiment_name + '/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    evaluate([bsol])

    sys.exit(0)

# todo: can we have it?
# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name + '/evoman_solstate'):

    print('\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print('\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux = open(experiment_name + '/gen.txt', 'r')
    ini_g = int(file_aux.readline())
    file_aux.close()

# todo: can we have it?
# saves results for first pop
file_aux = open(experiment_name + '/results.txt', 'a')
file_aux.write('\n\ngen best mean std')
print('\n GENERATION ' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
    round(std, 6)))
file_aux.write(
    '\n' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
file_aux.close()

# evolution

last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g + 1, gens):
    print(ini_g)
    print(f"!!!!!!!!!!!! generation number {i}")

    # amount of offspring is not constant in this solution

    offspring = crossover(pop)  # crossover
    fit_offspring = evaluate(offspring)  # evaluation
    pop = np.vstack((pop, offspring))
    fit_pop = np.append(fit_pop, fit_offspring)

    best = np.argmax(fit_pop)  # best solution in generation
    fit_pop[best] = float(evaluate(np.array([pop[best]]))[0])  # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    # selection
    fit_pop_cp = fit_pop
    fit_pop_norm = np.array(list(map(lambda y: norm(y, fit_pop_cp),
                                     fit_pop)))  # avoiding negative probabilities, as fitness is ranges from negative numbers
    probs = (fit_pop_norm) / (fit_pop_norm).sum()
    chosen = np.random.choice(pop.shape[0], npop, p=probs, replace=False)
    chosen = np.append(chosen[1:], best)
    pop = pop[chosen]
    fit_pop = fit_pop[chosen]

    # todo: read about it

    # searching new areas

    # todo: what is this

    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 15:
        file_aux = open(experiment_name + '/results.txt', 'a')
        file_aux.write('\ndoomsday')
        file_aux.close()

        pop, fit_pop = doomsday(pop, fit_pop)
        notimproved = 0

    best = np.argmax(fit_pop)
    std = np.std(fit_pop)
    mean = np.mean(fit_pop)

    # todo: can we have it?
    # saves results
    file_aux = open(experiment_name + '/results.txt', 'a')
    print('\n GENERATION ' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
        round(std, 6)))
    file_aux.write(
        '\n' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.close()

    # saves generation number
    file_aux = open(experiment_name + '/gen.txt', 'w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name + '/best.txt', pop[best])

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()

fim = time.time()  # prints total execution time for experiment
print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

file = open(experiment_name + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log()  # checks environment state
