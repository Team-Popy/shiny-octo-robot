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
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


# todo: remove deap when we don't need it anymore

""" DEAP config starts - I DIDN'T SET ANYTHING SO WE NEED TO WORK ON THAT STILL IT'S DEFAULT FOR NOW """
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

IND_SIZE = 10

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxUniform, indpb=0.1)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)
""" DEAP config ends """

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

""" CHANGE THE NAME TO ENEMY NUMBER, CROSSOVER NAME AND TRIAL """
experiment_name = 'enemy_1_new_survival'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
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
""" CHANGE IT TO TEST TO TEST THE RESULTS """
run_mode = 'train'  # train or test

# todo: understand this formula why are there these numbers
# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

""" ATTENTION - everytime you change anything (besides gens), delete evoman_solstate file that is used here """
lower_limit = -1
upper_limit = 1

npop = 100
gens = 30
crossover_threshold = 0.5
mutation_threshold = 0.2

last_best = 0


# runs simulation CODE FROM DEMO
def simulation(environment, x):
    fitness, player_life, enemy_life, game_time = environment.play(pcont=x)
    return fitness


############################################ CHECK
def normalization(x, population_fitness):
    denominator_check = max(population_fitness) - min(population_fitness)

    if denominator_check > 0:
        x_norm = (x - min(population_fitness)) / (max(population_fitness) - min(population_fitness))
    else:
        x_norm = 0.0000000001

    return x_norm


########################################### CHECK


################################################################# CHECK
def tournament_selection(population, population_fitness):
    # choosing 3 individuals from the population at random
    random_list = random.sample(range(0, population.shape[0]), 4)
    random_val_1 = random_list[0]
    random_val_2 = random_list[1]
    random_val_3 = random_list[2]
    random_val_4 = random_list[3]

    first_fitness = population_fitness[random_val_1]
    second_fitness = population_fitness[random_val_2]
    third_fitness = population_fitness[random_val_3]
    fourth_fitness = population_fitness[random_val_4]

    sorted_fitness = [first_fitness, second_fitness, third_fitness, fourth_fitness]
    sorted_fitness.sort(reverse=True)
    parent_1_fitness = sorted_fitness[0]
    parent_2_fitness = sorted_fitness[1]

    parent_1_index = list(population_fitness).index(parent_1_fitness)
    parent_2_index = list(population_fitness).index(parent_2_fitness)

    return population[parent_1_index], population[parent_2_index]


################################################################# CHECK


""" WEIGHTS INITIALIZATION """
init_population = np.random.uniform(lower_limit, upper_limit, (npop, n_vars))

""" Make the weight limited """


def limit_the_weights(weight):
    if weight > upper_limit:
        return upper_limit
    elif weight < lower_limit:
        return lower_limit
    else:
        return weight


def two_point_crossover_uniform_mutation(population_data):
    crossover_point = [np.uint8(n_vars / 4), np.uint8(n_vars - n_vars / 4)]
    total_offspring = []

    for p in range(0, population_data.shape[0], 2):
        offspring = np.zeros((2, n_vars))
        parent_1, parent_2 = tournament_selection(population_data, population_fitness)

        if np.array_equal(parent_1, parent_2):
            parent_1 = toolbox.mutate(parent_1)[0]

        for m in crossover_point:
            parent_1, parent_2 = single_point_crossover(parent_1, parent_2, m)

        offspring[0] = parent_1.copy()
        offspring[1] = parent_2.copy()
        # mutation
        """ if it's too slow, think how to make it faster """
        for idx in range(offspring.shape[0]):
            if np.random.uniform(0, 1.0, 1)[0] <= mutation_threshold:
                random_value = np.random.uniform(0, 1.0, 1)
                offspring[idx] = offspring[idx] + random_value
        offspring[0] = np.array(list(map(lambda y: limit_the_weights(y), offspring[0])))
        offspring[1] = np.array(list(map(lambda y: limit_the_weights(y), offspring[1])))

        total_offspring.append(offspring[0])
        total_offspring.append(offspring[1])


    final_total_offspring = np.vstack(total_offspring)
    return final_total_offspring


def single_point_crossover(parent_1, parent_2, crossover_point):
    parent_1_new = np.append(parent_1[:crossover_point], parent_2[crossover_point:])
    parent_2_new = np.append(parent_2[:crossover_point], parent_1[crossover_point:])
    return parent_1_new, parent_2_new


# def uniform_crossover_gausian_mutation(population_data):
#     total_offspring = np.zeros((0, n_vars))
#
#     for p in range(0, population_data.shape[0], 2):
#         parent_1 = tournament_selection(population_data, population_fitness)[::2]
#         parent_2 = tournament_selection(population_data, population_fitness)[1::2]
#
#         n_offspring = np.random.randint(1, 3 + 1, 1)[0]
#         offspring = np.zeros((n_offspring, n_vars))
#
#         for k in range(0, n_offspring):
#             if random.random() < crossover_threshold:
#                 toolbox.mate(parent_1, parent_2)
#
#             """ mutation """
#             for mutant in offspring:
#                 if random.random() < mutation_threshold:
#                     toolbox.mutate(mutant)
#
#             total_offspring = np.vstack((total_offspring, offspring[k]))
#
#     return total_offspring


def remove_worst_and_add_diversity(modify_pop, pop_length, population_fit):
    remove_n_samples = int(pop_length/6)
    worst_fitness_scores_indexes = np.argpartition(population_fit, remove_n_samples)[:remove_n_samples]
    modify_pop = np.delete(modify_pop, list(worst_fitness_scores_indexes))

    new_random_samples = np.random.uniform(lower_limit, upper_limit, (remove_n_samples, n_vars))

    modify_pop = np.vstack((modify_pop, new_random_samples))

    return modify_pop, population_fit


# todo: I think we can have it? but we can ask
# todo: when to change to train and when to test

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

    pop = np.random.uniform(lower_limit, upper_limit, (npop, n_vars))
    population_fitness = evaluate(pop)
    best = np.argmax(population_fitness)
    mean = np.mean(population_fitness)
    std = np.std(population_fitness)
    ini_g = 0
    solutions = [pop, population_fitness]
    env.update_solutions(solutions)

else:

    print('\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    population_fitness = env.solutions[1]

    best = np.argmax(population_fitness)
    mean = np.mean(population_fitness)
    std = np.std(population_fitness)

    # finds last generation number
    file_aux = open(experiment_name + '/gen.txt', 'r')
    ini_g = int(file_aux.readline())
    file_aux.close()

# saves results for first pop
file_aux = open(experiment_name + '/results.txt', 'a')
file_aux.write('\n\ngen best mean std')
print('\n GENERATION ' + str(ini_g) + ' ' + str(round(population_fitness[best], 6)) + ' ' + str(
    round(mean, 6)) + ' ' + str(
    round(std, 6)))
file_aux.write(
    '\n' + str(ini_g) + ' ' + str(round(population_fitness[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
        round(std, 6)))
file_aux.close()

# evolution

last_mean = np.mean(population_fitness)
not_improving = 0

for i in range(ini_g + 1, gens):
    print(ini_g)
    print(f"!!!!!!!!!!!! generation number {i}")

    # amount of offspring is not constant in this solution

    """ first do crossover """

    """ IF YOU WANT TO TEST THE SECOND CROSSOVER, CHANGE THE NAME """
    offspring = two_point_crossover_uniform_mutation(pop)

    """ then evaluate the fitness scores """
    fit_offspring = evaluate(offspring)

    """ combine old population with the offspring """
    pop = np.vstack((pop, offspring))

    """ it adds ndarrays horizontally """
    population_fitness = np.append(population_fitness, fit_offspring)

    """ survival selection """

    index_threshold = np.random.uniform(0.5, 0.8, 1)[0]
    best_amount = int(npop * index_threshold)
    worst_amount = int(npop - best_amount)

    best_fitness_scores_indexes = np.argpartition(population_fitness, -best_amount)[-best_amount:]
    worst_fitness_scores_indexes = np.argpartition(population_fitness, worst_amount)[:worst_amount]
    final_indexes = np.hstack((best_fitness_scores_indexes, worst_fitness_scores_indexes))

    """ normalizing - should we use this part of code? """
    # population_fitness_cp = population_fitness
    # population_fitness_norm = np.array(list(map(lambda y: norm(y, population_fitness_cp),
    #                                  population_fitness)))  # avoiding negative probabilities, as fitness is ranges from negative numbers
    # probs = (population_fitness_norm) / (population_fitness_norm).sum()
    #
    # chosen = np.random.choice(pop.shape[0], npop, p=probs, replace=False)
    # chosen = np.append(chosen[1:], best)

    """ update population and fitness scores - OUR CURRENT SURVIVAL SELECTION METHOD """
    pop = pop[final_indexes]
    population_fitness = population_fitness[final_indexes]

    """ statistics about the last fitness """
    best = np.argmax(population_fitness)
    std = np.std(population_fitness)
    current_mean = np.mean(population_fitness)

    """ using mean to decide on additional steps for the diversity """

    if current_mean <= last_mean:
        not_improving += 1
    else:
        last_mean = current_mean
        not_improving = 0

    if not_improving >= 5:
        file_aux = open(experiment_name + '/results.txt', 'a')
        file_aux.write('\nNOT IMPROVING !!!!!!')
        file_aux.close()

        pop, population_fitness = remove_worst_and_add_diversity(pop, population_fitness, npop)
        not_improving = 0

    # saves results
    file_aux = open(experiment_name + '/results.txt', 'a')
    print('\n GENERATION ' + str(i) + ' ' + str(round(population_fitness[best], 6)) + ' ' + str(
        round(mean, 6)) + ' ' + str(
        round(std, 6)))
    file_aux.write(
        '\n' + str(i) + ' ' + str(round(population_fitness[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
            round(std, 6)))
    file_aux.close()

    # saves generation number
    file_aux = open(experiment_name + '/gen.txt', 'w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name + '/best.txt', pop[best])

    # saves simulation state
    solutions = [pop, population_fitness]
    env.update_solutions(solutions)
    env.save_state()

fim = time.time()  # prints total execution time for experiment
print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

file = open(experiment_name + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log()  # checks environment state
