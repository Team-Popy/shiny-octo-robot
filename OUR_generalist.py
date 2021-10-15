""" GENERALIST """
""" for multiple enemies the default fitness function is this (it first creates a list of values per each enemy) 
values.mean() - values.std()
"""

# todo: check results when starting with init pop from previous task

import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os
from deap import base
import random
from deap import tools
from pathlib import Path


def run_the_whole_experiment(enemy_number, mutate_method, iteration_num):
    """ set experiment name """
    experiment_name = "enemies_" + str(enemy_number) + "_" + mutate_method + "_" + str(iteration_num)

    """ set mutation settings """
    toolbox = base.Toolbox()
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)

    """ set experiment parameters """
    population_length = 50
    generations = 20

    """ constant parameters """
    n_hidden_neurons = 10
    lower_limit = -1
    upper_limit = 1
    mutation_threshold = 0.2
    upper_sig = 0
    lower_sig = 0.5

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # todo: next [2,6]

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=enemy_number,
                      multiplemode="yes",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")

    env.state_to_log()  # checks environment state
    ini = time.time()  # sets time marker

    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    # todo: decide if to use random seed

    def evaluate(x):
        return np.array(list(map(lambda y: simulation(env, y), x)))

    def simulation(environment, x):
        fitness, player_life, enemy_life, game_time = environment.play(pcont=x)
        return fitness

    def normalization(x, pop_fitness):
        denominator_check = max(pop_fitness) - min(pop_fitness)
        if denominator_check > 0:
            x_norm = (x - min(pop_fitness)) / (max(pop_fitness) - min(pop_fitness))

            if x_norm <= 0:
                x_norm = 0.0000000001
        else:
            x_norm = 0.0000000001
        return x_norm

    def get_best_parent_for_tournament(population_data, fitness_data):
        fitness_parents_dict = {}
        random_list = random.sample(range(0, population_data.shape[0]), 4)
        random_val_1 = random_list[0]
        random_val_2 = random_list[1]
        random_val_3 = random_list[2]
        random_val_4 = random_list[3]
        fitness_parents_dict = {random_val_1: fitness_data[random_val_1], random_val_2: fitness_data[random_val_2],
                                random_val_3: fitness_data[random_val_3], random_val_4: fitness_data[random_val_4]}
        max_fitness_index = list(sorted(fitness_parents_dict, key=lambda k: (fitness_parents_dict[k], k)))[-1]
        max_fitness = fitness_parents_dict[max_fitness_index]
        return population_data[max_fitness_index], max_fitness

    def tournament_selection(population, fitness_for_tournament):
        final_parent, parent_fitness = get_best_parent_for_tournament(population, fitness_for_tournament)
        return final_parent, parent_fitness

    """ WEIGHTS INITIALIZATION """
    init_population = np.random.uniform(lower_limit, upper_limit, (population_length, n_vars))

    def limit_the_weights(weight):
        if weight > upper_limit:
            return upper_limit
        elif weight < lower_limit:
            return lower_limit
        else:
            return weight

    def limit_the_sigma(weight):
        if weight > upper_sig:
            return upper_sig
        elif weight < lower_sig:
            return lower_sig
        else:
            return weight

    def two_points_crossover(population_data, fitness_for_crossover, generation):
        first_point = int(np.random.uniform(0, n_vars, 1)[0])
        second_point = int(np.random.uniform(0, n_vars, 1)[0])
        crossover_point = [first_point, second_point]
        total_offspring = []
        mutation_rate = 0.4
        mutation_value = 0.4

        for p in range(0, population_data.shape[0], 2):
            offspring_crossover = np.zeros((2, n_vars))
            parent_1, parent_1_fitness = tournament_selection(population_data, fitness_for_crossover)
            parent_2, parent_2_fitness = tournament_selection(population_data, fitness_for_crossover)

            """ crossover """
            for m in crossover_point:
                parent_1, parent_2 = single_point_crossover(parent_1, parent_2, m)

            """ mutation """
            if mutate_method == "deap":
                total_offspring = mutate(offspring_crossover, parent_1, parent_2, total_offspring)
            elif mutate_method == "adaptive":
                total_offspring, mutation_rate = mutate_adapted_rate_evaluate(offspring_crossover, parent_1, parent_2,
                                                                              fitness_for_crossover, total_offspring,
                                                                              mutation_rate, generation)
            else:
                total_offspring = [parent_1, parent_2]

        final_total_offspring = np.vstack(total_offspring)
        return final_total_offspring

    def single_point_crossover(parent_1, parent_2, crossover_point):
        parent_1_new = np.append(parent_1[:crossover_point], parent_2[crossover_point:])
        parent_2_new = np.append(parent_2[:crossover_point], parent_1[crossover_point:])
        return parent_1_new, parent_2_new

    def mutate(offspring_uniform, parent_1, parent_2, total_offspring):
        offspring_uniform[0] = parent_1.copy()
        offspring_uniform[1] = parent_2.copy()

        mutated_offspring_1 = toolbox.mutate(offspring_uniform[0])
        mutated_offspring_2 = toolbox.mutate(offspring_uniform[1])

        mutated_offspring_1 = np.array(list(map(lambda y: limit_the_weights(y), mutated_offspring_1[0])))
        mutated_offspring_2 = np.array(list(map(lambda y: limit_the_weights(y), mutated_offspring_2[0])))

        total_offspring.append(mutated_offspring_1)
        total_offspring.append(mutated_offspring_2)

        return total_offspring

    def mutate_adapted_rate_evaluate(offspring_uniform, parent_1, parent_2, fitness_for_crossover,
                                     total_offspring, mutation_rate, generation):
        offspring_uniform[0] = parent_1.copy()
        offspring_uniform[1] = parent_2.copy()

        avg_population_fitness = np.average(fitness_for_crossover)

        if generation % 3 == 0:
            new_fitness = evaluate(offspring_uniform)

            print("------------------------", mutation_rate)
            mutation_rate = check(avg_population_fitness, mutation_rate, new_fitness, 0)
            mutated_offspring_1 = mutate_rate(mutation_rate, offspring_uniform[0])

            mutation_rate = check(avg_population_fitness, mutation_rate, new_fitness, 1)
            mutated_offspring_2 = mutate_rate(mutation_rate, offspring_uniform[1])

        else:
            mutated_offspring_1 = mutate_rate(mutation_rate, offspring_uniform[0])
            mutated_offspring_2 = mutate_rate(mutation_rate, offspring_uniform[1])

        mutated_offspring_1 = np.array(list(map(lambda y: limit_the_weights(y), mutated_offspring_1)))
        mutated_offspring_2 = np.array(list(map(lambda y: limit_the_weights(y), mutated_offspring_2)))

        total_offspring.append(mutated_offspring_1)
        total_offspring.append(mutated_offspring_2)

        return total_offspring, mutation_rate

    def mutate_adapted_rate_evaluate_sigma(offspring_uniform, parent_1, parent_2,
                                           total_offspring, generation):
        offspring_uniform[0] = parent_1.copy()
        offspring_uniform[1] = parent_2.copy()

        mutated_offspring_1 = mutate_rate_sigma(mutation_threshold, offspring_uniform[0])
        mutated_offspring_2 = mutate_rate_sigma(mutation_threshold, offspring_uniform[1])

        mutated_offspring_1 = np.array(list(map(lambda y: limit_the_weights(y), mutated_offspring_1)))
        mutated_offspring_2 = np.array(list(map(lambda y: limit_the_weights(y), mutated_offspring_2)))

        total_offspring.append(mutated_offspring_1)
        total_offspring.append(mutated_offspring_2)

        return total_offspring

    def check(avg_population_fitness, mutation, new_fitness, parent_number):
        parent_fitness = new_fitness[parent_number]
        if parent_fitness < avg_population_fitness:
            mutation += 0.0001
        elif mutation > 0.0001:
            mutation -= 0.0001
        else:
            mutation = 0.0001
        return mutation

    def mutate_rate(mutation_rate, parent_offspring):
        for k in range(0, len(parent_offspring)):
            if random.random() <= mutation_rate:
                parent_offspring[k] = parent_offspring[k] + np.random.normal(0, 0.2)

        return parent_offspring

    def mutate_rate_sigma(parent_offspring):
        for k in range(0, len(parent_offspring)):
            if random.random() <= mutation_threshold:
                sig = parent_offspring[len(parent_offspring) - 1]
                parent_offspring[k] = parent_offspring[k] + np.random.normal(0, limit_the_sigma(sig))
                print("sigma = ", sig)
        return parent_offspring

    def simplest_hybridization_climbing_hill(population, population_fit, climbing_index):
        best_fitness_indexes = np.argpartition(population_fit, -4)[-4:]

        for best_index in best_fitness_indexes:
            first_fitness = population_fit[best_index]
            first_individual = population[best_index]

            new_fitness = first_fitness.copy()
            new_individual = first_individual.copy()
            climb_mutation_rate = 0.2
            amount_of_climbing = 0
            while new_fitness <= first_fitness:
                for p in range(len(first_individual)):
                    if random.random() <= climb_mutation_rate:
                        new_individual[p] = first_individual[p] + np.random.normal(0, 0.25)

                new_fitness = evaluate([new_individual])
                amount_of_climbing += 1
                if amount_of_climbing > climbing_index:
                    break
                print("CLIMBING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", amount_of_climbing, first_fitness,
                      new_fitness)

            if new_fitness > first_fitness:
                cleaned_pop = np.delete(population, best_index, 0)
                cleaned_fitness = np.delete(population_fit, best_index, 0)
                population = np.vstack((new_individual, cleaned_pop))
                population_fit = np.hstack((new_fitness, cleaned_fitness))

        climbing_index -= 1
        return population, population_fit, climbing_index

    def elitism_attemp(population_data, fitness_data):
        elitism_ratio = np.random.uniform(0.01, 0.2, 1)[0]
        new_population = int(elitism_ratio * population_data.shape[0])

        elitist_indices = np.argpartition(fitness_data, -new_population)[-new_population:]
        elitist_members = population_data[elitist_indices]
        elitist_members_fitness = fitness_data[elitist_indices]

        return elitist_members, elitist_members_fitness

    def probability_survival_selection(population_data, fitness_data, offspring_data):
        """ combine old population with the offspring """
        population_data = np.vstack((population_data, offspring_data))

        """ it adds ndarrays horizontally """
        fitness_data = np.append(fitness_data, fit_offspring)
        index_threshold = np.random.uniform(0.02, 0.05, 1)[0]
        best_amount = int(population_length * index_threshold)
        rest_offspring = int(population_length - best_amount)

        best_fitness_scores_indexes = np.argpartition(fitness_data, -best_amount)[-best_amount:]

        population_fitness_copy = fitness_data.copy()
        population_fitness_normalized = np.array(
            list(map(lambda y: normalization(y, population_fitness_copy), fitness_data)))

        probability = population_fitness_normalized / population_fitness_normalized.sum()
        randomness_population = np.random.choice(population_data.shape[0], rest_offspring, p=probability,
                                                 replace=False)

        final_indexes = np.hstack((randomness_population, best_fitness_scores_indexes))

        final_population = population_data[final_indexes]
        final_fitness = fitness_data[final_indexes].copy()
        return final_population, final_fitness

    # initializes population loading old solutions or generating new ones
    if not os.path.exists(experiment_name + '/evoman_solstate'):

        print('\nNEW EVOLUTION\n')

        whole_population = np.random.uniform(lower_limit, upper_limit, (population_length, n_vars))
        first_population_fitness = evaluate(whole_population)
        best = np.argmax(first_population_fitness)
        mean = np.mean(first_population_fitness)
        std = np.std(first_population_fitness)
        ini_g = 0
        solutions = [whole_population, first_population_fitness]
        env.update_solutions(solutions)

    else:

        print('\nCONTINUING EVOLUTION\n')

        env.load_state()
        whole_population = env.solutions[0]
        first_population_fitness = env.solutions[1]

        best = np.argmax(first_population_fitness)
        mean = np.mean(first_population_fitness)
        std = np.std(first_population_fitness)

        # finds last generation number
        file_aux = open(experiment_name + '/gen.txt', 'r')
        ini_g = int(file_aux.readline())
        file_aux.close()

    # saves results for first pop
    file_aux = open(experiment_name + '/results.txt', 'a')
    file_aux.write('\n\ngen best mean std')
    print('\n GENERATION ' + str(ini_g) + ' ' + str(round(first_population_fitness[best], 6)) + ' ' + str(
        round(mean, 6)) + ' ' + str(
        round(std, 6)))
    file_aux.write(
        '\n' + str(ini_g) + ' ' + str(round(first_population_fitness[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
            round(std, 6)))
    file_aux.close()

    # evolution ********************************************************

    last_mean = np.mean(first_population_fitness)
    last_best = np.argmax(first_population_fitness)
    not_improving = 0

    population_fitness = first_population_fitness.copy()
    climbing_index = 20

    for i in range(ini_g + 1, generations):
        print(ini_g)
        print(f"!!!!!!!!!!!! generation number {i}")

        # 5 elite individuals (elitism)
        elite_members, elite_members_fitness = elitism_attemp(whole_population, population_fitness)

        """ choosing crossover_method """
        offspring = two_points_crossover(whole_population, population_fitness, i)

        """ then evaluate the fitness scores """
        fit_offspring = evaluate(offspring)

        """ does replacement """
        whole_population, population_fitness, climbing_index = simplest_hybridization_climbing_hill(whole_population,
                                                                                                    population_fitness,
                                                                                                    climbing_index)

        whole_population, population_fitness = probability_survival_selection(whole_population, population_fitness,
                                                                              offspring)

        total_population_with_elites = np.vstack((whole_population, elite_members))
        total_fitness_with_elites = np.hstack((population_fitness, elite_members_fitness))

        # best 100 individuals from 105 based on their fitness
        best_100_indexes = np.argpartition(total_fitness_with_elites, -population_length)[-population_length:]
        whole_population_with_elite = total_population_with_elites[best_100_indexes]
        elite_fitness = total_fitness_with_elites[best_100_indexes]

        """ statistics about the last fitness """
        best = np.argmax(population_fitness)
        std = np.std(population_fitness)
        current_mean = np.mean(population_fitness)
        current_best = np.argmax(population_fitness)
        mean = np.mean(population_fitness)

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
        np.savetxt(experiment_name + '/best.txt', whole_population[best])

        # saves simulation state
        solutions = [whole_population, population_fitness]
        env.update_solutions(solutions)
        env.save_state()

    fim = time.time()  # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

    file = open(experiment_name + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()

    env.state_to_log()  # checks environment state


""" 'train' TO START THE EVOLUTION or 'test' TO TEST THE RESULTS  """
choose_run_mode = 'train'

""" CHOOSE THE NAME OF THE CROSSOVER METHOD 'uniform' or 'two_points' """
mutation_method = "deap"

enemies_num = [7, 8]

for i in range(1, 11):
    run_the_whole_experiment(enemies_num, mutation_method, i)
