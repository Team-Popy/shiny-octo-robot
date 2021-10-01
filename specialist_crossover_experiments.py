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


def run_the_whole_experiment(enemy_number, crossover_method, run_mode, iteration_num, test_number):
    toolbox = base.Toolbox()
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)

    lower_limit = -1
    upper_limit = 1

    population_length = 100
    generations = 30
    crossover_threshold = 0.5
    n_hidden_neurons = 10

    experiment_name = "enemy_" + str(enemy_number) + "_" + crossover_method + "_" + str(iteration_num)

    # removing the visuals
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    """ We chose 2,5, and 8 enemies """
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy_number],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")

    env.state_to_log()  # checks environment state
    ini = time.time()  # sets time marker

    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    def evaluate(x):
        return np.array(list(map(lambda y: simulation(env, y), x)))

    # runs simulation
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

    def tournament_selection(population, fitness_for_tournament):
        # choosing 4 individuals from the population at random
        random_list = random.sample(range(0, population.shape[0]), 4)
        random_val_1 = random_list[0]
        random_val_2 = random_list[1]
        random_val_3 = random_list[2]
        random_val_4 = random_list[3]

        first_fitness = fitness_for_tournament[random_val_1]
        second_fitness = fitness_for_tournament[random_val_2]
        third_fitness = fitness_for_tournament[random_val_3]
        fourth_fitness = fitness_for_tournament[random_val_4]

        sorted_fitness = [first_fitness, second_fitness, third_fitness, fourth_fitness]
        sorted_fitness.sort(reverse=True)
        parent_1_fitness = sorted_fitness[0]
        parent_2_fitness = sorted_fitness[2]

        parent_1_index = list(fitness_for_tournament).index(parent_1_fitness)
        parent_2_index = list(fitness_for_tournament).index(parent_2_fitness)

        return population[parent_1_index], population[parent_2_index]

    """ WEIGHTS INITIALIZATION """
    init_population = np.random.uniform(lower_limit, upper_limit, (population_length, n_vars))

    def limit_the_weights(weight):
        if weight > upper_limit:
            return upper_limit
        elif weight < lower_limit:
            return lower_limit
        else:
            return weight

    def two_points_crossover(population_data, fitness_for_crossover):
        first_point = int(np.random.uniform(0, n_vars, 1)[0])
        second_point = int(np.random.uniform(0, n_vars, 1)[0])
        crossover_point = [first_point, second_point]
        total_offspring = []

        for p in range(0, population_data.shape[0], 2):
            offspring_crossover = np.zeros((2, n_vars))
            parent_1, parent_2 = tournament_selection(population_data, fitness_for_crossover)

            """ crossover """
            for m in crossover_point:
                parent_1, parent_2 = single_point_crossover(parent_1, parent_2, m)

            """ mutation """
            total_offspring = mutate(offspring_crossover, parent_1, parent_2, total_offspring)

        final_total_offspring = np.vstack(total_offspring)
        return final_total_offspring

    def single_point_crossover(parent_1, parent_2, crossover_point):
        parent_1_new = np.append(parent_1[:crossover_point], parent_2[crossover_point:])
        parent_2_new = np.append(parent_2[:crossover_point], parent_1[crossover_point:])
        return parent_1_new, parent_2_new

    def uniform_crossover(population_data, uniform_fitness):
        total_offspring = []

        for p in range(0, population_data.shape[0], 2):
            parent_1, parent_2 = tournament_selection(population_data, uniform_fitness)

            offspring_uniform = np.zeros((2, n_vars))
            """ crossover """
            if random.random() < crossover_threshold:
                parent_1, parent_2 = toolbox.mate(parent_1, parent_2)

            """ mutation """
            total_offspring = mutate(offspring_uniform, parent_1, parent_2, total_offspring)

        final_total_offspring = np.vstack(total_offspring)
        return final_total_offspring

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

    def remove_worst_and_add_diversity(input_pop, pop_length, population_fit):
        remove_n_samples = int(pop_length / 4)
        worst_fitness_scores_indexes = np.argpartition(population_fit, remove_n_samples)[:remove_n_samples]

        cleaned_pop = np.delete(input_pop, worst_fitness_scores_indexes, 0)
        cleaned_fitness = np.delete(population_fit, worst_fitness_scores_indexes, 0)

        new_random_samples = np.random.uniform(lower_limit, upper_limit, (remove_n_samples, n_vars))
        fitness_for_random = evaluate(new_random_samples)

        new_pop = np.vstack((new_random_samples, cleaned_pop))
        new_fitness = np.hstack((fitness_for_random, cleaned_fitness))

        return new_pop, new_fitness

    # loads file with the best solution for testing
    if run_mode == 'test':
        bsol = np.loadtxt(experiment_name + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed', 'normal')
        fitness_result = evaluate([bsol])[0]

        filename = Path(experiment_name + "/tests.txt")

        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        test_scores = open(filename, append_write)
        test_scores.writelines('\n' + str(fitness_result))
        test_scores.close()
        if iteration_num == 10 and test_number == 5:
            sys.exit(0)

    else:
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

        for i in range(ini_g + 1, generations):
            print(ini_g)
            print(f"!!!!!!!!!!!! generation number {i}")

            """ choosing crossover_method """
            if crossover_method == "two_points":
                offspring = two_points_crossover(whole_population, population_fitness)
            else:
                offspring = uniform_crossover(whole_population, population_fitness)

            """ then evaluate the fitness scores """
            fit_offspring = evaluate(offspring)

            """ combine old population with the offspring """
            whole_population = np.vstack((whole_population, offspring))

            """ it adds ndarrays horizontally """
            population_fitness = np.append(population_fitness, fit_offspring)

            """ survival selection """
            index_threshold = np.random.uniform(0.02, 0.05, 1)[0]
            best_amount = int(population_length * index_threshold)
            rest_offspring = int(population_length - best_amount)

            best_fitness_scores_indexes = np.argpartition(population_fitness, -best_amount)[-best_amount:]

            population_fitness_copy = population_fitness.copy()
            population_fitness_normalized = np.array(
                list(map(lambda y: normalization(y, population_fitness_copy), population_fitness)))

            probability = population_fitness_normalized / population_fitness_normalized.sum()
            randomness_population = np.random.choice(whole_population.shape[0], rest_offspring, p=probability,
                                                     replace=False)

            final_indexes = np.hstack((randomness_population, best_fitness_scores_indexes))

            whole_population = whole_population[final_indexes]
            population_fitness = population_fitness[final_indexes].copy()

            """ statistics about the last fitness """
            best = np.argmax(population_fitness)
            std = np.std(population_fitness)
            current_mean = np.mean(population_fitness)
            current_best = np.argmax(population_fitness)
            mean = np.mean(population_fitness)

            """ using mean to decide on additional steps for the diversity """

            if current_best <= last_best:
                not_improving += 1
            else:
                last_best = current_best
                not_improving = 0

            if not_improving >= 3:
                whole_population, population_fitness = remove_worst_and_add_diversity(whole_population, population_length,
                                                                                      population_fitness)
                not_improving = 0

            """ statistics about the last fitness """
            best = np.argmax(population_fitness)
            std = np.std(population_fitness)
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
cross_method = "uniform"
enemy_num = 7

if choose_run_mode == "test":
    for j in range(1, 11):
        for i in range(1, 6):
            run_the_whole_experiment(enemy_num, cross_method, choose_run_mode, j, i)
elif choose_run_mode == "train":
    for i in range(1, 11):
        run_the_whole_experiment(enemy_num, cross_method, choose_run_mode, i, 0)
