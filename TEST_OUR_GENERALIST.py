""" GENERALIST """
""" for multiple enemies the default fitness function is this (it first creates a list of values per each enemy) 
values.mean() - values.std()
"""

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

""" 'train' TO START THE EVOLUTION or 'test' TO TEST THE RESULTS  """
path_to_best_solution = "enemy_test_GENERALIST_mutation_0.3_parents_changed"

n_hidden_neurons = 10

# removing the visuals
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# initializes simulation in individual evolution mode, for single static enemy.


env.state_to_log()  # checks environment state
ini = time.time()  # sets time marker

n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5


def evaluate(x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


# runs simulation
def simulation(environment, x):
    fitness, player_life, enemy_life, game_time = environment.play(pcont=x)
    return fitness, player_life, enemy_life, game_time


def evaluate_the_enemies_5_times(path):
    best_sol = np.loadtxt(path + '/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')

    for i in range(1, 6):
        for j in range(1, 11):
            env = Environment(experiment_name=path_to_best_solution,
                              enemies=[j],
                              playermode="ai",
                              player_controller=player_controller(n_hidden_neurons),
                              enemymode="static",
                              level=2,
                              speed="fastest")

            env.update_parameter('speed', 'normal')
            fitness_result, player_life_result, enemy_life_result, game_time_result = evaluate([best_sol])[0]
            filename = Path(path + "/all_enemies_test.txt")

            if os.path.exists(filename):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'  # make a new file if not

            test_scores = open(filename, append_write)
            test_scores.writelines('\n' + str(fitness_result) + str(player_life_result) + str(enemy_life_result) +
                                   str(game_time_result))
            test_scores.close()

            fim = time.time()  # prints total execution time for experiment
            print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

            file = open(path + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
            file.close()

            env.state_to_log()  # checks environment state
