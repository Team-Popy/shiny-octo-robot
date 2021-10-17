import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os
from pathlib import Path


def run_the_whole_experiment(enemy_number, mutate_method, iteration_num):
    path_to_best_solution = "enemies_" + str(enemy_number) + "_" + mutate_method + "_" + str(iteration_num)

    n_hidden_neurons = 10

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    def evaluate(x, environment):
        return np.array(list(map(lambda y: simulation(environment, y), x)))

    def simulation(environment, x):
        fitness, player_life, enemy_life, game_time = environment.play(pcont=x)
        return fitness, player_life, enemy_life, game_time

    def evaluate_the_enemies_5_times(path):
        best_sol = np.loadtxt(path + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        filename = Path(path + "/all_enemies_test.txt")

        for j in range(1, 9):
            if os.path.exists(filename):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'

            test_scores = open(filename, append_write)
            test_scores.writelines('\n' + "-----------------------------------------")
            test_scores.close()
            for i in range(1, 6):
                env = Environment(experiment_name=path_to_best_solution,
                                  enemies=[j],
                                  playermode="ai",
                                  player_controller=player_controller(n_hidden_neurons),
                                  enemymode="static",
                                  level=2,
                                  speed="fastest")

                env.state_to_log()
                ini = time.time()
                env.update_parameter('speed', 'normal')
                fitness_result, player_life_result, enemy_life_result, game_time_result = evaluate([best_sol], env)[0]

                if os.path.exists(filename):
                    append_write = 'a'  # append if already exists
                else:
                    append_write = 'w'

                test_scores = open(filename, append_write)
                gain = player_life_result - enemy_life_result
                test_scores.writelines('\n' + "enemy: " + str(j) + ", fitness: " + str(round(fitness_result, 2)) +
                                       ", player: " + str(player_life_result) + ", enemy: " + str(enemy_life_result) +
                                       ", time: " + str(game_time_result) +
                                       ", gain_number: " + str(i) +
                                       ", gain: " + str(gain))

                test_scores.close()
                fim = time.time()  # prints total execution time for experiment
                print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')
                file = open(path + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
                file.close()
                env.state_to_log()  # checks environment state

    evaluate_the_enemies_5_times(path_to_best_solution)



enemy_number = [7,8]  # enemy group
mutate_method = "deap"  # deap / adaptive

for i in range(1, 11):
    run_the_whole_experiment(enemy_number, mutate_method, i)
