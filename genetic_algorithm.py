"""
This is a skeleton file for the implementation of your Genetic Algorithm (GA).
You can add functions to the file and implement your logic inside genetic_algorithm(),
but do not modify the other functions. print_generation_info() must be called on each iteration of the GA.
"""

# <--- ADD ADDITONAL IMPORTS HERE --->
import argparse
import numpy as np
from numba import njit
from numpy.random import shuffle
import time #TODO testing - remove

# <---------------------------------->


# <--- ADD ADDITONAL DEFINES HERE --->

# Population size: Defines how many individuals are in the initial population. (You can change this value)
POPULATION_SIZE = 100 # use this value for the generation of your inital population.
N_SIZE = 1024 # size of the chessboard and the number of queens
MAX_GENERATIONS = 10000 # maximum number of generations
ELITE_SIZE = 15 #15
TOURNAMENT_NUMBER = 15 #15
MUTATION_THRESHOLD = 0.00007 #0.00007
MUTATION_FACTOR = 0.05  #0,05
START_MUTATION_CHANCE = 0.15 #0,15

#31xx Generations


RANDOM_MUTATION_CHANCE = START_MUTATION_CHANCE
MAX_CONFLICTS = (N_SIZE * (N_SIZE - 1)) // 2

#TODO testing - remove
np.random.seed(42) #42
last_time_checkpoint = time.time()

#Todo end testing





# <---------------------------------->


# <--- ADD ADDITONAL FUNCTIONS HERE --->

def gen_initial_population() -> np.ndarray:
    population = np.empty((POPULATION_SIZE, N_SIZE), dtype=np.int64)
    for i in range(POPULATION_SIZE):
        population[i] = np.random.permutation(N_SIZE)

    return population

@njit
def count_conflicts(individual: np.ndarray) -> int:
    n = individual.size
    main_diag = np.zeros(2 * n - 1, dtype=np.int64)
    sec_diag = np.zeros(2 * n - 1, dtype=np.int64)

    for col in range(n):
        row = individual[col]
        main_diag[row - col + n - 1] += 1
        sec_diag[row + col] += 1

    conflicts = 0

    for d in (main_diag, sec_diag):
        for count in d:
            if count > 1:
                conflicts += count * (count - 1) // 2

    return conflicts

def fitness(individual: np.ndarray) -> float:
    return 1 - (count_conflicts(individual) / MAX_CONFLICTS)

def generate_new_generation(elite: np.ndarray, pop_fit_tuples: list) -> np.ndarray:
    new_population = np.empty((POPULATION_SIZE, N_SIZE), dtype=np.int64)
    new_population[:ELITE_SIZE] = elite

    for i in range(ELITE_SIZE, POPULATION_SIZE):
        parent1 = elite[np.random.randint(len(elite))]
        parent2 = select_parent(pop_fit_tuples)
        new_population[i] = reproduce(parent1, parent2)

    return new_population

def select_parent(pop_fit_tuples: list) -> np.ndarray:
    tournament_winners = np.empty((TOURNAMENT_NUMBER, N_SIZE), dtype=np.int64)

    segment_size = len(pop_fit_tuples) // TOURNAMENT_NUMBER

    for i in range(TOURNAMENT_NUMBER):
        start = i * segment_size
        end = start + segment_size
        tournament_winners[i] = max(pop_fit_tuples[start:end], key=lambda x: x[1])[0]

    return tournament_winners[np.random.randint(TOURNAMENT_NUMBER)]

def reproduce(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    child = crossover(parent1, parent2)
    mutated_child = mutate(child.copy())

    fit_child = fitness(child)
    fit_mutated_child = fitness(mutated_child)

    if fit_child < fit_mutated_child:
        return mutated_child

    if np.random.rand() < RANDOM_MUTATION_CHANCE:
        return mutated_child

    return child

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    child = np.full(N_SIZE, -1, dtype=np.int64)

    start, end = sorted(np.random.choice(range(N_SIZE), 2, replace=False))

    child[start:end] = parent1[start:end]

    mapping = {}
    for i in range(start, end):
        mapping[parent1[i]] = parent2[i]

    for i in range(N_SIZE):
        if i >= start and i < end:
            continue

        candidate = parent2[i]
        while candidate in mapping:
            candidate = mapping[candidate]

        child[i] = candidate

    return child

def mutate(individual: np.ndarray) -> np.ndarray:
    first_index = np.random.randint(N_SIZE)
    second_index = np.random.randint(N_SIZE)

    individual[first_index], individual[second_index] = individual[second_index], individual[first_index]

    return individual

# <------------------------------------>

def genetic_algorithm(gui_mode=False):
    """
    Implementation of your genetic algorithm.

    Args:
        gui_mode (bool): If True, run the algorithm with a GUI. Is completly free to you if you want to use that.
    """

    generation = 0
    last_best_fit = 0.0
    population = gen_initial_population()

    #while generation <= MAX_GENERATIONS: TODO original
    while True: #TODO testing remove
        current_pop_fit = np.array([fitness(individual) for individual in population])

        best_fit = np.max(current_pop_fit)
        mean_fit = np.mean(current_pop_fit)
        print_generation_info(generation, best_fit, float(mean_fit))

        pop_fit_tuple = list(zip(population, current_pop_fit))
        pop_fit_tuple.sort(key=lambda x: x[1], reverse=True)

        elite = np.stack([individual for individual, _ in pop_fit_tuple[:ELITE_SIZE]], axis=0)
        pop_fit_tuple = pop_fit_tuple[ELITE_SIZE:]

        shuffle(pop_fit_tuple)

        if best_fit == 1.0:
            print("Solution found!")
            print(pop_fit_tuple[0][0])
            break

        population = generate_new_generation(elite, pop_fit_tuple)

        if (best_fit - last_best_fit) < MUTATION_THRESHOLD:
            global RANDOM_MUTATION_CHANCE
            RANDOM_MUTATION_CHANCE = min(1.0, RANDOM_MUTATION_CHANCE + MUTATION_FACTOR)
        else:
            RANDOM_MUTATION_CHANCE = START_MUTATION_CHANCE

        generation += 1
        last_best_fit = best_fit

def print_generation_info(generation: int, best_fitness: float, mean_fitness: float) -> None:
    """
    Displays the statistics of the current population in a structured format.

    Args:
        generation (int): The current generation number.
        best_fitness (float): The best fitness value in the current population.
        mean_fitness (float): The arithmetic mean (average) fitness value of the current population.
    """

    N = 100  # Print every 100 generations (adjustable)
    if generation % N == 0:

        #Todo debug
        global last_time_checkpoint
        current_time = time.time()
        elapsed = current_time - last_time_checkpoint
        last_time_checkpoint = current_time
        #todo end debug

        #print(f" Generation {generation:>7} | Best Fitness: {best_fitness:.2f} | Mean Fitness: {mean_fitness:.2f} ") #TODO original
        print(f" Generation {generation:>7} | Best Fitness: {best_fitness:.6f} | Mean Fitness: {mean_fitness:.6f} | Δt = {elapsed:.2f}s") #TODO testing - remove


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Genetic Algorithm for solving the 1024-Queens Problem.")
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode for visualization.")

    genetic_algorithm(gui_mode=parser.parse_args().gui)
