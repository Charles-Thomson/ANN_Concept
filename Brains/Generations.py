import numpy as np
import random
import decimal
from Logging import CustomLogging as CL

# Set precision to a fixed value
decimal.getcontext().prec = 3

"""
Functions used in the creation of new generations
"""

parent_logger = CL.GenerateLogger(__name__, "ParentsLogger.log")

# This approach for cross over -- needs testing
def generation_crossover(
    New_Generation_Parents: list[object],
) -> tuple():
    parent_a, parent_b = random.choices(New_Generation_Parents, k=2)

    parent_logger.debug(
        f"Parent A: {parent_a.H_W[0]} Parent B: {parent_b.H_W[0]} , Parents_available: {len(New_Generation_Parents)}"
    )

    # crossover_weight = random.random()

    # New Generation weights
    # new_generation_weight_I_H = crossover_weights(
    #     crossover_weight, parent_a.H_W, parent_b.H_W
    # )

    # new_generation_weight_H_O = crossover_weights(
    #     crossover_weight, parent_a.O_W, parent_b.O_W
    # )

    # New Generation weights using merging
    new_generation_weight_I_H = cross_over_weights_mergining(parent_a.H_W, parent_b.H_W)

    new_generation_weight_H_O = cross_over_weights_mergining(parent_a.O_W, parent_b.O_W)

    parent_logger.debug(f"Resulting new weight : {new_generation_weight_I_H[0]} ")

    if mutation_check():
        random_selection = random.randint(0, 1)
        if random_selection == 0:
            new_generation_weight_I_H = apply_mutation(new_generation_weight_I_H)

        if random_selection == 1:
            new_generation_weight_H_O = apply_mutation(new_generation_weight_H_O)

    return (new_generation_weight_I_H, new_generation_weight_H_O)


def mutation_check() -> bool:
    mutation_chance = random.uniform(0.0, 1.0)
    mutation_threshold = 0.5

    return True if mutation_chance > mutation_threshold else False


# working on this < ----- needs cleaning up
def apply_mutation(weights: np.array) -> np.array:
    """
    Randomly select a weight and "mutate it by +/- 10%"
    """

    weights_shape = weights.shape

    # Select a random col + row
    x = random.randrange(weights_shape[0])
    y = random.randrange(weights_shape[1])

    choosen_weight = weights[x][y]

    mutation_amount = random.randint(1, 10)
    mutated_weight_subtraction = choosen_weight - (choosen_weight / mutation_amount)
    mutated_weight_addition = choosen_weight + (choosen_weight / mutation_amount)

    mutation = random.choice((mutated_weight_subtraction, mutated_weight_addition))

    weights[x][y] = mutation

    return weights


def crossover_weights(
    crossover_weight: float, weight_a: np.array, weight_b: np.array
) -> np.array:
    """
    Generates new weights based on two given weights(np.arrays)

    :param crossover_weight: Float, current weights multiplied against
    :param weight_a: np.array, multiplied by crossover_weight
    :param weight_b: np.array, multiplied by 1 - crossover_weight
    """

    # crossover_weight * weight_a
    weight_a = [[x * crossover_weight for x in y] for y in weight_a]

    crossover_weight = 1 - crossover_weight

    # (1 - crossover_weight) * weight_b
    weight_b = [[x * crossover_weight for x in y] for y in weight_b]

    crossover_weights = np.add(weight_a, weight_b)

    return crossover_weights


# Corss over by selection of random elements from each parent
def cross_over_weights_mergining(weights_a: np.array, weights_b: np.array) -> np.array:
    new_weights = weights_a
    for index_x, x in enumerate(weights_a):
        for index_y, y in enumerate(x):
            selection_chance = random.randrange(1, 100)
            if selection_chance > 50:
                new_weights[index_x][index_y] = weights_b[index_x][index_y]

    new_weights = np.array(new_weights)

    return new_weights
