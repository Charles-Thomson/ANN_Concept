import numpy as np
import random

"""
Functions used in the creation of new generations
"""

# This approach for cross over -- needs testing
def generation_crossover(
    New_Generation_Parents: list[object],
) -> tuple():

    parent_a, parent_b = random.choices(New_Generation_Parents, k=2)

    crossover_weight = random.random()

    # New Generation weights
    new_generation_weight_I_H = crossover_weights(
        crossover_weight, parent_a.H_W, parent_b.H_W
    )

    new_generation_weight_H_O = crossover_weights(
        crossover_weight, parent_a.O_W, parent_b.O_W
    )

    weights_inputs_to_hidden = new_generation_weight_I_H
    weights_hidden_to_output = new_generation_weight_H_O

    if mutation_check():
        weights_inputs_to_hidden, weights_hidden_to_output = apply_mutation(
            new_generation_weight_I_H, new_generation_weight_H_O
        )

    return (weights_inputs_to_hidden, weights_hidden_to_output)


def mutation_check() -> bool:
    mutation_chance = random.uniform(0.0, 1.0)
    mutation_threshold = 0.75

    return True if mutation_chance > mutation_threshold else False


# working on this < ----- needs cleaning up
def apply_mutation(weights_a: np.array, weights_b: np.array) -> np.array:
    """
    Randomly select a weight and "mutate it by +/- 10%"
    """
    select_weight_set = weights_a

    holder = select_weight_set.shape

    # Select a random col + row
    x = random.randrange(holder[0])
    y = random.randrange(holder[1])

    weight = select_weight_set[x][y]
    mutated_weight = weight - (weight / 10)  # hard coded to reduce on mutaion

    select_weight_set[x][y] = mutated_weight

    return select_weight_set, weights_b


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

    crossover_weights = np.round(crossover_weights, decimals=3)

    return crossover_weights
