import numpy as np
from math import sqrt


def find_goal_state(npMap: np.array, ncol: int) -> int:
    goal_state_coords = int(*np.where(npMap == 3))
    goal_state = to_state(goal_state_coords, ncol)
    return goal_state


def find_starting_state(npMap: np.array, ncol: int) -> int:
    start_state_coords = int(*np.where(npMap == 0))
    start_state = to_state(start_state_coords, ncol)
    return start_state


def find_start_state_hunter():
    pass


def to_state(coords: tuple(int, int), ncol: int) -> int:
    state = (coords[0] * ncol) + coords[1]
    return state


def to_coords(state: int, ncol: int) -> tuple(int, int):
    x = int(state / ncol)
    y = int(state % ncol)
    return (x, y)


def to_npMAP(MAP: list) -> np.array:
    npMap = np.array(MAP)
    Map_Dimensions = int(sqrt(len(npMap)))
    npMap = npMap.reshape(Map_Dimensions, Map_Dimensions)
    return npMap
