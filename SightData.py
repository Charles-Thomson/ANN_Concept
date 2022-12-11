from itertools import chain


def check_sight_lines(agent_state: int, nrow: int, ncol: int, env):
    state = agent_state
    row, col = to_coords(state, ncol)

    visable_env_data = [[0 for i in range(3)] for x in range(8)]

    for i, data in enumerate(visable_env_data):
        match i:
            case 0:  # Up + Left
                loc = [(row - i, col - i) for i in range(col + 1)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 1:  # Up
                loc = [(row - i, col) for i in range(row + 1)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 2:  # Up + Right
                loc = [(row - i, col + i) for i in range(nrow - row)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 3:  # Left
                loc = [(row, col - i) for i in range(col + 1)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 4:  # No move
                loc = [(row, col)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 5:  # Right
                loc = [(row, col + i) for i in range(nrow - col)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 6:  # Down + Left
                loc = [(row + i, col - i) for i in range(col + 1)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 7:  # Down
                loc = [(row + i, col) for i in range(nrow - row)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 8:  # Down + Right
                loc = [(row + i, col + i) for i in range(nrow - row)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

    visable_env_data = list(chain(*visable_env_data))

    return visable_env_data


# Pass in a list of values to check

"""
1 -> open 
2 -> obstical - kill agent 
3 -> goal
"""


def check_sightline(locations: list, nrow: int, ncol: int, env):
    sightline_data = [0.0, 0.0, 0.0]

    for distance, loc in enumerate(locations):
        x, y = loc

        if not 0 <= x < nrow or not 0 <= y < ncol:
            continue

        value = env.get_location_value_call((x, y))

        match value:
            case 1:  # Open
                if distance == 0:
                    sightline_data[0] += 0.6
                if distance == 1:
                    sightline_data[0] += 0.3
                if distance > 1:
                    sightline_data[0] += 0.1

            case 2:  # Obstical, also can't see through said obstical
                if distance == 0:
                    sightline_data[1] += 0.6
                    break
                if distance == 1:
                    sightline_data[1] += 0.3
                    break
                if distance > 1:
                    sightline_data[1] += 0.1
                    break

            case 3:
                if distance == 0:
                    sightline_data[2] += 1
                if distance == 1:
                    sightline_data[2] += 1
                if distance > 1:
                    sightline_data[2] += 1

    return sightline_data


# Convert the state -> (x,y) coords
def to_coords(state: int, ncol: int) -> tuple:
    x = int(state / ncol)
    y = int(state % ncol)
    return (x, y)
