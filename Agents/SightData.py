from itertools import chain


def check_sight_lines(agent_state: int, nrow: int, ncol: int, env):
    """
    Covering the 8 directions out from the agent
    """

    state = agent_state
    row, col = to_coords(state, ncol)

    visable_env_data = [[0 for i in range(3)] for x in range(8)]

    for i, data in enumerate(visable_env_data):
        match i:
            case 0:  # Up + Left
                loc = [(row - i, col - i) for i in range(1, col + 1)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 1:  # Up
                loc = [(row - i, col) for i in range(1, row + 1)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 2:  # Up + Right
                loc = [(row - i, col + i) for i in range(1, nrow - row)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 3:  # Left
                loc = [(row, col - i) for i in range(1, col + 1)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 4:  # Right
                loc = [(row, col + i) for i in range(1, nrow - col)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 5:  # Down + Left
                loc = [(row + i, col - i) for i in range(1, col + 1)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 6:  # Down
                loc = [(row + i, col) for i in range(1, nrow - row)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

            case 7:  # Down + Right
                loc = [(row + i, col + i) for i in range(1, nrow - row)]
                visable_env_data[i] = check_sightline(loc, nrow, ncol, env)

    visable_env_data = list(chain(*visable_env_data))

    return visable_env_data


def check_sightline(locations: list, nrow: int, ncol: int, env):
    sightline_data = [0.0, 0.0, 0.0]

    for distance, loc in enumerate(locations):
        x, y = loc

        if not 0 <= x < nrow or not 0 <= y < ncol:
            continue

        value = get_location_value(env, (x, y))

        match value:
            case 1:  # Open
                if distance == 0:
                    sightline_data[0] += 1
                if distance == 1:
                    sightline_data[0] += 0.6
                if distance > 1:
                    sightline_data[0] += 0.2

            case 2:  # Obstical, also can't see through said obstical - rethink this
                if distance == 0:
                    sightline_data[1] += 1
                    break
                if distance == 1:
                    sightline_data[1] += 0.7
                    break
                if distance > 1:
                    sightline_data[1] += 0.5
                    break

            case 3:
                if distance == 0:
                    sightline_data[2] += 5
                if distance == 1:
                    sightline_data[2] += 5
                if distance > 1:
                    sightline_data[2] += 5

    return sightline_data


# Convert the state -> (x,y) coords
def to_coords(state: int, ncol: int) -> tuple:
    x = int(state / ncol)
    y = int(state % ncol)
    return (x, y)


def get_location_value(env: object, coords: tuple) -> int:
    map = env.env_map
    x, y = coords
    value = map[x][y]
    return value
