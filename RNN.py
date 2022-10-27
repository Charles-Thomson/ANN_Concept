import numpy as np
import RNN_ENV as cenv


from dataclasses import dataclass

# Define the env
# env = cenv.MazeEnv()


@dataclass
class neural_network_parameters:

    # Steps in the episode
    EPISODES = 30000

    # Weight put on possile future reward
    GAMMA = 0.96

    # Update rate of the Q-Table - higher = faster learning / more changes to Q-table
    LEARNING_RATE = 0.98

    # ratio of random to choosen agent moves - decreases as Q_Table is filled
    epsilon = 0.8


class Q_LAEARNING_PROCESS:
    def __init__(self, MAP):
        # Define the env
        self.env = cenv.MazeEnv(MAP)
        parameters = neural_network_parameters

        # neural network parameters
        self.EPISODES = parameters.EPISODES
        self.GAMMA = parameters.GAMMA
        self.LEARNING_RATE = parameters.LEARNING_RATE
        self.epsilon = parameters.epsilon

        # Environemnt states/agent actions
        self.STATES = self.env.observation_space.n
        self.ACTIONS = self.env.action_space.n

        # Q_TABLE
        self.Q_TABLE = np.zeros((self.STATES, self.ACTIONS))

        # Path data
        self.goal_reached_on_episode = []
        self.path_to_goal_used = []

        self.env.render()
        self.main_process()

        # Print the shortest path used - [path]len(path)
        self.path_data = self.process_path_data()

    # Needed by API ?
    def return_path_data(self) -> tuple:
        return self.path_data

    def main_process(self):
        for episode in range(self.EPISODES):
            state = self.env.reset()
            path = []  # Tracking the overall path taken by the agent
            path.append(state)

            reward_tracker = 0
            q_table_flag = False

            for step in range(self.env.EPISODE_LENGTH):

                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(
                        self.Q_TABLE[state, :]
                    )  # Get the highest reward for the action in the current state
                    q_table_flag = True

                next_state, reward, terminated, info, goal_reached = self.env.step(
                    action
                )

                if terminated:
                    self.epsilon -= 0.00001  # Reduce the randomness of actions
                    self.env.reset()
                    break

                # Update Q_Table alg
                self.update_Q_table(
                    self.Q_TABLE,
                    state,
                    action,
                    self.LEARNING_RATE,
                    reward,
                    self.GAMMA,
                    next_state,
                )

                if goal_reached:
                    print(
                        f"EPISODE - {episode} -  REWARD : {reward_tracker}  Q_TABLE used : {q_table_flag}  ******* GOAL REACHED *********"
                    )
                    path.append(next_state)
                    self.goal_reached_on_episode.append(episode)
                    self.path_to_goal_used.append(path)
                    self.env.reset()
                    break

                state = next_state

                path.append(state)

                reward_tracker += reward

            print(
                f"EPISODE - {episode} -  REWARD : {reward_tracker}  Q_TABLE used : {q_table_flag}"
            )
        print(self.path_to_goal_used)
        # self.path_info()

    def update_Q_table(
        self, Q_TABLE, state, action, LEARNING_RATE, reward, GAMMA, next_state
    ) -> None:
        Q_Table_update_value = Q_TABLE[state, action] + LEARNING_RATE * (
            reward + GAMMA * np.max(Q_TABLE[next_state, :]) - Q_TABLE[state, action]
        )

        Q_TABLE[state, action] = Q_Table_update_value

    def process_path_data(self) -> None:
        if not self.path_to_goal_used:
            return [], 0

        best_path = self.path_to_goal_used[0]

        for path in self.path_to_goal_used:
            if len(path) < len(best_path):
                best_path = path

        shortes_route_length = min(len(path) for path in self.path_to_goal_used)

        return best_path  #  shortes_route_length


if __name__ == "__main__":
    """
    # MAP
    The current map being used is hard coded in the format of 5 x 5
    Map tiles;
    - 0 = Start position
    - 1 = Open tile
    - 2 = wall/obstical
    - 3 = Finish/ goal
    """
    MAP = [
        [0, 1, 1, 2, 3],
        [1, 1, 2, 2, 1],
        [1, 1, 2, 2, 1],
        [1, 1, 1, 2, 1],
        [1, 2, 1, 1, 1],
    ]
    Q_LAEARNING_PROCESS(MAP)
