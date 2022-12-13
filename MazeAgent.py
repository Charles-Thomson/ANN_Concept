import MazeENV as env
import MazeAgentBrain as brain
import CustomLogging as CL
import logging

# Basic logging config
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(
    level=logging.NOTSET,
)

Maze_agent_logger = CL.GenerateLogger(__name__, "PathloggingFile.log")

"""
For the NN
28 input = 8 directions * 3 (empty, wall, hunter )

# Need to define a policay for eac agent, the loss will then
        # Adjust the weights based on the difference in out come from the achivment
        # of the policay - so not needing to use a test set of data ?

        # Cross entropy as loss function baed on the policy

        # Rulu used on hidden layer

        # Testing Softmax activation function - used on ouput layer

"""


class MazeAgent:
    def __init__(self, agent_state):
        self.EPISODES = 5
        self.agent_state = agent_state
        self.env = env.MazeEnv(12)  # passing a hard coded start of 12
        self.nrow = self.env.nrow
        self.ncol = self.env.ncol
        self.path = []

        self.memory = []

        brain_init_data = (self.ncol, self.nrow, self.agent_state, self.env)
        self.brain = brain.Brain(brain_init_data)

        self.run_agent()

    def run_agent(self):
        for e in range(self.EPISODES):
            self.agent_state = self.env.reset()
            self.path = []
            reward = 0.0
            self.brain.update_weights()
            for _ in range(self.env.EPISODE_LENGTH):

                action = self.brain.process(self.agent_state)

                n_state, r, i, t = self.env.step(action)
                # Need a guard or something for an invalid action, or kill agent ?
                print(self.agent_state, action, t)

                last_action = action
                # brain.LossFunction(last_action, n_state)

                if t is True:
                    print("Termination")
                    self.path.append(n_state)
                    break

                reward += r
                self.agent_state = n_state
                self.path.append(n_state)

            Maze_agent_logger.debug(
                f"Episode: {e} Length: {self.path} Reward: {reward}"
            )

        # print(len(self.path), reward)
        # self.save_EPISODE(self.path, reward, e)
        # self.memory.append((len(self.path), reward))

    # \\ --------------------------------------------Save episode to file \\
    def save_EPISODE(self, agent_path: list, reward: int, episode: int):
        path_length = str(len(agent_path))
        reward = str(reward)
        episode = str(episode)
        f = open("EpisodeData.txt", "a")
        f.write("\n" + episode)
        f.write(", " + path_length)
        f.write(", " + reward)
        f.close()


if __name__ == "__main__":
    MazeAgent(12)
