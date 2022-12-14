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
        self.EPISODES = 1
        self.agent_state = agent_state
        self.env = env.MazeEnv(agent_state)
        self.nrow = self.env.nrow
        self.ncol = self.env.ncol
        self.path = []

        brain_init_data = (self.ncol, self.nrow, self.agent_state, self.env)
        self.brain = brain.Brain(brain_init_data)

        self.run_agent()

    def run_agent(self):
        for e in range(self.EPISODES):
            self.agent_state = self.env.reset()
            self.path = []
            reward = 0.0

            for _ in range(self.env.EPISODE_LENGTH):
                action = self.brain.process(self.agent_state)

                ns: int
                r: float
                i: list
                t: bool

                ns, r, i, t = self.env.step(self.agent_state, action)

                last_action = action
                # brain.LossFunction(last_action, n_state)

                if t is True:
                    print("Termination")  # Used for debug
                    self.path.append(ns)
                    break

                reward += r
                self.agent_state = ns
                self.path.append(ns)

            Maze_agent_logger.debug(
                f"Episode: {e} Length: {self.path} Reward: {reward}"
            )


if __name__ == "__main__":
    MazeAgent(12)
