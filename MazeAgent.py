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


class MazeAgent:
    def __init__(self, agent_state):
        self.EPISODES = 16
        self.env = env.MazeEnv(agent_state)
        self.agent_state = agent_state
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

            for step in range(self.env.EPISODE_LENGTH):
                action = self.brain.process(self.agent_state)

                ns: int
                r: float
                i: list
                t: bool

                ns, r, i, t = self.env.step(self.agent_state, action)

                if t is True:
                    # print("Termination")  # Used for debug
                    self.path.append(ns)
                    break

                reward += r
                self.agent_state = ns
                self.path.append(ns)

            Maze_agent_logger.debug(
                f"Episode: {e} Length: {self.path} Reward: {reward}"
            )

            self.brain.commit_to_memory(e, reward, step)  # if no termination
            self.brain.new_random_weights()

            if e == 5:
                self.brain.new_generation()
                self.brain.clear_memory()

            if e == 10:
                self.brain.new_generation()
                self.brain.clear_memory()

            if e == 15:
                self.brain.new_generation()
                self.brain.clear_memory()


if __name__ == "__main__":
    MazeAgent(12)
