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
Fitness_Logging = CL.GenerateLogger(__name__ + "fitness", "FitnessloggingFile.log")


class MazeAgent:
    def __init__(self, agent_state):
        self.EPISODES = 1000
        self.env = env.MazeEnv(agent_state)
        self.agent_state = agent_state
        self.nrow = self.env.nrow
        self.ncol = self.env.ncol
        self.path = []
        self.fitness_threshold = 1.5
        print("Running")

        self.using_generations = False

        brain_init_data = (self.ncol, self.nrow, self.agent_state, self.env)
        self.brain = brain.Brain(brain_init_data)

        self.run_agent()

    def run_agent(self):
        for e in range(self.EPISODES):
            self.agent_state = self.env.reset()
            self.path = []
            reward = 1.0
            fitness = 0.0

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
                reward = round(reward, 3)  # Keep to meaningful values
                self.agent_state = ns
                self.path.append(ns)

            h = ""

            if reward > 100:
                h = "REWARD FOUND"

            # Commit to memory the "best" in terms of reward or time alive

            # Does step need to be considered for fitness ?
            fitness = round(reward, 3)

            if fitness > self.fitness_threshold:
                self.brain.commit_to_memory(e, reward, step)

            Maze_agent_logger.debug(
                f"Episode: {e} Length: {self.path} Reward: {reward} Fitness: {fitness} {h}"
            )

            Fitness_Logging.debug(f"Fitness: {fitness}")

            if self.using_generations == True:
                self.brain.generation_crossover()
            else:
                self.brain.new_random_weights()

            # Check every 5 runs if we have enough in memory for new generation
            if e % 5 == 0:
                if self.brain.generation_possible():  # if there is enough in memeory
                    self.using_generations = True  # Using gnerations from now
                    self.fitness_threshold += 0.1  # Increase threshold over time ?
                    # Start a new generation
                    self.brain.start_new_generation()
                    Maze_agent_logger.info(
                        f"New Generation - Fitness Thresholg: {self.fitness_threshold}"
                    )
                else:
                    Maze_agent_logger.info("No new Generation")


if __name__ == "__main__":
    MazeAgent(12)
