import MazeEnvironment.MazeENV as env
import Brains.MazeAgentBrain as brain
import Logging.CustomLogging as CL
import logging
import decimal

# Set precision to a fixed value
decimal.getcontext().prec = 3


# Basic logging config
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(
    level=logging.NOTSET,
)

Maze_agent_logger = CL.GenerateLogger(__name__, "PathloggingFile.log")
Fitness_Logging = CL.GenerateLogger(__name__ + "fitness", "FitnessloggingFile.log")


class MazeAgent:
    def __init__(self, agent_start_state, EPISODES):
        self.EPISODES = EPISODES
        self.env = env.MazeEnv(agent_start_state)
        self.agent_state = agent_start_state
        self.path = []
        self.fitness_threshold = 1.5
        print("Running")

        self.using_generations = False

        nrow = self.env.nrow
        ncol = self.env.ncol

        # can be cleaned up
        brain_init_data = (ncol, nrow, self.agent_state, self.env)
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
                g: bool

                ns, r, i, t, g = self.env.step(self.agent_state, action)

                reward += r

                if t or g is True:
                    break

                self.agent_state = ns
                self.path.append(ns)

            h = "REWARD FOUND" if g is True else ""

            fitness = round(reward, 3)

            if fitness > self.fitness_threshold:
                self.brain.commit_to_memory(e, reward, step)

            Maze_agent_logger.debug(
                f"Episode: {e} Length: {self.path} Reward: {reward} Fitness: {fitness} {h}"
            )

            Fitness_Logging.debug(f"Fitness: {fitness}")

            if self.using_generations == True:
                self.brain.new_current_generation_weights()
            else:
                self.brain.new_random_weights()

            # Check every 5 if new generation possible
            if e % 5 == 0:
                if self.brain.generation_possible() is False:
                    Maze_agent_logger.info("No new Generation")
                    continue

                self.using_generations = True  # Using gnerations from now
                self.fitness_threshold += 0.1  # Increase threshold over time ?

                # Start a new generation
                self.brain.start_new_generation()
                self.brain.new_current_generation_weights()

                Maze_agent_logger.info(
                    f"New Generation - Fitness Threshold: {self.fitness_threshold}"
                )


if __name__ == "__main__":
    MazeAgent(12)
