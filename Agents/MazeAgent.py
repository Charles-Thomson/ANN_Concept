import MazeEnvironment.MazeENV as env
import Brains.MazeAgentBrain as brain
import Logging.CustomLogging as CL
import logging
import decimal
from Agents import SightData

# Basic logging config
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(
    level=logging.NOTSET,
)

# Set precision to a fixed value
decimal.getcontext().prec = 3

Maze_agent_logger = CL.GenerateLogger(__name__, "PathloggingFile.log")
Fitness_Logging = CL.GenerateLogger(__name__ + "fitness", "FitnessloggingFile.log")

"""
    MazeAgent() -> __init__(episodes, env)
        Init of a new agent

        param: episodes: int : Number of attempts at the env by an agent
        param: env: object: Environment the agent will use

        var: self.episode : int : Number of attempts at the env by an agent
        var: self.env : object : object: Environment the agent will use
        var: self.agent_state : int : Te current state of the agent in the env 
        var: self.nrow : int : number of rows in the enviroment
        var: self.ncol : int : number of columns in the enviroment
        var: self.brain : object : Brain instance used by this agent

        call: self.run_agent() : Start the agent - start learning

    run_agent(episodes, env)
        The main process of the agent 

        param: episodes: int : Number of attempts at the env by an agent
        
        var: using_generations : bool : Indicates if non random weights are being used
        var: fitness_threshold : float : "Elite" threshold for agent to be considered for new generation

        var: agent_state: int : The current state of the agent
        var: path : list[int] : The path taken by the agent in the current episode
        var: reward : float : The accumulated reward by the agent in the current episode

        var: sight_line_data: list[list[float]] : Resulting data of what is visable to the agent along the 8 sightlines
        var: action : int : The action to be taken based on the sight_line_data 
        var: ns : int : The new state of the agent following the last action 
        var: r : float : The reward from the previous action
        var: i : list : Requierment of Gym - not used
        var: t : bool : Indecates if the last action resulted in termination of the agent in this episode
        var: g : bool : Indecates if the last action resulted in reaching the "goal" state


"""


class MazeAgent:
    def __init__(self, episodes: int, env: object):
        self.env = env
        self.agent_state = env.get_agent_state()  # Need to create

        self.nrow, self.ncol = env.get_env_shape()  # need to create

        self.brain = brain.Brain()  # Clean up brain needed

        self.run_agent(episodes)

    def run_agent(self, episodes: int):
        print("Running")  # debug

        using_generations = False
        fitness_threshold = 1.6

        for e in range(episodes):
            self.env.reset()
            agent_state = self.env.get_agent_state()
            path = []
            reward = 1.0

            for step in range(self.env.episode_length):

                sight_line_data = SightData.check_sight_lines(
                    agent_state, self.nrow, self.ncol, self.env
                )

                action = self.brain.determine_action(sight_line_data)

                ns, r, i, t, g = self.env.step(agent_state, action)

                reward += r

                if t or g is True:
                    break

                agent_state = ns
                path.append(ns)

            if reward > fitness_threshold:
                self.brain.commit_to_memory(e, reward, step)

            if using_generations == True:
                self.brain.new_current_generation_weights()
            else:
                self.brain.new_random_weights()

            # Check every 5 if new generation possible
            if e % 5 == 0 and self.brain.new_generation():
                using_generations = True  # Using gnerations from now
                fitness_threshold += 0.1  # Increase threshold over time ?
                self.brain.new_current_generation_weights()

                Maze_agent_logger.info(
                    f"New Generation - Fitness Threshold: {fitness_threshold}"
                )

            Maze_agent_logger.debug(
                f"Episode: {e} Length: {path} Reward: {reward} Fitness: {reward} {'REWARD FOUND' if g is True else ''} "
            )

            Fitness_Logging.debug(f"Fitness: {reward}")


if __name__ == "__main__":
    MazeAgent(12)
