import Agents.MazeAgent as agent
import MazeEnvironment.MazeENV as env
import HyperPerameters as HP
import csv

import numpy as np

"""
    main() -> __init__(self, episodes, steps)
        Generate a new instance

        param: episodes: int : Number of attempts at the env by an agent
        param: steps: int : Number of actions per episode by an agent

    create_agent(episodes, env)
        Create new agent instance

        param: episodes: int : Number of attempts at the env by an agent
        param: env: object: Environment the agent will use
        rtn: agent : object : New agent instance

    create_env(steps)
        Create new environment instance

        param: steps: int : Number of actions per episode by an agent
        rtn: env : object : New environemnt instance    
"""

# Working on
# Need to return the step by step data from each run

# Do this as two pulls ?
# Pass the obstical + goal and agent start state build data with one
# Pass the agent path, score, R.socre, move count as other

# Going to try the split approach
# Now adding in the second api pull

# Agent data format
# <Agent_state, score from last move>

# Need to make a change to build the maze and start the learning seperatly


class main:
    def __init__(self):

        self.episodes = HP.episodes
        self.episode_length = HP.episode_length
        env = self.create_env(self.episode_length)
        self.agent = self.create_agent(self.episodes, env)
        self.writeHypers()

    def writeHypers(self):
        map_size, obstical_locations, goal_locations = HP.mapData()

        agent_path = self.agent.get_highest_fitness_path()
        agent_path_rewards = self.agent.get_highest_fitness_path_rewards()

        print(agent_path_rewards)
        print(agent_path)

        with open("AgentData.csv", "w", newline="") as agent_data:
            writer = csv.writer(agent_data)
            writer.writerow(agent_path)
            writer.writerow(agent_path_rewards)

        with open("BuildData.csv", "w", newline="") as build_data:
            writer = csv.writer(build_data)
            writer.writerow(map_size)
            writer.writerow(obstical_locations)
            writer.writerow(goal_locations)

            # writer.writerow(agent_path)

    def create_env(self, episode_length: int) -> object:
        return env.MazeEnv(episode_length)

    def create_agent(self, episodes: int, env: object) -> object:
        return agent.MazeAgent(episodes, env)


if __name__ == "__main__":
    main()
