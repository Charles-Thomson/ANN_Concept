import Agents.MazeAgent as MazeAgent
import MazeEnvironment.MazeENV as env
import HyperPerameters as HP
import csv

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

# Tasks today
# Clean up all the logging bs
# Need to test te front end following the Linting changes


class main:
    def __init__(self):
        env = self.buildEnvirontment(HP.episode_length)
        self.agent = self.buildAgent(HP.episodes, env)
        self.writeBuildData()

    def buildEnvirontment(self, episode_length) -> env:
        return env.MazeEnv(episode_length)

    def buildAgent(self, episodes, env) -> MazeAgent:
        return MazeAgent.MazeAgent(episodes, env)

    def runAgent(self):
        self.agent.run_agent()
        self.writeAgentData(self.agent)
        print("Agent completed run")

    def writeBuildData(self):
        with open("BuildData.csv", "w", newline="") as build_data:
            writer = csv.writer(build_data)
            writer.writerows([*HP.mapData()])

    def writeAgentData(self, agent: object):
        with open("AgentData.csv", "w", newline="") as agent_data:
            writer = csv.writer(agent_data)
            writer.writerows([*agent.get_Highest_Fitness_episode()])

    def getBuildData(self):
        return HP.mapData()

    def getAgentData(self):
        return self.agent.get_Highest_Fitness_episode()


if __name__ == "__main__":
    test = main()
