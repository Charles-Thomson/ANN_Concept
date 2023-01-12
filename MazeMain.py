import Agents.MazeAgent as agent
import MazeEnvironment.MazeENV as env

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


class main:
    def __init__(self, episodes: int, episode_length: int):
        env = self.create_env(episode_length)
        agent = self.create_agent(episodes, env)

    def create_agent(self, episodes: int, env: object) -> object:
        return agent.MazeAgent(episodes, env)

    def create_env(self, episode_length: int) -> object:
        return env.MazeEnv(episode_length)


if __name__ == "__main__":
    main(episodes=200, episode_length=10)
