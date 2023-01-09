import Agents.MazeAgent as agent
import MazeEnvironment.MazeENV as env


ENV_MAP = [
    [2, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
]

"""
    Move the initial creation of the Brains(NN) into the main method then assign
    to each agent
     - Removing work on the threads 
     - Remooving duplicated code in different agent types

     Need an overarching set of weights for each aget type
     - Adjust weights based on each full run 
     - Result should be the final set of weights for each agent
     - Then test on radndom mazes
"""


class main:
    def __init__(self):
        # self.env = env.MazeEnv(MAP, agent_start_state=12)
        self.agent = agent.MazeAgent(agent_start_state=12, EPISODES=200)


if __name__ == "__main__":
    main()
