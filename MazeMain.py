import MazeAgent as agent
import MazeENV as env


ENV_MAP = [
    [2, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
]


class main:
    def __init__(self, MAP: ENV_MAP):
        self.env = env.MazeEnv(MAP, agent_start_state=12)
        self.agent = agent.MazeAgent(agent_start_state=12)

        self.agent.view_environemnt()
