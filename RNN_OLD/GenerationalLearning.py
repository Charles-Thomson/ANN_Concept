import Logging.CustomLogging as CL
import numpy as np

from dataclasses import dataclass

Generation_weights = CL.GenerateLogger(__name__ + "Weights", "GenerationalLogger.log")


class Generation:
    def __init__(self):
        self.memory = list[dataclass]

    def commit_to_memory(
        self, episode: int, reward: float, time_alive: int, H_W: np.array, O_W: np.array
    ):
        """
        Commit a new episode to memory
        """
        new_memory = MemoryInstance(
            episode=episode, reward=reward, t_alive=time_alive, H_W=H_W, O_W=O_W
        )

        self.Memory.append(new_memory)

    def new_generation(self) -> tuple[np.array, np.array]:
        """
        Create a new generation
        -> Mutation of hidden & output weights of the most sucessful episodes
        """
        hs = self.get_highest_reward()
        ht = self.get_highest_time_alive()

        new_H_W = np.dot(hs.H_W, ht.H_W)
        new_O_W = np.dot(hs.O_W, ht.O_W)

        return (new_H_W, new_O_W)

    def get_highest_reward(self) -> dataclass:
        """
        Get the memory with the highest reward
        """
        highest_reward = 0
        highest_reward_memory: dataclass

        for m in self.memory:
            if m.reward > highest_reward:
                highest_reward = m.reward
                highest_reward_memory = m

        return highest_reward_memory

    def get_highest_time_alive(self) -> dataclass:
        """
        Get reward with the highest time alive
        """
        highest_talive = 0
        highest_t_alive_memory: dataclass

        for m in self.memory:
            if m.t_alive > highest_talive:
                highest_talive = m.t_alive
                highest_t_alive_memory = m

        return highest_t_alive_memory


@dataclass
class MemoryInstance:
    """
    Data class to store the data of each episode
    """

    episode: int
    reward: float
    t_alive: int
    H_W: np.array
    O_W: np.array
