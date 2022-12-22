from dataclasses import dataclass
import numpy as np
from numpy.random import randn
from math import sqrt
from numpy import dot
import random


from SightData import check_sight_lines
import CustomLogging as CL


To_H_W_Logging = CL.GenerateLogger(__name__ + "weights", "loggingToHiddenWeights.log")

Input_layer_logging = CL.GenerateLogger(
    __name__ + "inputLayer", "loggingInputLayer.log"
)

Hidden_layer_logging = CL.GenerateLogger(
    __name__ + "hiddenLayer", "logginghiddenLayer.log"
)

Output_layer_logging = CL.GenerateLogger(
    __name__ + "ouputLayer", "loggingOutputLayer.log"
)

Genertaion_logger = CL.GenerateLogger(
    __name__ + "Generation", "LoggingNewGenerations.log"
)


class Brain:
    def __init__(self, init_data: tuple):
        self.nrow, self.ncol, self.agent_state, self.env = init_data
        self.agent_coords = self.to_coords(self.agent_state, self.ncol)
        self.Memory: list[dataclass] = []

        self.build_network()

    # // ------------------------------------------------// Build

    def build_network(self):
        # // ---- // Build layers
        self.input_layer = input_layer = np.array([float(i) for i in range(24)])
        self.hidden_layer = hidden_layer = np.array([float(0) for h in range(9)])
        self.output_layer = output_layer = np.array([float(0) for o in range(9)])

        # // ---- // Build weights
        self.weights_inputs_to_hidden = self.generate_weighted_connections(
            input_layer, hidden_layer
        )
        self.weights_hidden_to_output = self.generate_weighted_connections(
            hidden_layer, output_layer
        )

        # Unacking requiers the brackets apparently
        # (
        #    self.weights_inputs_to_hidden,
        #    self.weights_hidden_to_output,
        # ) = self.generate_random_weights(
        #    weights_inputs_to_hidden, weights_hidden_to_output
        # )

        self.new_random_weights()

        To_H_W_Logging.debug(f"Current weights {self.weights_inputs_to_hidden}")

    def generate_weighted_connections(
        self, sending_layer: np.array, reciving_layer: np.array
    ) -> np.array:

        weights = np.array(
            [
                [float(i) for i in range(len(reciving_layer))]
                for x in range(len(sending_layer))
            ]
        )
        return weights

    # debugging this function
    def generate_random_weights(self, *weight_sets: np.array) -> list[np.array]:
        std = sqrt(2.0 / len(self.input_layer))  # Not happy about the self call
        numbers = randn(500)
        scaled = numbers * std
        completed = list(weight_sets)

        for i, set in enumerate(completed):
            completed[i] = [np.random.choice(scaled, len(i)).round(2) for i in set]

        return completed

    # // ------------------------------------------------// # Process

    def process(self, agent_state: int):
        self.input_layer = check_sight_lines(
            agent_state, self.nrow, self.ncol, self.env
        )

        self.hidden_layer = self.calculate_layer(
            self.input_layer, self.weights_inputs_to_hidden
        )
        self.hidden_layer_activation()

        # Need to appy ativation functions at each layer
        self.output_layer = self.calculate_layer(
            self.hidden_layer, self.weights_hidden_to_output
        )
        new_action = self.output_layer_activation()

        Input_layer_logging.debug(f"State: {agent_state} Input: {self.input_layer}")
        Hidden_layer_logging.debug(f"Hidden layer: {self.hidden_layer}")
        Output_layer_logging.debug(
            f" State: {agent_state} Output layer: {self.output_layer} Action: {new_action}"
        )
        return new_action

    def new_random_weights(self):
        """
        Generate new random weights and assign to the weights matrix's
        """
        (
            self.weights_inputs_to_hidden,
            self.weights_hidden_to_output,
        ) = self.generate_random_weights(
            self.weights_inputs_to_hidden, self.weights_hidden_to_output
        )

    def hidden_layer_activation(self):
        layer = self.hidden_layer
        # Applying RElu to each element in hidden layer
        layer = [np.maximum(0, x) for x in np.nditer(layer)]
        self.hidden_layer = np.array(layer)

    def output_layer_activation(self):
        layer = self.output_layer
        # applying sofmax to output layer to return final value
        e = np.exp(layer)
        layer = e / e.sum()
        return np.argmax(layer)

    def calculate_layer(self, inputs: np.array, weights: np.array) -> np.array:
        layer_output = dot(inputs, weights)
        return layer_output

    # // ------------------------------------------------// Generational Learning

    def new_generation(self):
        """
        Create a new generation
        -> Mutation of hidden & output weights of the most sucessful episodes
        """

        # pulling the same every time

        hr = self.get_highest_reward()
        ht = self.get_highest_time_alive()
        # print(hr.episode, ht.episode)

        new_H_W = np.add(hr.H_W, ht.H_W)
        new_O_W = np.add(hr.O_W, ht.O_W)

        # Add variance
        variance = random.uniform(0.0, 0.2)  # Change as needed

        # This will be right
        # new_H_W = np.devide(new_H_W, 2)
        # new_O_W = np.devide(new_O_W, 2)

        # needs teasting
        # new_H_W = np.multiply(new_H_W, variance)
        # new_O_W = np.multiply(new_O_W, variance)

        # new_H_W = new_H_W * variance
        # new_O_W = new_O_W * variance

        new_H_W = np.round(new_H_W, decimals=3)
        new_O_W = np.round(new_O_W, decimals=3)

        Genertaion_logger.info(
            f"Reward: {hr.episode} - Alive: {ht.episode} -- From Reward - Hidden: {hr.H_W[0]} From Alive - Hidden {ht.H_W[0]}"
        )

        Genertaion_logger.debug(
            f"New Generation - Hidden Weights: {new_H_W} \n  Output Weights: {new_O_W}"
        )

        self.weights_inputs_to_hidden = new_H_W
        self.weights_hidden_to_output = new_O_W

    def get_highest_reward(self) -> dataclass:
        """
        Get the memory with the highest reward
        """
        highest_reward = 0
        highest_reward_memory: self.MemoryInstance = dataclass

        for m in self.Memory:
            if m.reward > highest_reward:
                highest_reward = m.reward
                highest_reward_memory = m

        return highest_reward_memory

    def get_highest_time_alive(self) -> dataclass:
        """
        Get reward with the highest time alive
        """
        highest_talive = 0
        highest_t_alive_memory: self.MemoryInstance = dataclass

        for m in self.Memory:
            if m.t_alive > highest_talive:
                highest_talive = m.t_alive
                highest_t_alive_memory = m

        return highest_t_alive_memory

    def commit_to_memory(self, episode: int, reward: float, time_alive: int):
        """
        Commit a new episode to memory
        """
        H_W = self.weights_inputs_to_hidden
        O_W = self.weights_hidden_to_output

        new_memory = self.MemoryInstance(
            episode=episode, reward=reward, t_alive=time_alive, H_W=H_W, O_W=O_W
        )

        self.Memory.append(new_memory)

    def clear_memory(self):
        self.Memory = []

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

    # // ------------------------------------------------// Helper Functions

    # Save the variables of the brain to file
    def store_brain_data():
        pass

    # Convert the state -> (x,y) coords
    def to_coords(self, state: int, ncol: int) -> tuple:
        x = int(state / ncol)
        y = int(state % ncol)
        return (x, y)
