from dataclasses import dataclass
import numpy as np
from numpy.random import randn
from math import sqrt
from numpy import dot
import random
from Brains import Generations as Gen


from Agents.SightData import check_sight_lines
import Logging.CustomLogging as CL


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

"""
Take a base set of weights
- Eliteism approach 

1. Take the best of the generation and save them 
2. randomly select pairs to be crossed over

- ngen = parent_a(m) + parent_b(1-m)
- m = random number between 0 - 1 

- Add random mutation 
- Generate a number between 0-1, if it is above a set threshold mutate
- Mutate will adjust a random weight in the ngen by +/- 10% 

 -- past a certain point all the randomness comes from the mutation of the new generation 
    , not from generating completley new weights 



-- Handling generations 
- Load x number of "Fit" objects into memory
- From the fit objects create x new objects
- run again on new objects until x new objects are considered "Fit"
- repeat 

Idea  ... get 10, make 10, get 10,  make 10 <- apply mutation randomly 

Run the new generations until we have 10 that are "Elite", inc the mutation 
- then use that new 10 to build the next generation

process 

 - Run until 10 in memory
 - Save the 10 to "Selected for new generation"
 - run using combinations of "Selected for new generation" and save to memory
 - Once we have 10 in memory, repeat process


 TO DO 
 Reduce the reward for returning to the same location 

"""


class Brain:
    def __init__(self, init_data: tuple):
        self.nrow, self.ncol, self.agent_state, self.env = init_data
        self.agent_coords = self.to_coords(self.agent_state, self.ncol)
        self.Memory: list[dataclass] = []
        self.New_Generation_Parents: list[dataclass] = []

        self.New_Generation_Threshold: int = 5

        self.build_network()

    # // ------------------------------------------------// Build

    # Can maye build this and then assign to an agent
    # Keeps th threads light weight ?
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

        # base weights - changes for each new generation
        self.base_weights_to_hidden = self.weights_inputs_to_hidden
        self.base_weights_to_output = self.weights_hidden_to_output

        self.LR = 0.05

        self.new_random_weights()

        To_H_W_Logging.debug(f"Current weights {self.weights_inputs_to_hidden}")

    def generate_weighted_connections(
        self, sending_layer: np.array, reciving_layer: np.array
    ) -> np.array:

        weights = np.array(
            [
                [float(1) for i in range(len(reciving_layer))]
                for x in range(len(sending_layer))
            ]
        )
        return weights

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

    # // ------------------------------------------------// New Generation

    def generation_possible(self) -> bool:
        """
        Check if a new generation is possible
        Generation Possible if enough objects are in memory i.e have passed the "Elite" threshold

        """
        return True if len(self.Memory) >= self.New_Generation_Threshold else False

    def start_new_generation(self):
        """
        Move the agents that passed the threshold to
        the new generaition parents
        """
        self.New_Generation_Parents = self.Memory
        self.clear_memory()

    def new_current_generation_weights(self):
        """
        Generate new agent from current new generation parents
        """
        new_gen_W_I_H, new_gen_W_H_O = Gen.generation_crossover(
            New_Generation_Parents=self.New_Generation_Parents
        )
        self.weights_inputs_to_hidden = new_gen_W_I_H
        self.weights_hidden_to_output = new_gen_W_H_O

    # // ------------------------------------------------// Memory
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

    # // ------------------------------------------------// Dataclasses

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

    # Convert the state -> (x,y) coords
    def to_coords(self, state: int, ncol: int) -> tuple:
        x = int(state / ncol)
        y = int(state % ncol)
        return (x, y)
