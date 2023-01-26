from dataclasses import dataclass
import decimal
import numpy as np
from numpy.random import randn
from math import sqrt
from numpy import dot
from Brains import Generations as Gen
import HyperPerameters

# Set precision to a fixed value
decimal.getcontext().prec = 3


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

New_generation_weights_logger = CL.GenerateLogger(
    __name__ + "NewGenWeights", "loggingNewGenerationWeights.log"
)


class Brain:
    def __init__(self):
        self.Memory: list[dataclass] = []
        self.New_Generation_Parents: list[dataclass] = []
        self.New_Generation_Threshold: int = HyperPerameters.New_Generation_Threshold

        self.build_network()

    # // ------------------------------------------------// Build

    # Can maye build this and then assign to an agent
    # Keeps th threads light weight ?
    def build_network(self):
        # // ---- // Build layers
        self.input_size = 24
        input_layer = np.array([float(i) for i in range(24)])
        hidden_layer = np.array([float(0) for h in range(9)])
        output_layer = np.array([float(0) for o in range(9)])

        # // ---- // Build weights
        self.weights_inputs_to_hidden = self.generate_weighted_connections(
            input_layer, hidden_layer
        )
        self.weights_hidden_to_output = self.generate_weighted_connections(
            hidden_layer, output_layer
        )
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

    def generate_random_weights(self, *weight_sets: np.array) -> list[np.array]:
        std = sqrt(2.0 / self.input_size)  # Not happy about the self call
        numbers = randn(500)
        scaled = numbers * std
        completed = list(weight_sets)

        for i, set in enumerate(completed):
            completed[i] = [np.random.choice(scaled, len(i)).round(2) for i in set]

        return completed

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

    # // ------------------------------------------------// # Process

    def determine_action(self, sight_line_data: np.array) -> int:

        hidden_layer = self.layer_calculation(
            layer_depth=0, inputs=sight_line_data, weights=self.weights_inputs_to_hidden
        )

        output_layer = self.layer_calculation(
            layer_depth=1,
            inputs=hidden_layer,
            weights=self.weights_hidden_to_output,
        )

        new_action = np.argmax(output_layer)

        return new_action

    def layer_calculation(
        self, layer_depth: int, inputs: np.array, weights: np.array
    ) -> np.array:
        layer_calculated = dot(inputs, weights)

        match layer_depth:
            case 0:
                layer_activated = [
                    np.maximum(0, x) for x in np.nditer(layer_calculated)
                ]
                layer_activated = np.array(layer_activated)
            case 1:
                e = np.exp(layer_calculated)
                layer_activated = e / e.sum()

        return layer_activated

    # // ------------------------------------------------// New Generation

    def new_generation(self):
        if len(self.Memory) >= self.New_Generation_Threshold:
            self.New_Generation_Parents = self.Memory
            self.clear_memory()
            Genertaion_logger.debug(
                f"Generation_Parents: {self.New_Generation_Parents}"
            )
            return True
        return False

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
        new_memory = self.MemoryInstance(
            episode=episode,
            reward=reward,
            t_alive=time_alive,
            H_W=self.weights_inputs_to_hidden,
            O_W=self.weights_hidden_to_output,
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
