import numpy as np
from numpy.random import randn
from math import sqrt, fsum
from numpy import dot

from SightData import check_sight_lines
import CustomLogging as CL


Brain_weights_logger = CL.GenerateLogger(__name__ + "weights", "loggingFileWeights.log")

Input_layer_logging = CL.GenerateLogger(
    __name__ + "inputLayer", "loggingInputLayer.log"
)

Hidden_layer_logging = CL.GenerateLogger(
    __name__ + "hiddenLayer", "logginghiddenLayer.log"
)

Output_layer_logging = CL.GenerateLogger(
    __name__ + "ouputLayer", "loggingOutputLayer.log"
)


class Brain:
    def __init__(self, init_data: tuple):
        self.nrow, self.ncol, self.agent_state, self.env = init_data
        self.agent_coords = self.to_coords(self.agent_state, self.ncol)

        self.build_network()

    # // ------------------------------------------------// Build

    def build_network(self):
        self.input_layer = np.array([float(i) for i in range(24)])
        self.hidden_layer = np.array([float(0) for h in range(9)])
        self.output_layer = np.array([float(0) for o in range(9)])

        self.weights_inputs_to_hidden = np.array(
            [
                [float(i) for i in range(len(self.input_layer))]
                for x in range(len(self.hidden_layer))
            ]
        )

        self.weights_hidden_to_output = np.array(
            [
                [float(i) for i in range(len(self.hidden_layer))]
                for x in range(len(self.output_layer))
            ]
        )

        self.set_start_weights()

    def set_start_weights(self):
        std = sqrt(2.0 / len(self.input_layer))
        numbers = randn(500)
        scaled = numbers * std

        self.weights_inputs_to_hidden = [
            np.random.choice(scaled, len(i)).round(2)
            for i in self.weights_inputs_to_hidden
        ]

        self.weights_hidden_to_output = [
            np.random.choice(scaled, len(i)).round(2)
            for i in self.weights_hidden_to_output
        ]
        Brain_weights_logger.debug(f"Current weights {self.weights_hidden_to_output}")

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
        layer_output = np.array([])

        for w in weights:
            assert len(inputs) == len(w), "input != weights"
            dot_result = dot(inputs, w)
            layer_output = np.append(layer_output, dot_result)

        return layer_output

    def loss():
        pass

    # // ------------------------------------------------//

    # // ------------------------------------------------// Helper Functions

    # Save the variables of the brain to file
    def store_brain_data():
        pass

    # Convert the state -> (x,y) coords
    def to_coords(self, state: int, ncol: int) -> tuple:
        x = int(state / ncol)
        y = int(state % ncol)
        return (x, y)

    def update_weights(self):

        # currnently increasing weight of open and goal tile and decreasing for obstical tile
        for i in range(9):
            for j in range(0, 24, 3):
                self.weights_inputs_to_hidden[i][j] = (
                    self.weights_inputs_to_hidden[i][j] + 1
                )

        for i in range(9):
            for j in range(1, 24, 3):
                self.weights_inputs_to_hidden[i][j] = (
                    self.weights_inputs_to_hidden[i][j] - 0.1
                )

        for i in range(9):
            for j in range(2, 24, 3):
                self.weights_inputs_to_hidden[i][j] = (
                    self.weights_inputs_to_hidden[i][j] + 0.2
                )

        # print(self.weights_inputs_to_hidden)
        Brain_weights_logger.debug(f"Current weights {self.weights_inputs_to_hidden}")
