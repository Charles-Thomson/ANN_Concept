import numpy as np
from numpy.random import randn
from math import sqrt
from itertools import chain
from SightData import check_sight_lines


class Brain:
    def __init__(self, init_data: tuple):
        self.nrow, self.ncol, self.agent_state, self.env = init_data
        self.agent_coords = self.to_coords(self.agent_state, self.ncol)
        self.build_network()

    # // ------------------------------------------------// Build

    def build_network(self):
        self.input_layer = np.array([[float(i)] for i in range(24)])
        self.hidden_layer = np.array([[float(0)] for h in range(9)])
        self.output_layer = np.array([[float(0)] for o in range(9)])

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

        # Testing needed
        self.weights_inputs_to_hidden = [
            np.random.choice(scaled, len(i)) for i in self.weights_inputs_to_hidden
        ]

        # Testing needed
        self.weights_hidden_to_output = [
            np.random.choice(scaled, len(i)) for i in self.weights_hidden_to_output
        ]

        # print(self.weights_inputs_to_hidden[0])
        self.update_weights()
        # print(self.weights_hidden_to_output)

    # // ------------------------------------------------// # Process

    def process(self, agent_state: int):
        self.input_layer = check_sight_lines(
            agent_state, self.nrow, self.ncol, self.env
        )
        self.calculate_hidden_layer()
        self.calculate_output_layer()
        new_action = self.determine_output()
        return new_action

    """
    Weight layer
    0 -> Input to hidden
    1 -> hidden to output
    Weighting 
    0 = improve time alive
    1 = improve score
    """

    def update_weights(self):
        # currnently increasing weight of open and goal tile and decreasing for obstical tile
        for i in range(9):
            for j in range(0, 24, 3):
                self.weights_inputs_to_hidden[i][j] = (
                    self.weights_inputs_to_hidden[i][j] + 0.1
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

        print(self.weights_inputs_to_hidden[1])

    # Outputlayer # Softmax
    def determine_output(self):
        vector = self.output_layer
        vector = list(chain(*vector))
        vector = np.array(vector)
        e = np.exp(vector)
        vector = e / e.sum()

        return np.argmax(vector)

    # Relu Activation function - Hidden Layer
    def activation_function(self, x) -> int:
        return np.maximum(0, x)

    def loss():
        pass

    def calculate_hidden_layer(self):
        inputs = self.input_layer
        weights = self.weights_inputs_to_hidden
        hidden = self.hidden_layer

        for index, h in enumerate(hidden):
            hidden[index] = self.activation_function(np.matmul(inputs, weights[index]))

        self.hidden_layer = hidden

    def calculate_output_layer(self):
        weights = self.weights_hidden_to_output
        hidden = self.hidden_layer
        output = self.output_layer

        hidden = list(chain(*hidden))
        hidden = np.array(hidden)

        for index, o in enumerate(output):
            output[index] = self.activation_function(np.matmul(hidden, weights[index]))

        self.output_layer = output

    # // ------------------------------------------------// Helper Functions

    # Save the variables of the brain to file
    def store_brain_data():
        pass

    # Convert the state -> (x,y) coords
    def to_coords(self, state: int, ncol: int) -> tuple:
        x = int(state / ncol)
        y = int(state % ncol)
        return (x, y)
