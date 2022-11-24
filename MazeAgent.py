import numpy as np
import MazeENV as env
from itertools import chain
from math import sqrt
from numpy.random import randn

"""
For the NN
28 input = 8 directions * 3 (empty, wall, hunter )

# Need to define a policay for eac agent, the loss will then
        # Adjust the weights based on the difference in out come from the achivment
        # of the policay - so not needing to use a test set of data ?

        # Cross entropy as loss function baed on the policy

        # Rulu used on hidden layer

        # Testing Softmax activation function - used on ouput layer

"""


class MazeAgent:
    def __init__(self, agent_state):
        self.EPISODES = 4

        self.agent_state = agent_state
        self.env = env.MazeEnv(12)  # passing a hard coded start of 12
        self.nrow = self.env.nrow
        self.ncol = self.env.ncol

        input_data = self.collect_sightline_data()
        self.input_data = np.array(input_data)
        self.hidden_layer = np.array([])
        self.output_layer = np.array([])

        self.weights_inputs_to_hidden = np.array([])
        self.weights_hidden_to_output = np.array([])

        self.main()

    def main(self):
        self.build_test_network()
        # self.calculate_hidden_layer()
        # self.calculate_output_layer()
        # result = self.determine_output()
        # move = np.argmax(result)
        # print(move)  # the final resulting move
        self.run_network()

    def run_network(self):

        for e in range(self.EPISODES):
            self.input_data = self.collect_sightline_data()
            print(self.input_data)
            self.calculate_hidden_layer()
            self.calculate_output_layer()
            action_data = self.determine_output()

            action = np.argmax(action_data)

            n_state, r, i, t = self.env.step(action)
            if t is True:
                print("Termination")
                continue
            print(f"state: {self.agent_state} Action {action} -> new_state: {n_state} ")
            self.agent_state = n_state

    def loss_function(self):
        pass

    def determine_output(self):
        vector = self.output_layer
        vector = list(chain(*vector))
        vector = np.array(vector)

        e = np.exp(vector)
        return e / e.sum()

    def calculate_hidden_layer(self):
        inputs = self.input_data
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

    def build_test_network(self):
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

        self.init_weights(self.weights_inputs_to_hidden)
        self.init_weights(self.weights_hidden_to_output)

    # "he" initialization - all weights
    def init_weights(self, weights: np.array):
        std = sqrt(2.0 / len(self.input_data))
        numbers = randn(500)
        scaled = numbers * std

        for i, x in enumerate(weights):
            weights[i] = np.random.choice(scaled, len(x))

    # Relu Activation function - Hidden Layer
    def activation_function(self, x) -> int:
        return np.maximum(0, x)

    # Take in data from surrounding environemnt
    # 8 Directions to check
    # 3 perameters per direction
    # Each direction Total values for* (empty, wall, hunter )
    def collect_sightline_data(self):
        state = self.agent_state
        x, y = self.env.to_coords_call(state)

        visable_env_data = [[0 for i in range(3)] for x in range(8)]

        for i, data in enumerate(visable_env_data):
            match i:
                case 0:  # Up + Left
                    locations = [(x - 1, y - 1), (x - 2, y - 2)]
                    data = self.check_sightline(locations)
                    visable_env_data[i] = data

                case 1:  # Up
                    locations = [(x - 1, y), (x - 2, y)]
                    data = self.check_sightline(locations)
                    visable_env_data[i] = data

                case 2:  # Up + Right
                    locations = [(x - 1, y + 1), (x - 2, y + 2)]
                    data = self.check_sightline(locations)
                    visable_env_data[i] = data

                case 3:  # Left
                    locations = [(x, y - 1), (x, y - 2)]
                    data = self.check_sightline(locations)
                    visable_env_data[i] = data

                case 4:  # No move
                    locations = [(x, y)]
                    data = self.check_sightline(locations)
                    visable_env_data[i] = data

                case 5:  # Right
                    locations = [(x, y + 1), (x, y + 2)]
                    data = self.check_sightline(locations)
                    visable_env_data[i] = data

                case 6:  # Down + Left
                    locations = [(x + 1, y - 1), (x + 2, y - 2)]
                    data = self.check_sightline(locations)
                    visable_env_data[i] = data

                case 7:  # Down
                    locations = [(x + 1, y), (x + 2, y)]
                    data = self.check_sightline(locations)
                    visable_env_data[i] = data

                case 8:  # Down + Right
                    locations = [(x + 1, y + 1), (x + 2, y + 2)]
                    data = self.check_sightline(locations)
                    visable_env_data[i] = data

        visable_env_data = list(chain(*visable_env_data))
        return visable_env_data

    # Pass in a list of values to check
    def check_sightline(self, locations: list):
        sightline_data = [0.0, 0.0, 0.0]

        for distance, loc in enumerate(locations):
            x, y = loc

            if not 0 <= x < self.nrow or not 0 <= y < self.ncol:
                continue

            value = self.env.get_location_value_call((x, y))

            match value:
                case 1:  # Open
                    if distance == 0:
                        sightline_data[0] += 0.9
                    if distance == 1:
                        sightline_data[0] += 0.6

                case 2:
                    if distance == 0:
                        sightline_data[1] += 0.9
                    if distance == 1:
                        sightline_data[1] += 0.6

                case 3:
                    if distance == 0:
                        sightline_data[2] += 0.9
                    if distance == 1:
                        sightline_data[2] += 0.6

        return sightline_data


MazeAgent(12)
