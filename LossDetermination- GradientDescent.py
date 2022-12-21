import numpy as np
from numpy import dot, append
import logging
import CustomLogging as CL
from math import fsum


# Basic logging config
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(
    level=logging.NOTSET,
)

logger = CL.GenerateLogger(__name__, "LossLogging.log")

Derivs_logger = CL.GenerateLogger(__name__ + "Derivs", "LossDerivsLogging.log")


# Probably shouldent be a class
class LossDetermination:
    def __init__(self):
        pass
        # self.input_data = input_data
        # self.output_data = output_data

        # Inputs as list
        # Output for each output_node as list

        # weights for each output
        # broken down into weights for each output node

        # check out dot_product

        # Dot-product takes the inputs and current weights
        # Gives the value at the given node, sum(nth_input * nth_weight)
        # This is the exp valu eat that node

        # Need expected result
        # ////////////////////////////////////////////////

        # Given the inputs give an expected value at each node
        # Calculate the value at each node using dot_matrixe ect

        #

    # Need the current input data, break down into the each direction
    # Expected result of each node on the direction we would expect the ...
    # ... agent to take
    # Priority is towards the goal, else towards open/ away from obstical
    def expected_results(self) -> np.array:

        return np.array([])

    # Giving the put at each node
    # Returns ap.array fo the putput of the follwoing layers in order. Top node first.
    # inputs: np.array, weights: np.array

    def predicted_result(
        self,
        input_data: np.array,
        weights_input_to_hidden: np.array,
        weights_hidden_to_output: np.array,
    ):

        # hard coding this for the move 8
        expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        hidden_layer_result_dot = dot(input_data, weights_input_to_hidden)
        hidden_layer_result = self.hidden_layer_activation(hidden_layer_result_dot)

        # output layer before activation
        predicted_output = dot(hidden_layer_result, weights_hidden_to_output)

        logger.debug(f"Sum results at hidden: {hidden_layer_result}")

        # Softmax acivation on the output layer - this is the predicted output !!
        softmax_output = self.output_layer_activation(
            predicted_output
        )  # gives us the take it all value

        # print(softmax_output)

        # Gives a winner take it all matrix
        # winner_matrix = np.zeros(len(predicted_output))
        # winner_matrix[softmax_output] = 1

        # update weight values of output layer

        # Phase 1
        # Calculating the mean Squared Error
        error_out = np.array([], dtype=object)
        error_out = (1 / 2) * (np.power((softmax_output, expected_output), 2))
        # print(error_out.sum())

        # output_op - target_op
        derror_douto = softmax_output - expected_output
        # print(derror_douto)
        # input for outputlayer - hidden layer resut + activation applied
        douto_dino = self.output_layer_activation(predicted_output)
        # print(douto_dino)
        # result of hidden layer after activation
        dino_dwo = self.hidden_layer_activation(hidden_layer_result_dot)

        holder = derror_douto * douto_dino
        derror_dwo = np.dot(dino_dwo, holder)

        Derivs_logger.debug(f"End of phase one {derror_dwo} ")

        # Phase 2
        # Derivitives for phase 2

        derror_dino = derror_douto * douto_dino
        print(derror_dino)

        dino_douth = np.array(weights_hidden_to_output).T

        # print(dino_douth)

        derror_douth = np.dot(derror_dino, dino_douth)
        print(derror_douth)

        douth_dinh = self.hidden_layer_activation(hidden_layer_result_dot)
        # print(douth_dinh)

        dinh_dwh = np.array(input_data)[np.newaxis].reshape(1, 24)

        holder = douth_dinh * derror_douth  # this needs to give 24 x 9 matrix

        # Issue relating to the size of the matrixs being used in the final
        # Need to find out why the matrixs arn't lining up or if it's due to using the wrong matrixs ???

        derror_wh = np.dot(dinh_dwh, holder)

        learning_rate = 0.05
        weights_input_to_hidden -= learning_rate * derror_wh
        weights_hidden_to_output -= learning_rate * derror_dwo

        Derivs_logger.debug(
            f"Weight updates - Hidden {weights_input_to_hidden}  - Output {weights_hidden_to_output} "
        )

    def hidden_layer_activation(self, layer):

        # Applying RElu to each element in hidden layer
        layer = [np.maximum(0, x) for x in np.nditer(layer)]
        return np.array(layer)

    def output_layer_activation(self, layer):
        # applying sofmax to output layer to return final value
        e = np.exp(layer)
        layer = e / e.sum()
        return layer


if __name__ == "__main__":
    LossDetermination()
