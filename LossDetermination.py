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


# Probably shouldent be a class
class LossDetermination:
    def __init__(self):
        # self.input_data = input_data
        # self.output_data = output_data
        self.dot_product()

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
    def expeced_results(self) -> np.array:

        return np.array([])

    # Giving the put at each node
    # Returns ap.array fo the putput of the follwoing layers in order. Top node first.
    # inputs: np.array, weights: np.array
    def dot_product(self):
        inputs = np.array([[1.1, 0.6, 1.1], [1.1, 0.1, 3.1], [0.1, 1.0, 2.1]])
        weights = np.array([[0.8, 0.1, 0.1], [0.5, 0.2, 0.3], [0.1, 0.5, 0.4]])

        hidden = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        assert len(inputs) == len(weights)

        # Dot approach <- looks like this approach and a rework to the network as a whole
        dot_results = dot(inputs, weights)

        outputs = np.array([fsum(r) for r in dot_results])

        logger.debug(f"Dot Result {dot_results} - Sum results: {outputs}")

        # Matmul approach
        for index, h in enumerate(hidden):
            hidden[index] = np.matmul(inputs, weights[index])

        hidden_sum = np.array([fsum(r) for r in hidden])

        logger.debug(
            f"Approach 2 - Hidden layer: {hidden} Hidden Layer sum: {hidden_sum}"
        )


if __name__ == "__main__":
    LossDetermination()
