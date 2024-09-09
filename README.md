Concept build for the Artificial Neural Network Project

The aim of the project is to implement and test back-end processing tools for neural networks.

Agents operate within a defined environment with the objective of gaining the highest "reward" possible. This is achieved by reaching goal nodes in the environment and exploring the environment. Agents that reach a set threshold of reward are selected to be "parents" for the next generation of agents.

New generations of agents are created using attributes from "parent" agents. The goal is to create new generations with positive attributes, i.e., goal-finding abilities. New agents' brains (ANNs) are given the chance to mutate weightings to introduce variance that could yield better results.

The application terminates once a generation is reached whose "children" are unable to meet a set goal threshold.

Goal thresholds are set based on the success of previous generations.

API:
- Provides an endpoint for receiving the environment map.

Agents:

- Defines functions relating to the generation and "running" of an agent instance in an environment.
- Sight data is collected and passed to the agent, based on eight sight lines.
- Functions on a step-reward basis: an action is taken, processed by the environment, and the result of the action is returned to the agent.

Brains:

- The ANN (Artificial Neural Network) for each agent.
- Each network is generated based on the required brain type, i.e., from parent attributes or from random values.
- The ANN handles action determination for the agent based on the provided "sight data."
- New generations' brains (ANNs) are determined based on defined functions and mutation properties.

MazeEnvironment:
- Defines functions relating to the processing of an agent instance in the maze.
- Returns data to the agent per "step" or action.

Logging:
- Extensive logging is provided for tracking agent progression by generation, down to each agent's move.

RNN Old:
- Unusde
- Defines other forms of ANN's in seperate use cases
