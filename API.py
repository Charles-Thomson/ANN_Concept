from flask import Flask, request, jsonify

import RNN

app = Flask(__name__)

data = {}

TEST_MAP = [
    [0, 1, 1, 2, 3],
    [1, 1, 2, 2, 1],
    [1, 1, 2, 2, 1],
    [1, 1, 1, 2, 1],
    [1, 2, 1, 1, 1],
]


@app.route("/home")
def home() -> str:
    return "Home page working"


@app.route("/MAP", methods=["GET"])
def pass_MAP() -> dict:
    data = {}
    input_route = request.args["query"]  # Input route will take the MAP
    self = RNN.Q_LAEARNING_PROCESS(input_route)
    data["output"] = RNN.Q_LAEARNING_PROCESS.return_path_data(self)
    return data


# methods=["GET"]
@app.route("/data")
def return_data() -> dict:
    data = {}
    # input_route = request.args["query"]
    self = RNN.Q_LAEARNING_PROCESS(TEST_MAP)
    data["output"] = RNN.Q_LAEARNING_PROCESS.return_path_data(self)
    data["test"] = [
        1,
        1,
        1,
        1,
    ]
    # data["output"] = [1, 2, 3, 4, 5]
    return data


if __name__ == "__main__":
    app.run()
