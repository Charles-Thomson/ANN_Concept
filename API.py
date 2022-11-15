"""
API used to pass data to and from the back end NN and the Flutter application

Variables 

app -> Declare the app as a Flask app

formatinputroute()
Format the passed information from the Flutter UI to be passed to the NN
-> The data is passed as a string, this needs to be parsed into list<int>
-> Called by pass_MAP

pass_MAP()
Route used by Flutter UI to pass 'map' data.
-> Takes data in using the 'query' tag.
-> Formats the data usin formatinputroute().
-> Passes data to the NN
-> Takes output of the NN, passes back to the Flutter UI using tag 'output'

"""

from flask import Flask, request, jsonify

import RNN

app = Flask(__name__)

# data = {}


def formatinputroute(input_route: str) -> list:

    # must format to the correct size - currently in one flat list
    input_route = input_route.lstrip("[").rstrip("]")
    route = input_route.split(",")
    route = [int(x) for x in route if x != " "]

    return route


@app.route("/MAP", methods=["GET"])
def pass_MAP() -> dict:
    data = {}
    input_route = request.args["query"]  # Input route will take the MAP

    # Commented out for app testing
    formatted_input_route = formatinputroute(input_route)
    self = RNN.Q_LAEARNING_PROCESS(formatted_input_route)
    data["output"] = RNN.Q_LAEARNING_PROCESS.return_path_data(self)

    # data["output"] = [1, 2, 3, 4] #
    return data


if __name__ == "__main__":
    app.run()
