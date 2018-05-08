from . import api
from flask import request, json
from ..model.base_model import Response
from ..model.network_data import Neural
from ..train.train_tool import NeuralNetwork, TrainManager
from flask import render_template
import numpy as np

@api.route('/network/query_number', methods=['POST', 'GET'])
def query_number():
    # neural_id = request.form['neural_id']
    query_data = request.form.get("query_data")
    query_number = json.loads(query_data)

    neural = Neural.generate_from_local(None)
    network = neural.transform_to_network()
    result = network.query(np.asfarray(query_number))
    response = Response()
    response.error_code = 0
    response.log_msg = "query success"
    # response.data = {'number': result.tolist(), 'msg': query_data}
    response.data = result.tolist()
    return response.generate_to_json()


@api.route('/network/neural_list', methods=['POST', 'GET'])
def neural_list():
    neural = Neural.generate_from_local(None)
    response = Response()
    response.data = {'list': [123]}
    if neural is None:
        response.error_code = 404
    else:
        response.error_code = 0
        response.data = {'list': [neural.transform_to_json()]}
    response.log_msg = "neural list"
    return response.generate_to_json()


@api.route('/network/train', methods=['GET', 'POST'])
def train_neural_network():
    neural = TrainManager(learning_rate=0.3, input_nodes=784, hidden_nodes=[100], output_nodes=10)
    neural.train_neural_network(True, 1)
    network = neural.neural_network
    network_block = Neural(network)
    result = neural.test_nerual_network()
    network_block.correct_rate = result
    network_block.save_to_local()
    return "<h1>The correct rate: %f</h1><div>%f</div>" % (result, network_block.learning_rate)


@api.route('/network/homepage', methods=['POST', 'GET'])
def home_page():
    return render_template("homepage.html")
