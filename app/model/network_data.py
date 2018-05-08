import json
import pickle
import numpy as np
from ..train.train_tool import NeuralNetwork

class Neural:
    learning_rate = 0
    input_nodes = None
    output_nodes = None
    hidden_nodes = None
    wih = None
    whh_list = None
    who = None
    correct_rate = 0

    def __init__(self, learning_rate, input_nodes, output_nodes, hidden_nodes, wih, whh_list, who):
        self.learning_rate = learning_rate
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes
        self.wih = wih
        self.whh_list = whh_list
        self.who = who

    def __init__(self, neural_network):
        self.input_nodes = neural_network.input_node_count
        self.hidden_nodes = neural_network.hidden_node_count_arr
        self.output_nodes = neural_network.output_node_count
        self.learning_rate = neural_network.learning_rate
        self.wih = neural_network.wih
        self.whh_list = neural_network.whh_list
        self.who = neural_network.who

    def transform_to_network(self):
        neural_network = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes, self.learning_rate)
        neural_network.wih = self.wih
        neural_network.whh_list = self.whh_list
        neural_network.who = self.who
        return neural_network

    def save_to_local(self):
        save_path = 'neural.db'
        try:
            f = open(save_path, "wb")
            pickle.dump(self, f, True)
            f.close()
        except FileNotFoundError:
            save_path = None
        return save_path

    def transform_to_json(self):
        dic = {}
        dic["learning_rate"] = self.learning_rate
        dic["input_nodes"] = self.input_nodes
        dic["correct_rate"] = self.correct_rate
        dic["wih"] = self.wih.tolist()
        dic["whh_list"] = self.whh_list
        dic["who"] = self.who.tolist()
        return dic


    @staticmethod
    def generate_from_local(file_path):
        save_path = 'neural.db'
        model = None
        try:
            f = open(save_path, "rb")
            model = pickle.load(f)
            f.close()
        except EOFError:
            model = None
        return model


