import numpy as np
from scipy.special import expit


# neural network class definition
class NeuralNetwork:

    def __init__(self, input_node_count, hidden_node_count_arr, output_node_count, learning_rate):
        self.input_node_count = input_node_count
        self.hidden_node_count_arr = hidden_node_count_arr
        self.output_node_count = output_node_count
        self.learning_rate = learning_rate

        self.wih = np.random.normal(0.0, pow(hidden_node_count_arr[0], -0.5), (hidden_node_count_arr[0], input_node_count))
        self.whh_list = []
        for i in range(0, len(hidden_node_count_arr)):
            if i == 0:
                continue
            self.whh_list.append(np.random.normal(0.0, pow(hidden_node_count_arr[i], -0.5), (hidden_node_count_arr[i], hidden_node_count_arr[i-1])))

        self.who = np.random.normal(0.0, pow(output_node_count, -0.5), (output_node_count, hidden_node_count_arr[-1]))
        self.activation_function = lambda x: expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        middle_hidden_outputs_arr = list()
        middle_hidden_outputs_arr.append(hidden_outputs)

        for i in range(0, len(self.whh_list)):
            hidden_inputs = np.dot(self.whh_list[i], hidden_outputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            middle_hidden_outputs_arr.append(hidden_outputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        for i in range(len(self.whh_list)-1, -1, -1):
            self.whh_list[i] += self.learning_rate * np.dot((hidden_errors * middle_hidden_outputs_arr[i+1] * (1.0 - middle_hidden_outputs_arr[i+1])), np.transpose(middle_hidden_outputs_arr[i]))
            hidden_errors = np.dot(self.whh_list[i].T, hidden_errors)
            pass

        self.wih += self.learning_rate * np.dot((hidden_errors * middle_hidden_outputs_arr[0] * (1.0 - middle_hidden_outputs_arr[0])), np.transpose(inputs))
        pass

    def query(self, inputs):
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        for i in range(0, len(self.whh_list)):
            hidden_inputs = np.dot(self.whh_list[i], hidden_outputs)
            hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


class TrainManager:
    train_100_file = "/Users/caishilin/Desktop/neuralNetwork/app/static/mnist_train_100.csv"
    train_big_file = "/Users/caishilin/Desktop/neuralNetwork/app/static/mnist_train.csv"
    neural_network = None

    def __init__(self, learning_rate, input_nodes, hidden_nodes, output_nodes):
        self.learning_rate = learning_rate
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

    def train_neural_network(self, big_data, time):
        n = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes, self.learning_rate)
        self.neural_network = n
        data_file = open(self.train_big_file if big_data else self.train_100_file, 'r')
        data_list = data_file.readlines()
        data_file.close()
        for i in range(0, time):
            for record in data_list:
                all_values = record.split(",")
                scaled_input = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
                targets = np.zeros(self.output_nodes) + 0.01
                targets[int(record[0])] = 0.99
                n.train(scaled_input, targets)

    def test_nerual_network(self):
        if not self.neural_network:
            return -1
        test_data_file = open("/Users/caishilin/Desktop/neuralNetwork/app/static/mnist_test.csv", 'r')
        test_data_list = test_data_file.readlines()
        test_data_file.close()
        total = float(len(test_data_list))
        correct_count = 0.0
        for record in test_data_list:
            all_values = record.split(",")
            scaled_input = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
            outputs = self.neural_network.query(scaled_input)
            max_value = max(outputs)
            index = list(outputs).index(max_value)
            if index == int(record[0]):
                correct_count += 1
        return correct_count/total



# input_nodes = 784
# hidden_nodes = [100]
# output_nodes = 10
#
# learning_rate = 0.3
#
# manager = TrainManager(learning_rate, input_nodes, hidden_nodes, output_nodes)
# manager.train_neural_network(True,1)
# result = manager.test_nerual_network()
# print(result)
