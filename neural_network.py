import numpy as np
import math

class Neural_Network():
    def __init__(self, input_nodes, output_nodes, hidden_nodes):
        self.input_nodes = input_nodes 
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes


        self.weights_input_to_hidden = np.random.randn(self.input_nodes, self.hidden_nodes)
        self.weights_hidden_to_output = np.random.randn(self.hidden_nodes, self.output_nodes)
        print("Weights For Hidden: \n",self.weights_input_to_hidden)
        print("Weights For Output: \n", self.weights_hidden_to_output)

        # self.weights_input_to_hidden = np.array([[0.15, 0.25], [0.20, 0.30]])
        # print("Weights For Hidden: \n",self.weights_input_to_hidden)
        # self.weights_hidden_to_output = np.array([[0.40, 0.50], [0.45, 0.55]])
        # print("Weights For Output: \n", self.weights_hidden_to_output)

        # self.bias_hidden = np.zeros((1, self.hidden_nodes))
        # self.bias_ouput = np.zeros((1, self.output_nodes))
        # print("Bias For Hidden: ",self.bias_hidden)
        # print("Bias For Output: ",self.bias_ouput)

        self.bias_hidden = np.full((1, self.hidden_nodes), 0.35)
        self.bias_output = np.full((1, self.output_nodes), 0.60)
        print("\nBias For Hidden: ",self.bias_hidden)
        print("Bias For Output: ",self.bias_output)
    
    def sigmoid(self, z_value):
        return 1/(1+np.exp(-z_value))
    
    def derivative(self, output):
        return output * (1 - output)
    
    def update_weight(self, weight, learning_rate, delta):
        weight = weight - (learning_rate * delta)
        return weight
    
    def weighted_sum(self, weight, value):
        return np.dot(weight, value)
    
    def target_output_diff(self, target_output, output):
        return target_output - output
    
    def squared_error(self, target_output, output):
        return np.power((target_output - output), 2)
    
    def calculate_error(self, target_output, final_output):
        return final_output - target_output
    
    def calculate_delta(self, error, derivative):
        return error * derivative
    
    def mean_squared_error(self, target_outputs, outputs):
        squared_errors = self.squared_error(target_outputs, outputs)
        self.mse = np.mean(squared_errors)
        return self.mse 
    
    def forward_pass(self, X):
        self.hidden_input = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.final_input = self.weighted_sum(self.hidden_output, self.weights_hidden_to_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)

        return self.final_output
    
    def backpropagation(self, X, y, learning_rate):
        self.output_error = self.calculate_error(y, self.final_output)
        self.output_derivative = self.derivative(self.final_output)
        self.output_delta = self.calculate_delta(self.output_error, self.output_derivative)
        self.output_change = (self.output_delta * self.hidden_output.T)
 
        self.hidden_error = np.dot(self.output_delta, self.weights_hidden_to_output.T)
        self.hidden_delta = self.calculate_delta(self.hidden_error, self.derivative(self.hidden_output))
        

        self.weights_hidden_to_output -= (self.output_change * learning_rate)
        self.weights_input_to_hidden -= (np.dot(X.T, self.hidden_delta) * learning_rate)

        self.bias_output -= learning_rate * np.sum(self.output_delta, axis=0, keepdims=True)
        self.bias_hidden -= learning_rate * np.sum(self.hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward_pass(X)
            self.backpropagation(X, y, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss:{loss}")

        
        

# Create a neural network instance with 3 input nodes, 2 output nodes, and 4 hidden nodes
nn = Neural_Network(input_nodes=10, output_nodes=10, hidden_nodes=10)
target_outputs = np.array([[0.01, 0.99, 0.3, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.25]])

# Example input data (3 features)
input_data = np.array([[0.05, 0.10, 0.15, 0.70, 0.85, 0.30, 0.35, 0.20, 0.65, 0.07]])

# Learning Rate
learning_rate = 0.5

# Train
nn.train(X=input_data, y=target_outputs, learning_rate=learning_rate, epochs=1000)


# Test the trained model
output = nn.forward_pass(input_data)
print("Predictions after training:")
print(output)
# print(nn.weights_hidden_to_output)
# print(nn.weights_input_to_hidden)
# Perform a forward pass
# output = nn.forward_pass(input_data)

# # Perform backpropagation
# learning_rate = 0.5
# nn.backpropagation(X=input_data, y=target_outputs, learning_rate=learning_rate)

# # Print the output from the neural network
# print("\n\nNetwork output:", output)

# print("Mean Squared Error:", nn.mse)


# def sigmoid(z_value):
#     return 1/(1+math.exp(z_value))

# test_input_nodes = 2
# test_hidden_nodes = 4
# test_output_nodes = 2

# test_weights_input_to_hidden = np.random.randn(test_input_nodes, test_hidden_nodes)

# test_bias = np.ones((1, test_hidden_nodes))


# sigmoid_test = sigmoid(-0.3775)

# test_target_output = 0.01
# test_output = 0.7514
# square_test = math.pow((test_target_output - test_output), 2)


# def squared_error(target_output, output):
#         return math.pow((target_output - output), 2)
    
# def mean_squared_error(target_outputs, outputs):
#     mse = 0
#     for index, target_output in enumerate(target_outputs):
#         error = squared_error(target_output, outputs[index])
#         mse += error
#     return mse * 0.5

# target_outputs = [0.01, 0.99]
# outputs = [0.7514, 0.7661]

# # Calculate mean squared error
# mse = mean_squared_error(target_outputs, outputs)
# print("Mean Squared Error:", mse)
# print("\n")

# print(f"Adding: {test_bias}  \n\n  {test_weights_input_to_hidden}")
# print("\n")
# test = test_weights_input_to_hidden + test_bias
# print(test)