import numpy as np
import math

class Neural_Network():
    def __init__(self, input_nodes, output_nodes, hidden_nodes, weights_input_to_hidden=None, weights_hidden_to_output=None):
        self.input_nodes = input_nodes 
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes


        if weights_input_to_hidden is not None:
            self.weights_input_to_hidden = weights_input_to_hidden
        else:
            self.weights_input_to_hidden = np.random.randn(self.input_nodes, self.hidden_nodes)
        print("Hidden Weights Shapse:",self.weights_input_to_hidden.shape)
        if weights_hidden_to_output is not None:
            self.weights_hidden_to_output = weights_hidden_to_output
        else:
            self.weights_hidden_to_output = np.random.randn(self.hidden_nodes, self.output_nodes)
        print("Output Weights Shape:",self.weights_hidden_to_output.shape)
        print("\n")
        # print("Weights For Hidden: \n",self.weights_input_to_hidden)
        # print("Weights For Output: \n", self.weights_hidden_to_output)

        self.bias_hidden = np.full((1, self.hidden_nodes), 0.35)
        print("Hidden Bias Shape: ",self.bias_hidden.shape) 
        print("Hidden Bias: ",self.bias_hidden)
        print("\n")
        self.bias_output = np.full((1, self.output_nodes), 0.60)
        print("Output Bias Shape: ",self.bias_output.shape)
        print("Output Bias: ",self.bias_output)
        print("\n")
        # print("\nBias For Hidden: ",self.bias_hidden)
        # print("Bias For Output: ",self.bias_output)
    
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
    
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def forward_pass(self, X):
        self.hidden_input = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        # print("Hidden Input Shape: ", self.hidden_input.shape)
        # self.hidden_output = # Replace sigmoid with ReLU in your forward pass
        self.hidden_output = self.sigmoid(self.hidden_input)
        # print("Hidden Output Shape: ", self.hidden_output.shape)

        self.final_input = self.weighted_sum(self.hidden_output, self.weights_hidden_to_output) + self.bias_output
        # print("Final Input Shape: ", self.final_input.shape)
        self.final_output = self.sigmoid(self.final_input)
        # print("Final Output Shape: ", self.final_output.shape)

        return self.final_output
    
    def backpropagation(self, X, y, learning_rate):
        # y = y.reshape(-1, 1)
        print("Target Output: ", y)
        print("Final Output: ", self.final_output)
        self.output_error = self.calculate_error(y, self.final_output)
        print("Output Error: ", self.output_error)
        self.output_derivative = self.derivative(self.final_output)
        self.output_delta = self.calculate_delta(self.output_error, self.output_derivative)
        print("Output Delta: ", self.output_delta)

        self.output_change = np.dot(self.hidden_output.T, self.output_delta)
        # print("Output Change Shape: ", self.output_change.shape)
        
        
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
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss:{loss}")
    
    def save(self, filename):
        """Save the trained network's weights and biases to a file."""
        # Save weights and biases
        np.savez(filename, 
                 weights_input_to_hidden=self.weights_input_to_hidden,
                 weights_hidden_to_output=self.weights_hidden_to_output,
                 bias_hidden=self.bias_hidden,
                 bias_output=self.bias_output)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load weights and biases from a file."""
        # Load weights and biases
        data = np.load(filename)
        self.weights_input_to_hidden = data['weights_input_to_hidden']
        self.weights_hidden_to_output = data['weights_hidden_to_output']
        self.bias_hidden = data['bias_hidden']
        self.bias_output = data['bias_output']
        print(f"Model loaded from {filename}")
    
    # def run(self, neural_network):
    #     target_outputs = np.array([[0.01, 0.99, 0.3, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.25]])

    #     # Example input data (3 features)
    #     input_data = np.array([[0.05, 0.10, 0.15, 0.70, 0.85, 0.30, 0.35, 0.20, 0.65, 0.07]])

    #     # Learning Rate
    #     learning_rate = 0.5

    #     # Train
    #     neural_network.train(X=input_data, y=target_outputs, learning_rate=learning_rate, epochs=10000)

    #     # Test the trained model
    #     output = neural_network.forward_pass(input_data)
    #     print("Predictions after training:")
    #     print(output)

        
# weights_input_to_hidden = np.array([[0.15, 0.25],
#                                     [0.20, 0.30],
#                                     ])

# weights_hidden_to_output = np.array([[0.40, 0.50],
#                                     [0.45, 0.55]])
input_data = np.array([[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]])
target_outputs = np.array([[0.01, 0.99]])

# Create a neural network instance with 3 input nodes, 2 output nodes, and 4 hidden nodes
nn = Neural_Network(input_nodes=input_data.shape[1], output_nodes=target_outputs.shape[1], hidden_nodes=10)
# print(nn.weights_hidden_to_output)
# print(nn.weights_input_to_hidden)
# Perform a forward pass
learning_rate = 0.9
epochs = 10000
output = nn.forward_pass(input_data)

nn.backpropagation(input_data, target_outputs, learning_rate)

# nn.train(input_data, target_outputs, epochs, learning_rate)

# # Print the output from the neural network
# print("\n\nNetwork output:", nn.forward_pass(input_data))

# # Predictions after training
# test_input = np.array([[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]])
# print("Predictions after training:")
# print(nn.forward_pass(test_input))




