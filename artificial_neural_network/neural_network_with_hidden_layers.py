import numpy as np
import math

class Neural_Network():
    def __init__(self, input_nodes, output_nodes, hidden_layers):
        self.input_nodes = input_nodes 
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []

        # Input to first hidden layer
        self.weights.append(np.random.randn(self.input_nodes, self.hidden_layers[0]))
        self.biases.append(np.full((1, self.hidden_layers[0]), 0.35))

        # Hidden layers
        for i in range(len(self.hidden_layers) - 1):
            self.weights.append(np.random.randn(self.hidden_layers[i], self.hidden_layers[i + 1]))
            self.biases.append(np.random.rand(1, self.hidden_layers[i + 1]))

        # Last hidden layer to output
        self.weights.append(np.random.randn(self.hidden_layers[-1], self.output_nodes))
        self.biases.append(np.random.rand(1, self.output_nodes))

        # Print the shapes of weights and biases
        # for i, (w, b) in enumerate(zip(self.weights, self.biases)):
        #     print(f"Layer {i} Weights Shape: {w.shape}")
        #     print(f"Layer {i} Biases Shape: {b.shape}")
    
    def sigmoid(self, z_value):
        return 1/(1+np.exp(-z_value))
        return np.maximum(0, z_value)
    
    def derivative(self, z_value):
        return z_value * (1 - z_value)
        return np.where(z_value > 0, 1, 0)
    
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
        # Forward pass through all hidden layers
        self.activations = [X]
        self.z_values = []  # Store the z-values for each layer
        
        for i in range(len(self.hidden_layers)):
            z = self.weighted_sum(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.sigmoid(z)
            self.activations.append(activation)

        # Output layer
        final_z = self.weighted_sum(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.final_output = self.sigmoid(final_z)
        
        return self.final_output
    
    def backpropagation(self, X, y, learning_rate):
        y = y.reshape(-1, 1)  # Ensure y is the correct shape
        
        # Calculate output layer error and delta
        output_error = self.calculate_error(y, self.final_output)
        output_derivative = self.derivative(self.final_output)
        output_delta = self.calculate_delta(output_error, output_derivative)
        # print("Output Delta Shape: ", output_delta.shape)
        # Backpropagate through each hidden layer
        deltas = [output_delta]
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            hidden_error = np.dot(deltas[-1], self.weights[i+1].T)
            hidden_derivative = self.derivative(self.activations[i+1])
            hidden_delta = self.calculate_delta(hidden_error, hidden_derivative)
            deltas.append(hidden_delta)

        # Reverse deltas to match the order of layers
        deltas = deltas[::-1]
        for i in range(len(self.hidden_layers) + 1):
            # Update the weights with the momentum
            self.weights[i] -= learning_rate * np.dot(self.activations[i].T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)




    def train(self, X, y, epochs, learning_rate, patience=2000):
        prev_loss = float('inf')
        no_improvement_count = 0
        
        for epoch in range(epochs):
            output = self.forward_pass(X)
            self.backpropagation(X, y, learning_rate)
            
            # Calculate loss
            loss = np.mean(np.square(y - output))
            
            # Early stopping if no improvement
            if loss < prev_loss:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch}, no improvement in loss.")
                break
            
            prev_loss = loss
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss:{loss}")
            
            print(f"Epoch {epoch}, Loss:{loss}")

    def predict(self, X):
        # Make predictions for input X
        output = self.forward_pass(X)  # Get output from forward pass
        # print("Output shape: ", output)
        return (output >= 0.5).astype(int)  # Convert output to binary prediction (dog or not_dog)

    
    def save(self, filename, X, y, learning_rate, epochs, hidden_layers, input_nodes, output_nodes, X_val, y_val):
        """Save the trained network's weights and biases to a file."""
        # Save weights and biases
        model_data = {}
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            model_data[f"layer_{i}_weights"] = weight
            model_data[f"layer_{i}_biases"] = bias
        np.savez(filename, 
                **model_data,
                X=X, 
                y=y, 
                learning_rate=learning_rate, 
                epochs=epochs,
                hidden_layers=hidden_layers,
                input_nodes=input_nodes,
                output_nodes=output_nodes,
                X_val=X_val,
                y_val=y_val,
                final_output=self.final_output
                )
        print(f"Model saved to {filename}")
    
    
    def run(self, neural_network):
        target_outputs = np.array([[0.01, 0.99, 0.3, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.25]])

        # Example input data (3 features)
        input_data = np.array([[0.05, 0.10, 0.15, 0.70, 0.85, 0.30, 0.35, 0.20, 0.65, 0.07]])

        # Learning Rate
        learning_rate = 0.5

        # Train
        neural_network.train(X=input_data, y=target_outputs, learning_rate=learning_rate, epochs=10000)

        # Test the trained model
        output = neural_network.forward_pass(input_data)
        print("Predictions after training:")
        print(output)

        
        

# Create a neural network instance with 3 input nodes, 2 output nodes, and 4 hidden nodes
# nn = Neural_Network(input_nodes=10, output_nodes=10, hidden_nodes=10)
# nn.run(nn)
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