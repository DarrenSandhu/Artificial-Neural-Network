import numpy as np
import cv2 as cv2  # OpenCV for image handling
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

class Neural_Network():
    def __init__(self, input_nodes, output_nodes, hidden_nodes_list):
        self.input_nodes = input_nodes 
        self.output_nodes = output_nodes
        self.hidden_nodes_list = hidden_nodes_list

        # Initialize weights and biases for multiple hidden layers
        self.weights_input_to_hidden = [np.random.randn(self.input_nodes, self.hidden_nodes_list[0])]
        self.bias_hidden = [np.full((1, self.hidden_nodes_list[0]), 0.35)]

        # Create weights and biases for all hidden layers
        for i in range(1, len(hidden_nodes_list)):
            self.weights_input_to_hidden.append(np.random.randn(self.hidden_nodes_list[i-1], self.hidden_nodes_list[i]))
            self.bias_hidden.append(np.full((1, self.hidden_nodes_list[i]), 0.35))
        
        # Final output layer weights
        self.weights_hidden_to_output = np.random.randn(self.hidden_nodes_list[-1], self.output_nodes)
        self.bias_output = np.full((1, self.output_nodes), 0.60)
    
    def sigmoid(self, z_value):
        return 1 / (1 + np.exp(-z_value))
    
    def derivative(self, output):
        return output * (1 - output)
    
    def forward_pass(self, X):
        self.hidden_outputs = []
        self.hidden_inputs = []

        # Pass through all hidden layers
        for i in range(len(self.hidden_nodes_list)):
            if i == 0:
                self.hidden_inputs.append(np.dot(X, self.weights_input_to_hidden[i]) + self.bias_hidden[i])
            else:
                self.hidden_inputs.append(np.dot(self.hidden_outputs[i-1], self.weights_input_to_hidden[i]) + self.bias_hidden[i])
            
            self.hidden_outputs.append(self.sigmoid(self.hidden_inputs[i]))
        
        # Output layer
        self.final_input = np.dot(self.hidden_outputs[-1], self.weights_hidden_to_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)

        return self.final_output
    
    def backpropagation(self, X, y, learning_rate):
        output_error = y - self.final_output
        output_delta = output_error * self.derivative(self.final_output)

        # Backpropagate the error through all hidden layers
        hidden_deltas = []
        hidden_errors = output_delta
        for i in range(len(self.hidden_nodes_list)-1, -1, -1):
            hidden_error = np.dot(hidden_errors, self.weights_hidden_to_output.T) if i == len(self.hidden_nodes_list)-1 else np.dot(hidden_errors, self.weights_input_to_hidden[i+1].T)
            hidden_delta = hidden_error * self.derivative(self.hidden_outputs[i])
            hidden_deltas.append(hidden_delta)
            hidden_errors = hidden_delta

        hidden_deltas.reverse()

        # Update weights and biases for hidden layers
        for i in range(len(self.hidden_nodes_list)-1, -1, -1):
            if i == 0:
                self.weights_input_to_hidden[i] += np.dot(X.T, hidden_deltas[i]) * learning_rate
            else:
                self.weights_input_to_hidden[i] += np.dot(self.hidden_outputs[i-1].T, hidden_deltas[i]) * learning_rate
            self.bias_hidden[i] += np.sum(hidden_deltas[i], axis=0, keepdims=True) * learning_rate

        # Update weights and bias for the output layer
        self.weights_hidden_to_output += np.dot(self.hidden_outputs[-1].T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward_pass(X)
            self.backpropagation(X, y, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")
                
    def predict(self, X):
        return self.forward_pass(X)

# Helper function for loading and preprocessing the image dataset
def load_data(image_dir, image_size=(64, 64)):
    images = []
    labels = []

    for label in ['cat', 'dog']:
        label_dir = os.path.join(image_dir, label)
        print(f"Loading images from {label_dir}")
        for filename in os.listdir(label_dir):
            # Skip non-image files like Thumbs.db
            if filename.lower() == 'thumbs.db':
                continue
            img_path = os.path.join(label_dir, filename)
            assert os.path.exists(img_path), f"File not found: {img_path}"
            try:
                # Use Pillow to open the image
                if img_path.endswith('6.jpg'):
                    continue
                img = Image.open(img_path)


                # Resize the image using Pillow
                img = img.resize(image_size)
                
                # Convert image to NumPy array and normalize
                img = img.convert('RGB')
                img_array = np.array(img) / 255.0  # Normalize the image (between 0 and 1)

                # Only add valid image arrays with the expected shape (64, 64, 3)
                if img_array.shape == (image_size[0], image_size[1], 3):  # Check for 3 channels (RGB)
                    images.append(img_array)
                    labels.append(label)
                else:
                    print(f"Skipping invalid image {img_path} with shape {img_array.shape}")
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue  # Skip corrupt images
    
    # Convert the list into a numpy array
    images = np.array(images)
    labels = np.array(labels)
    
    # Encode the labels as one-hot vectors
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = np.expand_dims(labels, axis=1)
    labels = np.where(labels == 0, [1, 0], [0, 1])  # One-hot encoding

    return images, labels

# Load data
image_dir = os.path.join(BASE_DIR, 'training_data')
X, y = load_data(image_dir)
print(f"X shape: {X}, y shape: {y}")

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the neural network
input_nodes = X_train.shape[1]  # Number of input nodes equals the number of pixels in an image
output_nodes = 2  # Two classes: cat and dog
hidden_nodes_list = [128, 64, 32]  # Example with 3 hidden layers of sizes 128, 64, 32

nn = Neural_Network(input_nodes, output_nodes, hidden_nodes_list)
nn.train(X_train, y_train, epochs=10000, learning_rate=0.1)

# Evaluate the network's accuracy
predictions = nn.predict(X_test)
predictions = np.round(predictions)  # Round to nearest integer (either 0 or 1)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy * 100}%')
