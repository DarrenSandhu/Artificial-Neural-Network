from tests.neural_network_with_hidden_layers import Neural_Network
import numpy as np
from tensorflow.keras.preprocessing import image
import os


def open_trained_nn(filename):
    data = np.load(filename)

    # Load the initaliser data
    input_nodes = data['input_nodes']
    output_nodes = data['output_nodes']
    hidden_layers = data['hidden_layers']
    print(f"Input nodes: {input_nodes}, Output nodes: {output_nodes}, Hidden layers: {hidden_layers}")

    
    # Create an instance of your Neural Network class
    # Assuming the number of layers and neurons are known
    nn = Neural_Network(input_nodes=input_nodes, hidden_layers=hidden_layers, output_nodes=output_nodes)
    
    # Load weights and biases
    weights = [data[f"layer_{i}_weights"] for i in range(len(hidden_layers) + 1)]
    biases = [data[f"layer_{i}_biases"] for i in range(len(hidden_layers) + 1)]

    # Load validation data
    nn.X_val = data['X_val']
    nn.y_val = data['y_val']
    print("Validation data shape: ", nn.X_val.shape)
    print("Validation labels shape: ", nn.y_val.shape)

    # Load other training parameters
    nn.learning_rate = data['learning_rate']
    nn.epochs = data['epochs']

    
    # Assign weights and biases to the neural network
    nn.weights = weights
    nn.biases = biases

    # Assign training data
    nn.X = data['X']
    nn.y = data['y']
    nn.learning_rate = data['learning_rate']
    nn.epochs = data['epochs']
    
    print(f"Model loaded from {filename}")
    print("\n\n\n\n")
    return nn

nn = open_trained_nn("trained_model.npz")
predictions = nn.predict(nn.X_val)
dog_count = 0
cat_count = 0
for i in range(len(predictions)):
    if predictions[i] > 0.5:
        cat_count += 1
    else:
        dog_count += 1
print(f"Number of cats: {cat_count}, Number of dogs: {dog_count}")

# Load a new image for prediction
# img = image.load_img('validation_data/dog/21.jpg', target_size=(64, 64))
# img_array = image.img_to_array(img)  # Convert image to array
# img_array = img_array / 255.0  # Rescale pixel values if needed
# img_array = img_array.reshape(1, -1)  # Flatten to match input shape

dog_true_count = 0.0
dog_false_count = 0.0

cat_true_count = 0.0
cat_false_count = 0.0

for i in range(len(os.listdir('validation_data/dog'))):
    try:
        img = image.load_img(f'validation_data/dog/{i}.jpg', target_size=(64, 64))
        img_array = image.img_to_array(img)  # Convert image to array
        img_array = img_array / 255.0  # Rescale pixel values if needed
        img_array = img_array.reshape(1, -1)  # Flatten to match input shape

        predictions = nn.predict(img_array)
        if predictions > 0.5:
            # print(f"Image {i} is a dog!")
            dog_true_count += 1
        else:
            # print(f"Image {i} is not a dog!")
            dog_false_count += 1

    except Exception as e:
        continue

for i in range(len(os.listdir('validation_data/cat'))):
    try:
        img = image.load_img(f'validation_data/cat/{i}.jpg', target_size=(64, 64))
        img_array = image.img_to_array(img)  # Convert image to array
        img_array = img_array / 255.0  # Rescale pixel values if needed
        img_array = img_array.reshape(1, -1)  # Flatten to match input shape

        predictions = nn.predict(img_array)
        if predictions < 0.5:
            # print(f"Image {i} is a cat!")
            cat_true_count += 1
        else:
            # print(f"Image {i} is not a cat!")
            cat_false_count += 1

    except Exception as e:
        continue

dog_accuracy = dog_true_count / (dog_true_count + dog_false_count)
print(f"Dog Accuracy: {dog_accuracy}")

cat_accuracy = cat_true_count / (cat_true_count + cat_false_count)
print(f"Cat Accuracy: {cat_accuracy}")
# Load a new image for prediction


# predictions = nn.predict(img_array2)
# if predictions > 0.5:
#     print("It's not a cat!")
# else:
#     print("It's a dog!")