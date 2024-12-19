from tests.neural_network_with_hidden_layers import Neural_Network
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import UnidentifiedImageError

# Print the list of available devices
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)

# Check if a GPU is available
if tf.config.list_physical_devices('GPU'):
    print("Training is happening on the GPU.")
else:
    print("Training is happening on the CPU.")

print("Experimental: ", tf.config.experimental.list_physical_devices('GPU'))


# Example for preprocessing
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "training_data",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary', # Binary for dog vs non-dog
)

val_data = datagen.flow_from_directory(
    "validation_data",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
)

print(train_data.class_indices)

input_nodes = 64 * 64 * 3  # for 64x64 grayscale images
output_nodes = 1  # Single output for binary classification (dog vs non-dog)
hidden_layers = [128, 64, 32, 16, 8, 4, 2]  # Number of neurons in each hidden layer

# Prepare training data (flatten images)
X_train, y_train = [], []
for i in range(len(train_data)):
    try:
        X_batch, y_batch = train_data.__next__()  # Use __next__ instead of .next()
        X_train.append(X_batch.reshape(X_batch.shape[0], -1))  # Flatten to 1D (for custom NN)
        y_train.append(y_batch)
    except Exception as e:
        print(f"Error processing batch {i}: {e}")
        continue

X_train = np.vstack(X_train)  # Stack all batches
print("X_train shape: ", X_train.shape)
y_train = np.hstack(y_train)  # Flatten the label list
print("y_train shape: ", y_train.shape)
print("y_train: ", y_train)

# Prepare validation data (flatten images)
X_val, y_val = [], []
for _ in range(len(val_data)):
    try:
        X_batch, y_batch = val_data.__next__()  # Use __next__ instead of .next()
        X_val.append(X_batch.reshape(X_batch.shape[0], -1))  # Flatten to 1D
        y_val.append(y_batch)
    except Exception as e:
        print(f"Error processing batch {i}: {e}")
        continue

X_val = np.vstack(X_val)
print("X_val shape: ", X_val.shape)
y_val = np.hstack(y_val)
print("y_val shape: ", y_val.shape)
print()

nn = Neural_Network(input_nodes=input_nodes, output_nodes=output_nodes, hidden_layers=hidden_layers)

# Training
epochs = 30
learning_rate = 0.01

# Set the device to GPU (if available)
with tf.device('/GPU:0'):
    # Your training code here
    nn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)

nn.save("trained_model.npz", 
        X=X_train, 
        y=y_train, 
        learning_rate=learning_rate, 
        epochs=epochs, 
        hidden_layers=hidden_layers, 
        input_nodes=input_nodes, 
        output_nodes=output_nodes,
        X_val=X_val,
        y_val=y_val
    )
        