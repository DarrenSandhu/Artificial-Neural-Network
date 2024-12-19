import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vgg import VGG  # Import your custom class
from architecture.vgg_layer_list_torch import VGG_Layers_Torch
from architecture.fully_connected_layer_list_torch import Fully_Connected_Layers_Torch
from data_methods.get_cnn_training_data_torch import images, labels

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Device: ",device)
torch.set_default_device(device)

# Create Model Parameters
input_nodes = images.shape[1] * images.shape[2] * images.shape[3] # Number of input nodes equals the number of pixels in an image
print("Images Shape: ",images.shape)
output_nodes = 2
target = labels
print("Target Shape: ",target.shape)
batch_size = 16
dropout_ratio = 0.3 
lambda_l2 = 0.0003
strides = [1, 1, 1]
conv_blocks = [3,3,3]

conv_layers = VGG_Layers_Torch(3, [[64,3], [64,3], [128,3]], ['relu', 'relu', 'relu'], images.shape, batch_size, conv_block=conv_blocks, strides=strides)

num_features = conv_layers.conv_layers[-1].max_pool_images.view(batch_size, -1).shape[1]
print("Num Features: ",num_features)
fully_connected_layers = Fully_Connected_Layers_Torch(4, output_sizes=[num_features, num_features*2, num_features, output_nodes], activation_funcs=['relu', 'relu', 'relu', 'sigmoid'])

# Create the model
cnn = VGG(input_nodes, 
            output_nodes, 
            conv_layers,
            fully_connected_layers,
            dropout_ratio=dropout_ratio,
            lambda_l2=lambda_l2)

# Shorten image list for faster training
images = torch.cat((images[:100], images[1000:1100]), dim=0)
target = torch.cat((target[:100], target[1000:1100]), dim=0)
cnn.train(images, target, 1000, 0.001, batch_size)
cnn.save("vgg_model.pth")