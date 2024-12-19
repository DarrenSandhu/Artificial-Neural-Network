import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cnn_torch_multiple_fc_layers import Convolutional_Neural_Network_Multiple_FC_Layers  # Import your custom class
from architecture.conv_layer_list_torch import Convolutional_Layers_Torch  # Import your custom class
from architecture.fully_connected_layer_list_torch import Fully_Connected_Layers_Torch  # Import your custom class
from data_methods.get_cnn_training_data_torch import images, labels  # Import your custom data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Device: ",device)
torch.set_default_device(device)

# Create Model Parameters
input_nodes = images.shape[1] * images.shape[2] * images.shape[3] # Number of input nodes equals the number of pixels in an image
print("Images Shape: ",images.shape)
output_nodes = 2
target = labels
print("Target Shape: ",target.shape)
batch_size = 32
dropout_ratio = 0.3
lambda_l2 = 0.0003
strides = [1,1]
weight_init = 'kaiming-uniform'


#conv_layers = Convolutional_Layers_Torch(2, [[4,5], [8,3]], ['relu', 'relu'], images.shape, batch_size)
# conv_layers = Convolutional_Layers_Torch(2, [[16,5], [32,3]], ['relu', 'relu'], images.shape, batch_size)
#conv_layers = Convolutional_Layers_Torch(3, [[4,5], [8,3], [16,3]], ['relu', 'relu', 'relu'], images.shape, batch_size)
# conv_layers = Convolutional_Layers_Torch(4, [[16,5], [32,3], [64,3], [128,3]], ['relu', 'relu', 'relu', 'relu'], images.shape, batch_size)
conv_layers = Convolutional_Layers_Torch(2, [[8,5], [16,3]], ['relu', 'relu'], images.shape, batch_size, strides, weight_init)


num_features = conv_layers.conv_layers[-1].max_pool_images.view(batch_size, -1).shape[1]
print("Num Features: ",num_features)
# fully_connected_layers = Fully_Connected_Layers_Torch(5, output_sizes=[num_features, num_features*2, num_features, num_features // 2, output_nodes], activation_funcs=['relu', 'relu', 'relu', 'relu', 'sigmoid'])
fully_connected_layers = Fully_Connected_Layers_Torch(3, output_sizes=[num_features, num_features*2, output_nodes], activation_funcs=['relu', 'relu', 'sigmoid'])

cnn = Convolutional_Neural_Network_Multiple_FC_Layers(
    input_nodes, 
    output_nodes,
    conv_layers,
    fully_connected_layers,
    dropout_ratio=dropout_ratio,
    lambda_l2=lambda_l2
    )

# Shorten image list for faster training
# images = torch.cat((images[:300], images[1000:1300]), dim=0)
# target = torch.cat((target[:300], target[1000:1300]), dim=0)
cnn.train(images, target, 1000, 0.01, batch_size)
cnn.save("cnn_model.pth")