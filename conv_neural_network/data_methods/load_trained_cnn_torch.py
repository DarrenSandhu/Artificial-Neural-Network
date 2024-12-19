import torch
import os
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.cnn_torch import Convolutional_Neural_Network  # Import your custom class
from torch.cnn_torch_multiple_fc_layers import Convolutional_Neural_Network_Multiple_FC_Layers  # Import your custom class
from architecture.conv_layer_list_torch import Convolutional_Layers_Torch  # Import your custom class
from architecture.conv_layer_torch import Convolution_Layer_Torch  # Import your custom class
from architecture.fully_connected_layer_list_torch import Fully_Connected_Layers_Torch  # Import your custom class
from architecture.fully_connected_layer_torch import Fully_Connected_Layer  # Import your custom class

torch.serialization.add_safe_globals([Convolutional_Layers_Torch])
torch.serialization.add_safe_globals([Convolution_Layer_Torch])
torch.serialization.add_safe_globals([Fully_Connected_Layers_Torch])
torch.serialization.add_safe_globals([Fully_Connected_Layer])

BASE_DIR = os.path.dirname(os.path.abspath(__name__))

# Load the trained model
def open_trained_torch_cnn(filename):
    data = torch.load(filename, weights_only=True, map_location='cpu')

    # Model parameters
    input_nodes = data['input_nodes']
    # print(f"Input nodes: {input_nodes}")
    output_nodes = data['output_nodes']
    # print(f"Output nodes: {output_nodes}")
    fully_connected_weights = data['fully_connected_weights']
    # print(f"Fully connected weights Shape: {fully_connected_weights.shape}")
    bias_output = data['bias_output']
    # print(f"Bias output Shape: {bias_output.shape}")
    bias_conv_layers = data['bias_conv_layers']
    # print(f"Bias conv layers Shape: {len(bias_conv_layers)}")
    conv_layers = data['conv_layers']
    # print(f"Conv layers Shape: {conv_layers}")

    # Create the model
    cnn = Convolutional_Neural_Network(input_nodes, output_nodes, conv_layers, fully_connected_weights, bias_output, bias_conv_layers)
    print(f"Model loaded from {filename}")
    return cnn

def open_trained_torch_multiple_fc_cnn(filename):
    data = torch.load(filename, weights_only=True, map_location='cpu')

    # Model parameters
    input_nodes = data['input_nodes']
    # print(f"Input nodes: {input_nodes}")
    output_nodes = data['output_nodes']
    # print(f"Output nodes: {output_nodes}")
    fc_layers = data['fc_layers']
    # print(f"Fully connected weights Shape: {fully_connected_weights.shape}")
    bias_output = data['bias_output']
    # print(f"Bias output Shape: {bias_output.shape}")
    bias_conv_layers = data['bias_conv_layers']
    # print(f"Bias conv layers Shape: {len(bias_conv_layers)}")
    conv_layers = data['conv_layers']
    # print(f"Conv layers Shape: {conv_layers}")

    # Create the model
    cnn = Convolutional_Neural_Network_Multiple_FC_Layers(input_nodes, output_nodes, conv_layers, fc_layers, bias_output, bias_conv_layers)
    print(f"Model loaded from {filename}")
    return cnn


# filename = "cnn_model2.pth" 
# cnn = open_trained_torch_cnn(filename) 
