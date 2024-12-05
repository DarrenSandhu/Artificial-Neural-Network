import numpy as np
import cv2 as cv2  # OpenCV for image handling
import os
from PIL import UnidentifiedImageError, Image
import warnings
warnings.filterwarnings("ignore", message="Corrupt JPEG data")
BASE_DIR = os.path.dirname(os.path.abspath(__name__))
print(BASE_DIR)

class Convulutional_Neural_Network():

    def __init__(self, input_nodes, output_nodes, conv_layer_kernels, conv_layer_activation_images, conv_layer_max_pool_images, conv_layer_2_activation_images=None, conv_layer_2_kernels=None, conv_layer_2_max_pool_images=None, conv_layer_stride=1):
        self.input_nodes = input_nodes 
        self.output_nodes = output_nodes
        self.conv_layer_kernels = conv_layer_kernels
        self.conv_layer_stride = conv_layer_stride
        self.conv_layer_activation_images = conv_layer_activation_images
        self.conv_layer_max_pool_images = conv_layer_max_pool_images
        self.conv_layer_2_activation_images = conv_layer_2_activation_images
        self.conv_layer_2_kernels = conv_layer_2_kernels
        self.conv_layer_2_max_pool_images = conv_layer_2_max_pool_images

        # Intialize weights and bias for the fully connected layer
        self.fully_connected_weights = np.random.randn(self.conv_layer_2_max_pool_images.shape[0] * self.conv_layer_2_max_pool_images.shape[1] * self.conv_layer_2_max_pool_images.shape[2], self.output_nodes)
        print("Full Connected Weights Shape: ",self.fully_connected_weights.shape)
        self.bias_output = np.random.randn(1, self.output_nodes)
        print("Output Bias Shape: ",self.bias_output.shape)


        # Intialize bias for the convolutional layer
        self.bias_conv_layer = np.random.randn(1, self.conv_layer_kernels.shape[0])
        
        self.bias_conv_layer_2 = np.random.randn(1, self.conv_layer_2_kernels.shape[0] // self.conv_layer_max_pool_images.shape[2])
        print("Conv Layer Bias Shape:\n ",self.bias_conv_layer.shape)
        print("Conv Layer 2 Bias Shape:\n ",self.bias_conv_layer_2.shape)
    
    def relu(self, x):
        return np.maximum(0, x)    
    
    def sigmoid(self, z_value):
        return 1/(1+np.exp(-z_value))
    
    def derivative(self, output):
        return output * (1 - output)
    
    def mean_squared_error(self, target_output, output):
        return np.mean(np.square(target_output - output))
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
    
    def convulate(self, image, kernel, stride):
        num_filters = kernel.shape[0]
        patch_number = 0
        activation_image_row = self.conv_layer_activation_images.shape[0]
        activation_image_col = self.conv_layer_activation_images.shape[1]

        for i in range(num_filters):
            for j in range(activation_image_row):
                for k in range(activation_image_col):
                    # Calculate the starting and ending index of the patch for this filter
                    start_row = j * stride
                    end_row = start_row + kernel.shape[1]
                    start_col = k * stride
                    end_col = start_col + kernel.shape[2]
                    # print("Start Row: ",start_row)
                    # print("End Row: ",end_row)
                    # print("Start Col: ",start_col)
                    # print("End Col: ",end_col)
                    # Extract the current patch from the image
                    patch = image[start_row:end_row, start_col:end_col, :kernel.shape[3]]
                    # print("Patch Shape: ",patch.shape)

                    # Calculate the dot product of the patch and the kernel
                    # print("Kernel Shape: ",self.conv_layer_kernels[i].shape)
                    conv_value = np.sum(patch * self.conv_layer_kernels[i]) + self.bias_conv_layer[0, i]

                    # Apply the ReLU activation function
                    relu_value = self.relu(conv_value)

                    self.conv_layer_activation_images[j, k, i] = relu_value

                    patch_number += 1
                    # print("Patch: \n",patch_number)

    def max_pooling(self, stride, activation_images, max_pool_images):
        pool_size = 2  # Typically, max pooling uses a 2x2 pool size
        # Determine the shape of the output max-pooling images
        num_filters = activation_images.shape[2]

        for i in range(num_filters):
            for j in range(0, activation_images.shape[0] - pool_size + 1, stride):
                for k in range(0, activation_images.shape[1] - pool_size + 1, stride):
                    # Extract the patch of size (2, 2, depth) for pooling
                    patch = activation_images[j:j+pool_size, k:k+pool_size, i]

                    # Calculate the max of the patch
                    max_value = np.max(patch)

                    # Store the result in the max pooling layer
                    max_pool_images[j // stride, k // stride, i] = max_value

        # for max_pool_image in self.conv_layer_max_pool_images:
        #     for activation_image in self.conv_layer_activation_images:  # Iterate over each image in the activation layer
        #         for i in range(activation_image.shape[2]):  # Iterate over each filter
        #             for j in range(0, activation_image.shape[0] - pool_size + 1, stride):  # Iterate over rows
        #                 for k in range(0, activation_image.shape[1] - pool_size + 1, stride):  # Iterate over columns
        #                     # Extract the patch of size (2, 2, depth) for pooling
        #                     patch = activation_image[j:j+pool_size, k:k+pool_size, i]

        #                     # Calculate the max of the patch
        #                     max_value = np.max(patch)

        #                     # Store the result in the max pooling layer
        #                     max_pool_image[j // stride, k // stride, i] = max_value
    
    def convulate_2(self, max_pool_images, kernel, stride):
        num_filters = kernel.shape[0]
        patch_number = 0
        activation_image_row = self.conv_layer_2_activation_images.shape[0]
        activation_image_col = self.conv_layer_2_activation_images.shape[1]

        filters_across_images = max_pool_images.shape[2] # Number of filters altogether per images

        number_of_filters = int(kernel.shape[0] / filters_across_images)
        print("Number of Filters: ",number_of_filters)
        # Use 16 filters across each image 32 times
        kernel_index = 0
        for i in range(0, num_filters, filters_across_images):
            for j in range(activation_image_row):
                for k in range(activation_image_col):
                    # Calculate the starting and ending index of the patch for this filter
                    start_row = j * stride
                    end_row = start_row + kernel.shape[1]
                    start_col = k * stride
                    end_col = start_col + kernel.shape[2]

                    # Extract the current patch from the images
                    patches = []
                    for f in range(self.conv_layer_activation_images.shape[2]):
                        max_pool_image = max_pool_images[:, :, f]
                        patches.append(max_pool_image[start_row:end_row, start_col:end_col])
                    
                    filters = []
                    for l in range(filters_across_images):
                        filters.append(kernel[i + l][2])
                        # print("Kernel: ",kernel[0][i + l].shape)
                    
                    patches = np.array(patches)
                    filters = np.array(filters)
                    # print("Patches Shape: \n",patches.shape)
                    # print("Kernel: \n",filters)
                    
                    conv_value = np.sum(patches * filters) + self.bias_conv_layer_2[0, kernel_index]

                    # Apply the Sigmoid activation function
                    relu_value = self.sigmoid(conv_value)
                    # print("Relu Value: ",relu_value)
                    self.conv_layer_2_activation_images[j, k, kernel_index] = relu_value

                    # patch_number += 1
                    # print("Patch: \n",patch_number)

            kernel_index += 1
        


    def flatten(self, images):
        flattened_images = [image.flatten() for image in images]
        flattened_images = np.array(flattened_images)
        flattened_images = flattened_images.flatten()
        return flattened_images
    
    def forward_pass(self, X):
        self.convulate(X, self.conv_layer_kernels, 1)
        self.max_pooling(2, self.conv_layer_activation_images, self.conv_layer_max_pool_images)
        self.convulate_2(self.conv_layer_max_pool_images, self.conv_layer_2_kernels, 1)
        self.max_pooling(2, self.conv_layer_2_activation_images, self.conv_layer_2_max_pool_images)
        self.flattened_images = self.flatten(self.conv_layer_2_max_pool_images)

        # Fully connected layer
        self.final_input = np.dot(self.flattened_images, self.fully_connected_weights) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backpropagation(self, X, y, learning_rate):
        # 1. Compute the gradient of the loss with respect to the final output
        output_error = self.final_output - y # This is the derivative of MSE w.r.t. output
        print("Output Error: ",output_error)
        d_output = self.derivative(self.final_output) * output_error  # This is the derivative of the sigmoid function
        print("D Output: ",d_output)
        d_output_delta = d_output * output_error
        print("D Output Delta: ",d_output_delta)
        d_output_change = np.dot(self.flattened_images.T, d_output_delta)  # Gradient of the weights of the fully connected layer
        
        # 2. Backpropagate the gradient through the fully connected layer
        fully_conndected_error = np.dot(d_output, self.fully_connected_weights.T)  # Gradient of the fully connected layer
        print("Fully Connected Error: ",fully_conndected_error)
        # d_weights_fc = np.dot(self.flattened_images.T, d_output)  # Gradient of the weights of the fully connected layer
        # print("Weights FC: ",d_weights_fc)
        fully_conndected_delta = fully_conndected_error * self.derivative(self.flattened_images)
        d_bias_fc = d_output  # Gradient of the bias of the fully connected layer
        
        # Update fully connected weights and bias
        self.fully_connected_weights -= learning_rate * d_output_change
        self.bias_output -= learning_rate * d_bias_fc

        # 3. Backpropagate the gradient through the second convolutional layer (Conv Layer 2)
        # We need to compute the gradient for the convolutional layer 2 weights, biases, and activations
        d_activation_layer_2 = np.zeros_like(self.conv_layer_2_activation_images)
        
        for activation_image in self.conv_layer_2_activation_images:
            i = 0
            for j in range(activation_image.shape[1]):  # Iterate over the rows
                for k in range(activation_image.shape[2]):  # Iterate over the columns
                    d_activation_layer_2[i, j, k] = d_output[0, i]  # Propagate the output error backward
            i += 1

        # Now compute the gradients for the kernels in Conv Layer 2
        d_kernel_2 = np.zeros_like(self.conv_layer_2_kernels)
        for i in range(self.conv_layer_2_kernels.shape[0]):
            for j in range(self.conv_layer_2_kernels.shape[1]):
                for k in range(self.conv_layer_2_kernels.shape[2]):
                    # Compute the gradient for each kernel by convolving the activation layer with the error
                    d_kernel_2[i, j, k] = np.sum(d_activation_layer_2 * self.conv_layer_2_activation_images[:, j, k])
        
        # Update Conv Layer 2 kernels and biases
        self.conv_layer_2_kernels -= learning_rate * d_kernel_2
        self.bias_conv_layer_2 -= learning_rate * np.sum(d_activation_layer_2)

        # 4. Backpropagate the gradient through the first convolutional layer (Conv Layer 1)
        d_activation_layer_1 = np.dot(self.conv_layer_activation_images, d_activation_layer_2)  # Gradient of activation layer 1
        d_kernel_1 = np.sum(d_activation_layer_1 * self.conv_layer_kernels, axis=0)  # Gradient of Conv Layer 1 kernel

        # Update Conv Layer 1 kernels and biases
        self.conv_layer_kernels -= learning_rate * d_kernel_1
        self.bias_conv_layer -= learning_rate * np.sum(d_activation_layer_1)

        return output_error

    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward_pass(X)
            self.backpropagation(X, y, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")
                    
        
    
            
            
    
    



# Test cat image
cat_directory = os.path.join(BASE_DIR, 'training_data', 'cat')
image_size = (64, 64)

cat_image = cv2.imread(os.path.join(cat_directory, '1.jpg'))
cat_image = cv2.resize(cat_image, image_size)

cat_image_2 = cv2.imread(os.path.join(cat_directory, '2.jpg'))
cat_image_2 = cv2.resize(cat_image_2, image_size)

# Normalize the image data
# images = []
# images.append(cat_image)
# images.append(cat_image_2)
images = np.array(cat_image) / 255.0
# print("Images: ", images)

# Create labels for the images
labels = np.array([1, 1])
# print("Labels: ", labels)

print("Images Shape: ", images.shape)


# Create a neural network

############################################
# Convolutional Neural Network Parameters #
############################################
input_nodes = 64 * 64 * 3
output_nodes = 2
target = np.array([1, 0])


############################################
# Convolutional Layer 1 Parameters #
############################################
conv_layer_kernels = np.random.randn(16, 5, 5, 3)
conv_layer_activation_images = (
np.zeros((
    images.shape[0] - conv_layer_kernels.shape[1] + 1,
    images.shape[1] - conv_layer_kernels.shape[2] + 1,
    conv_layer_kernels.shape[0]
)))   

conv_layer_max_pooling_images = (
np.zeros((
    conv_layer_activation_images.shape[0] // 2,
    conv_layer_activation_images.shape[1] // 2,
    conv_layer_activation_images.shape[2],
)))


############################################
# Convolutional Layer 2 Parameters #
############################################
conv_layer_2_kernels = np.random.randn(conv_layer_max_pooling_images.shape[2] * 32, 3, 3, 3)
conv_layer_2_activation_images = (
np.zeros((
    conv_layer_max_pooling_images.shape[0] - conv_layer_2_kernels.shape[1] + 1,
    conv_layer_max_pooling_images.shape[1] - conv_layer_2_kernels.shape[2] + 1,
    conv_layer_max_pooling_images.shape[2] * 2
)))

conv_layer_2_max_pooling_images = (
np.zeros((
    conv_layer_2_activation_images.shape[0] // 2,
    conv_layer_2_activation_images.shape[1] // 2,
    conv_layer_2_activation_images.shape[2]
)))

############################################
############################################
print()                 

print("Conv Layer Kernels Shape: ",conv_layer_kernels.shape)
print("Activation Layer 1 Shape: \n",conv_layer_activation_images.shape)
print("Conv Layer Max Pooling Layers Shape: ",conv_layer_max_pooling_images.shape)


print("\n")


print("Conv Layer 2 Activation Images Shape: \n",conv_layer_2_activation_images.shape)
print("Conv Layer 2 Kernels Shape: \n",conv_layer_2_kernels.shape)
print("Conv Layer 2 Max Pooling Layers Shape: \n",conv_layer_2_max_pooling_images.shape)

print("\n")
nn = Convulutional_Neural_Network(
    input_nodes, 
    output_nodes,
    conv_layer_kernels, 
    conv_layer_activation_images=conv_layer_activation_images, 
    conv_layer_max_pool_images=conv_layer_max_pooling_images,
    conv_layer_2_kernels=conv_layer_2_kernels,
    conv_layer_2_activation_images=conv_layer_2_activation_images,
    conv_layer_2_max_pool_images=conv_layer_2_max_pooling_images
    )

final_output = nn.forward_pass(images)
print("Flattened Images Shape: ",nn.flattened_images.shape)
print("Final Output: ",final_output)


# nn.convulate(images, nn.conv_layer_kernels, 1)

# nn.max_pooling(2, nn.conv_layer_activation_images, nn.conv_layer_max_pool_images)

# nn.convulate_2(nn.conv_layer_max_pool_images, nn.conv_layer_2_kernels, 1)

# nn.max_pooling(2, nn.conv_layer_2_activation_images, nn.conv_layer_2_max_pool_images)



# Print ammount of activation layer 2 images
# print("Conv Layer 2 Activation Images Shape: ",nn.conv_layer_2_activation_images.shape)
# for filter_index in range(nn.conv_layer_2_activation_images.shape[2]):
#     max_pool_image = nn.conv_layer_2_max_pool_images[:, :, filter_index]
#     print("Max Pool Image Shape: ",max_pool_image.shape)
#     cv2.imshow(f"Max Pool Image {filter_index}", max_pool_image)
#     cv2.waitKey(0)

# for filter_index in range(nn.conv_layer_2_activation_images.shape[2]):
#     activation_layer_2_image = nn.conv_layer_2_activation_images[:, :, filter_index]
#     print(f"Activation Layer 2 Image {filter_index} Shape : ",activation_layer_2_image.shape)
#     cv2.imshow(f"Activation Layer 2 Image {filter_index}", activation_layer_2_image)
#     cv2.waitKey(0)

# for filter_index in range(nn.conv_layer_activation_images.shape[2]):
#     max_pool_image = nn.conv_layer_max_pool_images[:, :, filter_index]
#     print("Max Pool Image Shape: ",max_pool_image.shape)
#     cv2.imshow(f"Max Pool Image {filter_index}", max_pool_image)
#     cv2.waitKey(0)

# for filter_index in range(nn.conv_layer_activation_images.shape[2]):
    
#     activation_image = conv_layer_activation_images[:, :, filter_index]

#     cv2.imshow(f"Activation Image {filter_index} Shape: ",activation_image)
#     cv2.waitKey(0)

# print("Activation Layer 1 Size: ",len(nn.conv_layer_activation_images))
# Scale the values to 0-255 if they are in the range 0-1 (assuming ReLU or similar)
    # activation_image = np.clip(activation_image * 255, 0, 255).astype(np.uint8)
    # cv2.imshow(f"Filter {filter_index + 1}, Activation Image {filter_index}", nn.conv_layer_activation_images[:, :, filter_index])
    # Extract the activation image for the current filter


# nn.train(images, target, 10000, 0.01)



# print("Conv Layer Activation Layers: \n",nn.conv_layer_activation_images)
# print("Conv Layer Activation Layers Shape: ",nn.conv_layer_activation_images.shape) 

# Train the neural ns















# Load the training data for cats
# cat_directory = os.path.join(BASE_DIR, 'training_data', 'cat')
# cat_directory_files = os.listdir(cat_directory)

# images = []
# labels = []
# image_size = (64, 64)
# for filename in cat_directory_files:
#     img_path = os.path.join(cat_directory, filename)
#     assert os.path.exists(img_path), f"File not found: {img_path}"
#     try:
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, image_size)
#         images.append(img)
#         labels.append(1)
#     except Exception as e:
#         # print(f"Error processing image {img_path}: {e}")
#         continue  # Skip corrupt images

# # Load the training data for dogs
# dog_directory = os.path.join(BASE_DIR, 'training_data', 'dog')
# dog_directory_files = os.listdir(dog_directory)
# for filename in dog_directory_files:
#     img_path = os.path.join(dog_directory, filename)
#     assert os.path.exists(img_path), f"File not found: {img_path}"
#     try:
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, image_size)
#         images.append(img)
#         labels.append(0)
#     except Exception as e:
#         # print(f"Error processing image {img_path}: {e}")
#         continue  # Skip corrupt



# nn.convulate(images, nn.conv_layer_kernels, 1)

# print("Conv Layer Activation Shape: ",nn.conv_layer_activation_images[0].shape)
# print()

# print("Conv Layer 2 Kernels Shape: ",nn.conv_layer_2_kernels.shape)
# # print("Conv Layer 2 Kernels:\n ",nn.conv_layer_2_kernels)
# nn.max_pooling(2)

# # max_pool_image = nn.conv_layer_max_pool_images
# # print(max_pool_image.shape)

# # print("\n")
# # print("Conv Layer 2 Kernels Shape: \n",nn.conv_layer_2_kernels.shape)
# # print("Conv Layer 2 Activation Shape: \n",nn.conv_layer_2_activation_images.shape)

# print("\n")
# # print("Conv Layer 2 Activation Image 1 Before: \n",nn.conv_layer_2_activation_images[0])
# nn.convulate_2(nn.conv_layer_max_pool_images, nn.conv_layer_2_kernels, 1)
# # print("Conv Layer 2 Activation Image 1 After: \n",nn.conv_layer_2_activation_images[0])

# print("Conv Layer 2 Activation Image 1  Shape: ",conv_layer_2_activation_images[3].shape)
# print("Conv Layer 2 Max Pooling Layers Shape: ",conv_layer_2_max_pooling_images[0].shape)

# nn.max_pooling(2)

# flattened_images = [image.flatten() for image in nn.conv_layer_2_max_pool_images]
# flattened_images = np.array(flattened_images)
# flattened_images = flattened_images.flatten()
# print("Flattened Images Shape: ",flattened_images.shape)
