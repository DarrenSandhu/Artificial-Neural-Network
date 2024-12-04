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

        # Intialize bias for the convolutional layer
        self.bias_conv_layer = np.random.randn(1, self.conv_layer_kernels.shape[0])
        self.bias_conv_layer_2 = np.random.randn(1, self.conv_layer_2_kernels.shape[0])
        print("Conv Layer Bias:\n ",self.bias_conv_layer)
        print("Conv Layer 2 Bias:\n ",self.bias_conv_layer_2)
    
    def relu(self, x):
        return np.maximum(0, x)    
    
    def sigmoid(self, z_value):
        return 1/(1+np.exp(-z_value))
    
    def convulate(self, image, kernel, stride):
        num_filters = kernel.shape[0]
        patch_number = 0
        for activation_layer in self.conv_layer_activation_images:
            for i in range(num_filters):
                for j in range(activation_layer.shape[0]):
                    for k in range(activation_layer.shape[1]):
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

                        # Calculate the dot product of the patch and the kernel
                        conv_value = np.sum(patch * kernel[i]) + self.bias_conv_layer[0, i]

                        # Apply the ReLU activation function
                        relu_value = self.relu(conv_value)

                        activation_layer[j, k, i] = relu_value

                        patch_number += 1
                        # print("Patch: \n",patch_number)

    def max_pooling(self, stride):
        pool_size = 2  # Typically, max pooling uses a 2x2 pool size
        for max_pool_image in self.conv_layer_max_pool_images:
            for activation_image in self.conv_layer_activation_images:  # Iterate over each image in the activation layer
                for i in range(activation_image.shape[2]):  # Iterate over each filter
                    for j in range(0, activation_image.shape[0] - pool_size + 1, stride):  # Iterate over rows
                        for k in range(0, activation_image.shape[1] - pool_size + 1, stride):  # Iterate over columns
                            # Extract the patch of size (2, 2, depth) for pooling
                            patch = activation_image[j:j+pool_size, k:k+pool_size, i]

                            # Calculate the max of the patch
                            max_value = np.max(patch)

                            # Store the result in the max pooling layer
                            max_pool_image[j // stride, k // stride, i] = max_value
    
    def convulate_2(self, max_pool_images, kernel, stride):
        num_filters = kernel.shape[0]
        patch_number = 0

        for activation_image in self.conv_layer_2_activation_images:
            for i in range(num_filters):
                for j in range(activation_image.shape[0]):
                    for k in range(activation_image.shape[1]):
                        # Calculate the starting and ending index of the patch for this filter
                        start_row = j * stride
                        end_row = start_row + kernel.shape[1]
                        start_col = k * stride
                        end_col = start_col + kernel.shape[2]

                        # Extract the current patch from the images
                        patches = []
                        for max_pool_image in max_pool_images:
                            patches.append(max_pool_image[start_row:end_row, start_col:end_col, :kernel.shape[3]])

                        patches = np.array(patches)
                        # print("Patches Shape: \n",patches)
                        # print("Kernel Shape: \n",kernel[i])
                        # Calculate the dot product of the patch and the kernel
                        conv_value = np.sum(patches * kernel[i]) + self.bias_conv_layer_2[0, i]

                        # Apply the Sigmoid activation function
                        relu_value = self.sigmoid(conv_value)

                        activation_image[j, k, i] = relu_value

                        patch_number += 1
                        # print("Patch: \n",patch_number)

    def flatten(self, images):
        flattened_images = [image.flatten() for image in images]
        flattened_images = np.array(flattened_images)
        flattened_images = flattened_images.flatten()
        return flattened_images
    
    def forward_pass(self, X):
        self.convulate(X, self.conv_layer_kernels, 1)
        self.max_pooling(2)
        self.convulate_2(self.conv_layer_max_pool_images, self.conv_layer_2_kernels, 1)
        self.max_pooling(2)
        self.flattened_images = self.flatten(self.conv_layer_2_max_pool_images)

    
    # def train(self, X, y, epochs, learning_rate):
    #     for epoch in range(epochs):
    #         output = self.forward_pass(X)
    #         self.backpropagation(X, y, learning_rate)
    #         if epoch % 1000 == 0:
    #             loss = np.mean(np.square(y - output))
    #             print(f"Epoch {epoch}, Loss: {loss}")
                    
        
    
            
            
    
    



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
input_nodes = 64 * 64 * 3
output_nodes = 2
conv_layer_kernels = np.random.randn(2, 5, 5, 3)
conv_layer_activation_images = [np.zeros((
                                         images.shape[0] - conv_layer_kernels.shape[1] + 1,
                                         images.shape[1] - conv_layer_kernels.shape[2] + 1,
                                         conv_layer_kernels.shape[0],
                                         )),
                                np.zeros((
                                         images.shape[0] - conv_layer_kernels.shape[1] + 1,
                                         images.shape[1] - conv_layer_kernels.shape[2] + 1,
                                         conv_layer_kernels.shape[0],
                                         ))]



conv_layer_max_pooling_images = [np.zeros((
    conv_layer_activation_images[0].shape[0] // 2,
    conv_layer_activation_images[0].shape[1] // 2,
    conv_layer_activation_images[0].shape[2],
)),
np.zeros((
    conv_layer_activation_images[1].shape[0] // 2,
    conv_layer_activation_images[1].shape[1] // 2,
    conv_layer_activation_images[1].shape[2],
))]


conv_layer_2_kernels = np.random.randn(4, 2, 3, 2)
# print("Conv Layer 2 Kernels: \n",conv_layer_2_kernels)
conv_layer_2_activation_images = []
for i in range(conv_layer_2_kernels.shape[0]):
    conv_layer_2_activation_images.append(np.zeros((
        conv_layer_max_pooling_images[0].shape[0] - conv_layer_2_kernels.shape[1] + 1,
        conv_layer_max_pooling_images[0].shape[1] - conv_layer_2_kernels.shape[2] + 1,
        conv_layer_2_kernels.shape[0],
    )))

conv_layer_2_max_pooling_images = []
for i in range(conv_layer_2_activation_images[0].shape[2]):
    conv_layer_2_max_pooling_images.append(np.zeros((
        conv_layer_2_activation_images[0].shape[0] // 2,
        conv_layer_2_activation_images[0].shape[1] // 2,
        conv_layer_2_activation_images[0].shape[2],
    )))
# [np.zeros((
#     conv_layer_2_activation_images[0].shape[0] // 2,
#     conv_layer_2_activation_images[0].shape[1] // 2,
#     conv_layer_2_activation_images[0].shape[2],
# )),
# np.zeros((
#     conv_layer_2_activation_images[1].shape[0] // 2,
#     conv_layer_2_activation_images[1].shape[1] // 2,
#     conv_layer_2_activation_images[1].shape[2],
# ))]
                                

print("Conv Layer Kernels Shape: ",conv_layer_kernels.shape)
print("Conv Layer Activation Layers Shape: ",conv_layer_activation_images[0].shape)
print("Conv Layer Max Pooling Layers Shape: ",conv_layer_max_pooling_images[0].shape)


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

nn.forward_pass(images)
print("Flattened Images Shape: ",nn.flattened_images.shape)



# print("Conv Layer Activation Layers: \n",nn.conv_layer_activation_images)
# print("Conv Layer Activation Layers Shape: ",nn.conv_layer_activation_images.shape) 

# Train the neural ns


# Show the activation images
# for i in range(2):
#     for j in range(2):
#         cv2.imshow(f"Activation Image {i + 1}", nn.conv_layer_activation_images[i][:, :, j])
#         cv2.waitKey(0)














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