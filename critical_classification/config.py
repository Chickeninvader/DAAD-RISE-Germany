import pandas as pd

# Framework to use, tensorflow or torch
framework = 'torch'

# Batch size for training
batch_size = 1

# Loss function to use, here it's Binary Cross-Entropy (BCE)
loss = 'BCE'

# Number of epochs to train the model
num_epochs = 10

# Learning rate for the optimizer
lr = 0.0001

# Name of the model architecture being used, including Monocular3D, YOLOv1_image, YOLOv1_video, ResNet3D
model_name = 'YOLOv1_video'

# Input image representation, depend on model
img_representation = 'CHW'

# image size of input. Also depend on model
img_size = 448

# Additional information to be appended to the saving file name
additional_saving_info = 'experiment_1'

# Path to pretrained model weights, if any
pretrained_path = None
# pretrained_path = 'critical_classification/save_models/file_name'

# Flag to indicate whether to save the trained model
save_files = True

# Type of data representation being used (e.g., 'original', 'gaussian', etc.)
representation = 'original'

# Duration of video segments to be processed, in seconds
duration = 0.5
# duration = 0.5/15  # take 1 image only

# Load metadata from an Excel file, which contains information about the dataset
metadata = pd.read_excel('critical_classification/dashcam_video/metadata.xlsx')
