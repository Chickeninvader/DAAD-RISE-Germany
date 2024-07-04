import pandas as pd

# Batch size for training
batch_size = 4

# Loss function to use, here it's Binary Cross-Entropy (BCE)
loss = 'BCE'

# Number of epochs to train the model
num_epochs = 2

# Learning rate for the optimizer
lr = 0.0001

# Name of the model architecture being used, including Monocular3D
model_name = 'Monocular3D'

# Additional information to be appended to the saving file name
additional_saving_info = 'experiment_1'

# Path to pretrained model weights, if any
pretrained_path = ''

# Flag to indicate whether to save the trained model
save_files = True

# Type of data representation being used (e.g., 'original', 'gaussian', etc.)
representation = 'original'

# Duration of video segments to be processed, in seconds
duration = 0.5

# Load metadata from an Excel file, which contains information about the dataset
metadata = pd.read_excel('critical_classification/dashcam_video/metadata.xlsx')
