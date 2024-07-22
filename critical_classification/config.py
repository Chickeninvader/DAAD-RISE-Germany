import pandas as pd
from datetime import datetime


class Config:
    def __init__(self):
        # Device to use: cuda:0 or cpu
        self.device_str = 'cuda:0'

        # Framework to use, tensorflow or torch
        self.framework = 'torch'

        # Dataset to use: Dashcam, Carcarsh or all
        self.dataset_name = 'all'

        # Batch size for training. Set to 1 to get 1 video at the time (use for YOlO)
        self.batch_size = 4

        # Loss function to use, here it's Binary Cross-Entropy (BCE)
        self.loss = 'BCE'

        # Number of epochs to train the model
        # self.num_epochs = 1  # for debugging
        self.num_epochs = 40

        # Learning rate for the optimizer
        self.lr = 0.00001

        # Name of the model architecture being used, including Monocular3D, YOLOv1_image, YOLOv1_video, ResNet3D, Swin3D
        self.model_name = 'YOLOv1_video'
        # self.model_name = None

        # Input image representation, depend on model
        self.img_representation = 'CHW'  # for YOLOv1_image, YOLOv1_video, Swin3D
        # self.img_representation = 'HWC'  # for Monocular3D

        # image size of input. Also depend on model
        # self.img_size = 448  # For YOLOv1_image, YOLOv1_video
        self.img_size = 224  # For Monocular3D, Swin3D

        # Additional information to be appended to the saving file name
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.model_name == 'YOLOv1_video':
            hidden_size = 256
            lstm_layer = 4
            self.additional_saving_info = f'experiment_{current_time}_hidden_size_{hidden_size}_lstm_layer_{lstm_layer}'
        else:
            self.additional_saving_info = f'experiment_{current_time}
        # Path to pretrained model weights, if any
        self.pretrained_path = None
        # self.pretrained_path = 'critical_classification/save_models/file_name'

        # Scheduler: step, exponential or cosine
        self.scheduler = 'cosine'

        # Flag to indicate whether to save the trained model
        self.save_files = True

        # Type of data representation being used (e.g., 'original', 'gaussian', etc.)
        self.representation = 'original'

        # Duration of video segments to be processed, in seconds
        self.sample_duration = 0.5
        # self.duration = 0.5/15  # take 1 image only

        self.FRAME_RATE = 30

        # Load metadata from an Excel file, which contains information about the dataset
        self.metadata = pd.read_excel('critical_classification/critical_dataset/metadata.xlsx')

        # Data_location
        self.data_location = 'critical_classification/critical_dataset/'

        # Infer from all video flag
        self.infer_all_video = False

    def print_config(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
