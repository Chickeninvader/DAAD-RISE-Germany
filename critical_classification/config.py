import argparse
import os

import pandas as pd
from datetime import datetime


class Config:
    """
    Base configuration class for setting up common configurations.

    Attributes:
        device_str (str): Device for computation, e.g., 'cuda:0' or 'cpu'.
        framework (str): The deep learning framework to use, e.g., 'torch'.
        dataset_name (str): The name of the dataset being used.
        video_batch_size (int): Batch size for video data.
        image_batch_size (int): Batch size for image data.
        loss (str): The loss function to use, e.g., 'BCE' for binary cross-entropy.
        num_epochs (int): Number of epochs for training.
        lr (float): Learning rate for the optimizer.
        scheduler (str): The learning rate scheduler, e.g., 'cosine', 'step', exponential.
        save_files (bool): Whether to save models and other outputs.
        representation (str): Data representation format, e.g., 'original'.
        sample_duration (float): Duration to sample from videos in second.
        metadata (pd.DataFrame): Metadata for the dataset.
        infer_all_video (bool): Whether to infer all videos in the dataset (for inference only).
        additional_config (str): Additional configuration information.
        data_location (str): Path to the data location.
        img_representation (str): Image representation format, e.g., 'CHW'.
        model_name (str): Name of the model architecture.
        img_size (int): Size of the input images.
        additional_saving_info (str): Additional information for saving outputs.
        pretrained_path (str): Path to pretrained model weights.
        current_time (str): Current timestamp for saving outputs.
    """
    def __init__(self,
                 additional_config=''):
        # Common configurations
        self.device_str = 'cuda:0'
        self.framework = 'torch'
        self.dataset_name = 'all'
        self.video_batch_size = 4
        self.image_batch_size = 10
        self.loss = 'BCE'
        self.num_epochs = 20
        self.lr = 0.0001
        self.scheduler = 'cosine'
        self.save_files = True
        self.representation = 'original'
        self.sample_duration = self.image_batch_size / 30
        self.metadata = pd.read_excel('critical_classification/critical_dataset/metadata.xlsx')
        self.infer_all_video = False
        self.additional_config = additional_config

        # These variable need to specify during training/inference
        self.data_location = None
        self.img_representation = None
        self.model_name = None
        self.img_size = None
        self.additional_saving_info = None

        # Path to pretrained model weights, if any
        self.pretrained_path = None

        # Current time for saving info
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_file_name(self, file_type, current_epoch=0, additional_info=''):
        """
        Get the file name for saving outputs based on the type and current epoch.

        Args:
            file_type (str): The type of file ('fig', 'model', 'pickle').
            current_epoch (int): The current epoch number.
            additional_info (str): Additional information to append to the file name.

        Returns:
            str: The file path for saving the output.
        """
        file_name = (f"D{self.dataset_name}_M{self.model_name}_lr{self.lr}_loss{self.loss}_e"
                     f"{self.num_epochs}_s{self.scheduler}_A{self.additional_saving_info}")

        os.makedirs(f'critical_classification/output/loss_visualization/{file_name}/', exist_ok=True)
        os.makedirs(f'critical_classification/save_models/{file_name}/', exist_ok=True)

        if file_type == 'fig':
            return f"critical_classification/output/loss_visualization/{file_name}/{file_name}_{additional_info}.png"
        elif file_type == 'model':
            return (f"critical_classification/save_models/{file_name}/"
                    f"{file_name}_e{current_epoch}_{additional_info}.pth")
        elif file_type == 'pickle':
            return f"critical_classification/output/loss_visualization/{file_name}/{file_name}_{additional_info}.pkl"
        else:
            raise NotImplementedError()

    def print_config(self):
        """Print the configuration settings."""
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")


class YOLOv1VideoConfig(Config):
    """
    Configuration class for YOLOv1_video model-specific settings.

    Inherits common configurations from Config class.

    Attributes:
        additional_config (str): Additional configuration information, currently have no_fc for no fc in YOLO last layer
    """
    def __init__(self,
                 additional_config):
        super().__init__(additional_config)
        # Model-specific configurations
        self.model_name = 'YOLOv1_video'
        self.img_representation = 'CHW'
        self.img_size = 448
        self.video_batch_size = 1
        self.additional_saving_info = f'experiment_{self.current_time}_{additional_config}'


class Swin3DConfig(Config):
    def __init__(self,
                 additional_config):
        """
        Configuration class for Swin3D model-specific settings.

        Inherits common configurations from Config class.

        Attributes:
            additional_config (str): Additional configuration information, currently have train_from_scratch
        """
        super().__init__(additional_config)
        # Model-specific configurations
        self.model_name = 'Swin3D'
        self.img_representation = 'CHW'
        self.lr = 0.00001
        self.img_size = 224
        self.additional_saving_info = f'experiment_{self.current_time}'


class ResNet3DConfig(Config):
    def __init__(self,
                 additional_config):
        super().__init__(additional_config)
        # Model-specific configurations
        self.model_name = 'ResNet3D'
        self.img_representation = 'CHW'
        self.img_size = 224
        self.additional_saving_info = f'experiment_{self.current_time}'


class Monocular3DConfig(Config):
    def __init__(self,
                 additional_config):
        super().__init__(additional_config)
        # Model-specific configurations
        self.model_name = 'Monocular3D'
        self.img_representation = 'HWC'
        self.img_size = 224
        self.additional_saving_info = f'experiment_{self.current_time}'


class InferConfig(Config):
    def __init__(self,
                 additional_config):
        super().__init__(additional_config)
        # Model-specific configurations
        self.model_name = None
        self.img_representation = 'CHW'
        self.img_size = 448
        self.video_batch_size = 1
        self.additional_saving_info = f'experiment_{self.current_time}'


def GetConfig():
    """
    Parse command-line arguments and return the appropriate configuration object.

    Returns:
        Config: An instance of a configuration class (YOLOv1VideoConfig, Swin3DConfig, etc.)
    """
    parser = argparse.ArgumentParser(description="Train and/or infer pipeline")
    parser.add_argument('--data_location', type=str, help='Path to the data location',
                        default='critical_classification/critical_dataset/')
    parser.add_argument('--model_name', type=str, help='pytorch model',
                        default=None)
    parser.add_argument('--pretrained_path', type=str, help='Path to model location',
                        default=None)
    parser.add_argument('--infer_all_video', action='store_true', help='Do inference for infer video')
    parser.add_argument('--image_batch_size', type=int, help='Number of frames to sample video',
                        default=15)
    parser.add_argument('--additional_config', type=str, help='Other necessary arguments, ...',
                        default='')

    args = parser.parse_args()

    if args.model_name == 'YOLOv1_video':
        config = YOLOv1VideoConfig(args.additional_config)
    elif args.model_name == 'Swin3D':
        config = Swin3DConfig(args.additional_config)
    else:
        config = InferConfig(args.additional_config)

    config.data_location = args.data_location
    config.image_batch_size = args.image_batch_size
    config.sample_duration = args.image_batch_size / 30
    config.pretrained_path = args.pretrained_path
    config.infer_all_video = args.infer_all_video
    config.print_config()

    return config
