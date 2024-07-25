import argparse

import pandas as pd
from datetime import datetime


class Config:
    def __init__(self,
                 ):
        # Common configurations
        self.device_str = 'cuda:0'
        self.framework = 'torch'
        self.dataset_name = 'all'
        self.video_batch_size = 4
        self.image_batch_size = 10
        self.loss = 'BCE'
        self.num_epochs = 10
        self.lr = 0.0001
        self.scheduler = 'cosine'
        self.save_files = True
        self.representation = 'original'
        self.sample_duration = self.image_batch_size / 30
        self.metadata = pd.read_excel('critical_classification/critical_dataset/metadata.xlsx')
        self.infer_all_video = False

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

    def get_file_name(self, num_epochs):
        file_name = (f"D{self.dataset_name}_M{self.model_name}_lr{self.lr}_loss{self.loss}_e"
                     f"{num_epochs}_s{self.scheduler}_A{self.additional_saving_info}")
        save_fig_path = f"critical_classification/output/loss_visualization/{file_name}"
        save_model_path = f"critical_classification/save_models/{file_name}.pth"
        return save_fig_path, save_model_path

    def print_config(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")


class YOLOv1VideoConfig(Config):
    def __init__(self,
                 ):
        super().__init__()
        # Model-specific configurations
        self.model_name = 'YOLOv1_video'
        self.img_representation = 'CHW'
        self.img_size = 448
        self.video_batch_size = 1
        hidden_size = 128
        lstm_layer = 4
        self.additional_saving_info = f'experiment_{self.current_time}_hidden_size_{hidden_size}_lstm_layer_{lstm_layer}'


class Swin3DConfig(Config):
    def __init__(self):
        super().__init__()
        # Model-specific configurations
        self.model_name = 'Swin3D'
        self.img_representation = 'CHW'
        self.lr = 0.00001
        self.img_size = 224
        self.additional_saving_info = f'experiment_{self.current_time}'


class ResNet3DConfig(Config):
    def __init__(self):
        super().__init__()
        # Model-specific configurations
        self.model_name = 'ResNet3D'
        self.img_representation = 'CHW'
        self.img_size = 224
        self.additional_saving_info = f'experiment_{self.current_time}'


class Monocular3DConfig(Config):
    def __init__(self):
        super().__init__()
        # Model-specific configurations
        self.model_name = 'Monocular3D'
        self.img_representation = 'HWC'
        self.img_size = 224
        self.additional_saving_info = f'experiment_{self.current_time}'


class InferConfig(Config):
    def __init__(self,
                 ):
        super().__init__()
        # Model-specific configurations
        self.model_name = None
        self.img_representation = 'CHW'
        self.img_size = 448
        self.video_batch_size = 1
        self.additional_saving_info = f'experiment_{self.current_time}'


def GetConfig():
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

    args = parser.parse_args()

    if args.model_name == 'YOLOv1_video':
        config = YOLOv1VideoConfig()
    elif args.model_name == 'Swin3D':
        config = Swin3DConfig()
    else:
        config = InferConfig()

    config.data_location = args.data_location
    config.image_batch_size = args.image_batch_size
    config.sample_duration = args.image_batch_size / 30
    config.pretrained_path = args.pretrained_path
    config.infer_all_video = args.infer_all_video
    config.print_config()

    return config
