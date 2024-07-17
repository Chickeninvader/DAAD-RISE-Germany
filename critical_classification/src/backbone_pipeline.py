import os
import sys

import torch.utils.data

sys.path.append(os.getcwd())

from critical_classification.src import data_preprocessing
from critical_classification.config import Config
from critical_classification.src.models_for_project_torch import YOLOv1_video_binary, YOLOv1_image_binary, VideoMAE


def initiate(config: Config):
    """
    Initializes and prepares the models, datasets, and devices for training.

    Args:
        config: Config object

    Returns:
        tuple: A tuple containing:
            - fine_tuners (list): A list of VITFineTuner model objects.
            - loaders (dict): A dictionary of data loaders for training, validation, and testing.
            - devices (list): A list of torch.device objects for model placement.
    """
    batch_size = config.batch_size
    model_name = config.model_name
    pretrained_path = config.pretrained_path
    sample_duration = config.sample_duration
    if config.device_str == 'cuda:0' and torch.cuda.is_available():
        device_str = config.device_str
    else:
        device_str = 'cpu'
    # device_str = 'mps' if utils.is_local() and torch.backends.mps.is_available() \
    #     else ("cuda:0" if torch.cuda.is_available() else 'cpu')

    if config.framework == 'torch':
        device = torch.device(device_str)
        print(f'Using {device}')
    elif config.framework == 'tensorflow':
        import tensorflow as tf
        from critical_classification.src.models_for_project_tensorflow import CriticalClassification
        print(f"Tensorflow is using GPU: {tf.config.list_physical_devices('GPU')}")
        device = None
    else:
        raise ValueError('Not support framework')

    datasets = data_preprocessing.get_datasets(
        config=config
    )

    if model_name == 'VideoMAP':
        fine_tuner = VideoMAE()
    elif model_name == 'YOLOv1_image':
        fine_tuner = YOLOv1_image_binary(split_size=14, num_boxes=2, num_classes=13, device=device)
    elif model_name == 'YOLOv1_video':
        fine_tuner = YOLOv1_video_binary(split_size=14, num_boxes=2, num_classes=13, device=device)

    elif model_name == 'Monocular3D':
        fine_tuner = CriticalClassification(
            mono3d_weights_path='critical_classification/save_models/mobilenetv2_weights.h5',
            binary_model_weights_path=pretrained_path,
            device=device_str)
    else:
        print('No model mode, fine tuner not initiated')
        fine_tuner = None

    if pretrained_path is not None:
        print(f'Loading pretrain weight model at {pretrained_path}')
        state_dict = torch.load(pretrained_path, map_location=device)
        fine_tuner.load_state_dict(state_dict)
        print('Pretrained weights loaded successfully.')

    loaders = data_preprocessing.get_loaders(datasets=datasets,
                                             batch_size=batch_size)
    print(f'Model use: {model_name}')
    print(f"Total frames number of train video: {len(loaders['train'].dataset)}\n"
          f"Total frames number of test video: {len(loaders['test'].dataset)}")
    print(f'Each frame are {sample_duration * 30} images')

    return fine_tuner, loaders, device
