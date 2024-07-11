import os
import sys

import pandas as pd
import torch.utils.data

sys.path.append(os.getcwd())

from critical_classification.src import utils, data_preprocessing
from critical_classification import config
from critical_classification.src.models_for_project_torch import YOLOv1_video_binary, YOLOv1_image_binary, VideoMAE

if config.framework == 'tensorflow':
    import tensorflow as tf
    from critical_classification.src.models_for_project_tensorflow import CriticalClassification


def initiate(metadata: pd.DataFrame,
             batch_size: int,
             img_representation: str,
             img_size: int,
             model_name: str = None,
             pretrained_path: str = None,
             sample_duration: float = 2
             ):
    """
    Initializes and prepares the models, datasets, and devices for training.

    Args:
        metadata (pd.DataFrame): Metadata containing information about the dataset.
        batch_size (int): The size of batches for data loading.
        model_name (str, optional): The name of the model to use. Defaults to None.
        pretrained_path (str, optional): Path to a pretrained model (if any). Defaults to None.
        representation (str, optional): The representation format of the data. Defaults to 'original'.
        sample_duration (float, optional): The duration of each sample in seconds. Defaults to 2.

    Returns:
        tuple: A tuple containing:
            - fine_tuners (list): A list of VITFineTuner model objects.
            - loaders (dict): A dictionary of data loaders for training, validation, and testing.
            - devices (list): A list of torch.device objects for model placement.
    """
    device_str = 'mps' if utils.is_local() and torch.backends.mps.is_available() \
        else ("cuda" if torch.cuda.is_available() else 'cpu')
    if config.framework == 'torch':
        device = torch.device(device_str)
        print(f'Using {device}')
    else:
        device = None
        print(f"Tensorflow is using GPU: {tf.config.list_physical_devices('GPU')}")

    datasets = data_preprocessing.get_datasets(
        metadata=metadata,
        img_representation=img_representation,
        sample_duration=sample_duration,
        model_name=model_name,
        img_size=img_size,
    )

    if pretrained_path is not None:
        print(f'Loading pretrain weight model at {pretrained_path}')

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

    loaders = data_preprocessing.get_loaders(datasets=datasets,
                                             batch_size=batch_size)
    print(f'Model use: {model_name}')
    print(f"Total frames number of train video: {len(loaders['train'].dataset)}\n"
          f"Total frames number of test video: {len(loaders['test'].dataset)}")
    print(f'Each frame are {sample_duration * 30} images')

    return fine_tuner, loaders, device
