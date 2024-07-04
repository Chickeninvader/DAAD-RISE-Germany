import os
import sys

import pandas as pd
import torch.utils.data
import tensorflow as tf

sys.path.append(os.getcwd())

from critical_classification.src import utils, models_for_project, data_preprocessing


def initiate(metadata: pd.DataFrame,
             batch_size: int,
             model_name: str = 'resnet_3d',
             pretrained_path: str = None,
             representation: str = 'original',
             sample_duration: float = 2
             ):
    """
    Initializes save_models, datasets, and devices for training.
    :param model_name:
    :param pretrained_path: Path to a pretrained model (optional).
    :return: A tuple containing:
             - fine_tuners: A list of VITFineTuner model objects.
             - loaders: A dictionary of data loaders for train, val, and test.
             - devices: A list of torch.device objects for model placement.
    """
    framework = 'torch'
    device_str = 'mps' if utils.is_local() and torch.backends.mps.is_available() \
        else ("cuda" if torch.cuda.is_available() else 'cpu')
    device = None

    datasets = data_preprocessing.get_datasets(
        metadata=metadata,
        representation=representation,
        sample_duration=sample_duration,
        model_name=model_name
    )

    if model_name == 'VideoMAP':
        fine_tuner = models_for_project.VideoMAE()
    elif model_name == 'Monocular3D':
        fine_tuner = models_for_project.CriticalClassification(
            mono3d_weights_path='critical_classification/save_models/mobilenetv2_weights.h5',
            binary_model_weights_path=pretrained_path,
            device=device_str)
        framework = 'tensorflow'
    else:
        fine_tuner = models_for_project.ResNet3D()

    loaders = data_preprocessing.get_loaders(datasets=datasets,
                                             batch_size=batch_size)
    print(f'Model use: {model_name}')
    print(f"Total frames number of train video: {len(loaders['train'].dataset)}\n"
          f"Total frames number of test video: {len(loaders['test'].dataset)}")
    print(f'Each frame are {sample_duration * 30} images')

    if framework == 'torch':
        device = torch.device(device_str)
        print(f'Using {device}')
    else:
        print(f"Tensorflow is using GPU: {tf.config.list_physical_devices('GPU')}")
    return fine_tuner, loaders, device
