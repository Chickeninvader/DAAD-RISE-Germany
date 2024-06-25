import os
import sys

import pandas as pd
import torch.utils.data

sys.path.append(os.getcwd())

from critical_classification.src import utils, models_for_project, data_preprocessing


def initiate(metadata: pd.DataFrame,
             batch_size: int,
             model_name: str = 'resnet_3d',
             pretrained_path: str = None,
             representation: str = 'gaussian',
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

    datasets = data_preprocessing.get_datasets(
        metadata=metadata,
        representation=representation,
        sample_duration=sample_duration,
        model_name=model_name
    )

    device = torch.device('mps' if utils.is_local() and torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else 'cpu'))
    print(f'Using {device}')

    if model_name == 'VideoMAP':
        fine_tuner = models_for_project.VideoMAE()
    else:
        fine_tuner = models_for_project.ResNet3D()
    loaders = data_preprocessing.get_loaders(datasets=datasets,
                                             batch_size=batch_size)

    print(f"Total number of train video: {len(loaders['train'].dataset)}\n"
          f"Total number of test video: {len(loaders['test'].dataset)}")

    return fine_tuner, loaders, device
