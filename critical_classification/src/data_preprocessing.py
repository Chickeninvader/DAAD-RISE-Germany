import os
import random
import typing
import re

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
)
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Resize,
)
from transformers import VideoMAEImageProcessor


# random.seed(42)
# np.random.seed(42)


def get_video_frames_as_tensor(train_or_test: str,
                               index: int,
                               metadata: pd.DataFrame,
                               model_name: str,
                               sample_duration: int = 2,
                               frame_rate: int = 30,
                               ):
    """
    This function reads a video at a specific time and captures frames for a
    given duration, returning them as a NumPy array.

    Args:
      video_path: Path to the video file.
      start_time: Time in seconds from where to start capturing frames.
      sample_duration: Duration of video
      frame_rate: Optional frame rate of the video (default 30fps).

    Returns:
      A NumPy array with captured video frames stacked together.

    Raises:
      ValueError: If video cannot be opened or frame is not read successfully.
    """
    video_path = metadata['full_path'][index]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file!")

    critical_time = metadata['critical_driving_time'][index]
    video_duration = int(metadata['duration'][index])
    label = 0 if random.random() < 0.5 or not isinstance(critical_time, str) else 1
    start_time = get_critical_mid_time(sample_time=video_duration,
                                       critical_driving_time=critical_time,
                                       label=label)

    start_time_in_ms = int(start_time * 1000) - sample_duration * 500
    duration_in_ms = int(1000 * sample_duration)

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_in_ms)

    frames = []
    for i in range(duration_in_ms // int(1000 / frame_rate)):
        ret, frame = cap.read()
        if ret:
            # frame = cv2.resize(frame, (448, 448))
            # frames.append(transform(frame))
            frames.append(frame)
        else:
            raise ValueError("Error: Frame not read!")

    cap.release()

    # Convert frames to Torch.tensor
    frames_tensor = torch.from_numpy(np.stack(frames, axis=1)).permute(3, 1, 0, 2).float()
    frames_tensor = dataset_transforms(video_tensor={'video': frames_tensor},
                                       train_or_test=train_or_test,
                                       model_name=model_name)
    return frames_tensor, start_time, label


def get_critical_mid_time(critical_driving_time,
                          sample_time,
                          label):
    """
    Generates a random time within or between specified critical time ranges in a video.

    Args:
    critical_driving_time (str): A string containing comma-separated time ranges (e.g., "1:09-1:10, 1:47-1:48"). If the
        string is not provided or invalid, a random time within the video duration is returned.
    sample_time (float): The total duration of the video in seconds.
    label (int): A label indicating whether to pick a time within a critical range (1) or between ranges (other values).

    Returns:
    float: A random time within the specified conditions.
    """

    if not isinstance(critical_driving_time, str):
        return random.uniform(0, sample_time)
    time_ranges = []
    for start_end_time in critical_driving_time.split(","):
        start_time, end_time = start_end_time.split('-')
        start_time_in_second = sum(x * int(t) for x, t in zip([60, 1], start_time.split(":")))
        end_time_in_second = sum(x * int(t) for x, t in zip([60, 1], end_time.split(":")))
        time_ranges.append((start_time_in_second, end_time_in_second))

    # Pick a random index from the list
    random_index = random.randint(0, len(time_ranges) - 1)

    if label == 1:
        # Generate a random float number within the chosen range
        return random.uniform(time_ranges[random_index][0], time_ranges[random_index][1])

    if random_index == len(time_ranges) - 1:
        # Avoid loading video error at the end of the video
        return random.uniform(time_ranges[random_index][1], sample_time - 2)

    return random.uniform(time_ranges[random_index][1], time_ranges[random_index + 1][0])


def get_video_duration_opencv(video_path):
    """Gets the duration of a video using OpenCV."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file at {video_path}!")

    # Get frame rate
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Get total number of frames (may not be accurate for all video formats)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calculate duration in seconds (assuming frame count is accurate)
    duration_in_seconds = num_frames / frame_rate

    cap.release()
    return duration_in_seconds


def is_valid_time_format(critical_driving_time):
    """
    Checks if the given string follows the time range format like:
    '1:09-1:10, 1:47-1:48, 5:07-5:08, 5:29-5:30, 5:55-5:56, 7:31-7:32'

    Args:
    time_str (str): The input string to validate.

    Returns:
    bool: True if the string is valid, False otherwise.
    """
    try:
        if not isinstance(critical_driving_time, str):
            return True

        time_ranges = []
        for start_end_time in critical_driving_time.split(","):
            start_time, end_time = start_end_time.split('-')
            start_time_in_second = sum(x * int(t) for x, t in zip([60, 1], start_time.split(":")))
            end_time_in_second = sum(x * int(t) for x, t in zip([60, 1], end_time.split(":")))
            time_ranges.append((start_time_in_second, end_time_in_second))
        return True
    except ValueError or AttributeError:
        return False


class VideoDataset(Dataset):
    def __init__(self,
                 metadata: pd.DataFrame,
                 duration,
                 frame_rate,
                 test: bool,
                 representation: str,
                 model_name: str):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.metadata = metadata[metadata['train_or_test'] == 'test'] if test \
            else metadata[metadata['train_or_test'] == 'train']

        if representation == 'rectangle':
            folder_path = 'bounding_box_mask_video'
            mask_str = '_mask'
            print('use rectangle representation')
        elif representation == 'gaussian':
            folder_path = 'gaussian_mask_video'
            mask_str = '_mask'
            print('use gaussian representation')
        else:
            folder_path = 'original_video'
            mask_str = ''
            print('use original representation')
        self.metadata['full_path'] = [
            os.path.join(os.getcwd(),
                         f"critical_classification/dashcam_video/{folder_path}", f'{filename[:-4]}{mask_str}.mp4')
            for filename in self.metadata['path']
        ]

        self.metadata['duration'] = [get_video_duration_opencv(path) for path in self.metadata['full_path']]
        self.metadata = self.metadata.reset_index()
        self.duration = duration
        self.frame_rate = frame_rate
        self.train_or_test = 'test' if test else 'train'
        self.model_name = model_name

        invalid_rows = self.metadata['critical_driving_time'].apply(is_valid_time_format)
        if not invalid_rows.all():
            raise ValueError(f"Invalid time format found in rows:\n{invalid_rows}")

    def __len__(self):
        # sample 100x data
        return len(self.metadata) * 100

    def __getitem__(self, idx):
        idx = idx % len(self.metadata)
        video, start_time, label = get_video_frames_as_tensor(train_or_test=self.train_or_test,
                                                              index=idx,
                                                              metadata=self.metadata,
                                                              sample_duration=self.duration,
                                                              frame_rate=self.frame_rate,
                                                              model_name=self.model_name)

        return video, label, (self.metadata['full_path'][idx], start_time)


def get_datasets(metadata: pd.DataFrame,
                 representation: str,
                 sample_duration: float,
                 model_name: str = 'VideoMAP'):
    """
    Instantiates and returns train and test datasets

    Parameters
    ----------
    """
    datasets = {}

    for train_or_test in ['train', 'test']:
        print(f'get error detector loader for {train_or_test}')
        datasets[train_or_test] = VideoDataset(
            metadata=metadata,
            duration=sample_duration,
            frame_rate=30,
            representation=representation,
            test=train_or_test == 'test',
            model_name=model_name
        )

    return datasets


def get_loaders(datasets: typing.Dict[str, torchvision.datasets.ImageFolder],
                batch_size: int,
                ) -> typing.Dict[str, torch.utils.data.DataLoader]:
    """
    Instantiates and returns train and test torch data loaders

    Parameters
    ----------
        :param datasets:
        :param batch_size:
    """
    loaders = {}

    for split in ['train', 'test']:
        loaders[split] = torch.utils.data.DataLoader(
            dataset=datasets[split],
            batch_size=batch_size,
            shuffle=split == 'train',
            num_workers=4,
        )
    return loaders


def dataset_transforms(video_tensor,
                       train_or_test: str,
                       model_name: str = 'VideoMAE') -> torch.tensor:
    """
    Returns the transforms required for the VIT for training or test datasets
    """

    if model_name == 'VideoMAE':
        model_ckpt = "MCG-NJU/videomae-base"
        image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        mean = image_processor.image_mean
        std = image_processor.image_std
        if "shortest_edge" in image_processor.size:
            height = width = image_processor.size["shortest_edge"]
        else:
            height = image_processor.size["height"]
            width = image_processor.size["width"]

        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop((height, width)),
                        ]
                    ),
                ),
            ]
        )

        test_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((height, width)),
                        ]
                    ),
                ),
            ]
        )
    else:
        height, width = 224, 224
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop((height, width)),
                            Lambda(lambda x: x / 255.0),
                        ]
                    ),
                ),
            ]
        )

        test_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Lambda(lambda x: x / 255.0),
                            Resize((height, width)),
                        ]
                    ),
                ),
            ]
        )

    return train_transform(video_tensor)['video'] if train_or_test == 'train' else test_transform(video_tensor)['video']
