import os
import random
import re
import sys
import typing

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import VideoMAEImageProcessor

from critical_classification.src.utils import add_row_metadata

sys.path.append(os.getcwd())

from critical_classification.src import utils
from critical_classification.config import Config


def get_frames_from_cv2(video_path: str,
                        start_time_in_ms: int,
                        sample_duration_in_ms: int,
                        frame_rate: int):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file!")

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_in_ms)

    frames = []

    # Always get 15 frames
    for i in range(int(frame_rate / 2)):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            continue
        raise ValueError(
            f"{video_path} sample at time: {start_time_in_ms / 1000} second, "
            f"current frame: {i}"
            f"with sample duration {sample_duration_in_ms / 1000} second, has some errors"
        )

    cap.release()
    return frames


def get_frames_from_moviepy(video_path, start_time_in_ms, sample_duration_in_ms, frame_rate):
    clip = VideoFileClip(video_path)
    start_time = start_time_in_ms / 1000
    end_time = start_time + (sample_duration_in_ms / 1000)
    frames = [frame for frame in clip.subclip(start_time, end_time).iter_frames(fps=frame_rate)]
    return frames


def get_critical_mid_time(sample_time,
                          video_duration):
    """
    Generates a random time (sec) within or between specified critical time ranges in a video.

    Args:
    sample_time (float): The time in seconds that we want to sample video.

    Returns:
    float: A random time in second within the specified conditions.
    """
    if not isinstance(sample_time, str):
        # get random time for car crash dataset. the frame rate is 10fps. Pick random 15 consecutive frames and get
        # mid-time of them
        num_critical_frame = sample_time.count(1)
        random_idx = random.randint(0, num_critical_frame)
        random_time = 5 - random_idx / 10 - 10 / 10

    else:
        # get random time for other dataset. the frame rate is approx 30fpx.
        start_time, end_time = sample_time.split('-')
        start_time_in_second = sum(x * int(t) for x, t in zip([60, 1], start_time.split(":")))
        end_time_in_second = sum(x * int(t) for x, t in zip([60, 1], end_time.split(":")))

        random_time = random.uniform(start_time_in_second if start_time_in_second != 0 else 1,
                                     end_time_in_second if end_time_in_second >= video_duration else video_duration - 1)

    return random_time


def get_video_duration_opencv(video_path):
    """Gets the duration of a video using OpenCV."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(video_path)
        return 0
        # raise ValueError(f"Error opening video file at {video_path}!")

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
    '1:09-1:10, 1:47-1:48, 5:07-5:08, 5:29-5:30, 5:55-5:56, 7:31-7:32' or '0, 1, 1, ..., 0'

    Args:
    critical_driving_time (str): The input string to validate.

    Returns:
    bool: True if the string is valid, False otherwise.
    """
    if not isinstance(critical_driving_time, str):
        return True

    # Pattern for the "0, 1, 1, ..., 0" format
    zero_one_pattern = r'^(0|1)(, (0|1))*$'
    if re.fullmatch(zero_one_pattern, critical_driving_time):
        return True

    # Pattern for the time range format "1:09-1:10, 1:47-1:48, etc."
    time_range_pattern = r'^\d{1,2}:\d{2}-\d{1,2}:\d{2}(, \d{1,2}:\d{2}-\d{1,2}:\d{2})*$'
    if re.fullmatch(time_range_pattern, critical_driving_time):
        try:
            for start_end_time in critical_driving_time.split(", "):
                start_time, end_time = start_end_time.split('-')

                # Validate times
                start_time_parts = start_time.split(':')
                end_time_parts = end_time.split(':')

                if not (0 <= int(start_time_parts[0]) <= 23 and 0 <= int(start_time_parts[1]) <= 59):
                    return False
                if not (0 <= int(end_time_parts[0]) <= 23 and 0 <= int(end_time_parts[1]) <= 59):
                    return False

                # Convert times to seconds
                start_time_in_seconds = int(start_time_parts[0]) * 60 + int(start_time_parts[1])
                end_time_in_seconds = int(end_time_parts[0]) * 60 + int(end_time_parts[1])

                # Ensure start time is less than end time
                if start_time_in_seconds >= end_time_in_seconds:
                    return False
            return True
        except ValueError:
            return False

    return False


def get_video_frames_as_tensor(config: Config,
                               train_or_test: str,
                               metadata: pd.DataFrame,
                               idx: int
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

    index = idx
    sample_duration = config.sample_duration
    frame_rate = config.FRAME_RATE
    model_name = config.model_name
    img_representation = config.img_representation
    img_size = config.img_size

    sample_time = metadata['sample_duration'][index]
    video_duration = metadata['video_duration'][index]

    start_time = get_critical_mid_time(sample_time=sample_time,
                                       video_duration=video_duration)

    start_time_in_ms = int(start_time * 1000 - sample_duration * 1000)
    sample_duration_in_ms = int(1000 * sample_duration)
    video_path = metadata['full_path'][index]

    # try:
    if video_path.lower().endswith('.mp4'):
        frames = get_frames_from_cv2(video_path, start_time_in_ms, sample_duration_in_ms, frame_rate)
    elif video_path.lower().endswith('.mov'):
        frames = get_frames_from_moviepy(video_path, start_time_in_ms, sample_duration_in_ms, frame_rate)
    else:
        raise FileNotFoundError(f'file not support: {video_path}')

    frames_array = np.stack(frames, axis=0)
    frames_array = dataset_transforms(video_array=frames_array,
                                      train_or_test=train_or_test,
                                      img_size=img_size,
                                      model_name=model_name)
    if img_representation == 'HWC':
        # Final shape: (num_frames, height, width, channel)
        assert frames_array.shape[3] == 3, (f'output representation not match with HWC, shape {frames_array.shape},'
                                            f'video path: {video_path}')
    else:
        # Final shape: (num_frames, channel, height, width)
        assert frames_array.shape[1] == 3, (f'output representation not match with CHW, shape {frames_array.shape},'
                                            f'video path: {video_path}')

    return frames_array, start_time


class CriticalDataset(Dataset):
    def __init__(self,
                 config: Config,
                 test: bool):
        """
        Arguments:
            test:
            config:
        """

        self.metadata = config.metadata
        self.img_representation = config.img_representation
        self.model_name = config.model_name
        self.img_size = config.img_size
        self.data_location = config.data_location
        self.dataset_name = config.dataset_name
        self.frame_rate = config.FRAME_RATE
        self.config = config

        if self.dataset_name == 'Dashcam':
            self.metadata = self.metadata[self.metadata['video_type'] != 'Carcarsh']
        elif self.dataset_name == 'Carcrash':
            self.metadata = self.metadata[self.metadata['video_type'] != 'Dashcam']

        print(f'dataset in use: bdd100k and {self.dataset_name}')

        self.metadata = self.metadata[self.metadata['train_or_test'] == 'test'] if test \
            else self.metadata[self.metadata['train_or_test'] == 'train']

        base_path_mapping = {
            'Dashcam': os.path.join(config.data_location, 'Dashcam_video/'),
            'Carcrash': os.path.join(config.data_location, 'Car_crash_video/'),
            'Bdd100k': os.path.join(config.data_location, 'Bdd100k_video/')
        }

        # Update the full_path assignment in the __init__ method
        self.metadata['full_path'] = [
            os.path.join(base_path_mapping.get(video_type, ''), f'{filename}')
            for video_type, filename in zip(self.metadata['video_type'], self.metadata['path'])
        ]

        # Filter metadata based on path existence
        valid_indices = [True if os.path.exists(path) else False for path in self.metadata['full_path']]
        self.metadata = self.metadata[valid_indices]

        self.metadata['duration'] = [int(get_video_duration_opencv(path)) for path in self.metadata['full_path']]
        self.metadata = self.metadata.reset_index(drop=True)

        invalid_rows = self.metadata['critical_driving_time'].apply(is_valid_time_format)
        if not invalid_rows.all():
            raise ValueError(f"Invalid time format found in rows with data:\n{self.metadata[~invalid_rows]}")

        # Add sample_duration and labels
        self.metadata = self.expand_metadata(self.metadata)

        self.train_or_test = 'test' if test else 'train'
        self.num_positive_class = sum(self.metadata['label'] == 1)
        self.num_negative_class = sum(self.metadata['label'] == 0)

        print(f'{self.train_or_test} dataset contains: ')
        print(utils.green_text(f"{self.num_positive_class} videos with 15-16 frames contain critical data"))
        print(utils.red_text(f"{self.num_negative_class} videos with 15-16 frames contain non-critical data"))

        self.config = config

    def expand_metadata(self, metadata):
        expanded_metadata = []
        for _, row in metadata.iterrows():
            video_path = row['full_path']
            video_duration = row['duration']
            critical_times = row.get('critical_driving_time', '')
            dataset_name = row.get('video_type', '')

            expanded_metadata = add_row_metadata(expanded_metadata=expanded_metadata,
                                                 dataset_name=dataset_name,
                                                 video_path=video_path,
                                                 video_duration=video_duration,
                                                 critical_times=critical_times)

        return pd.DataFrame(expanded_metadata)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        idx = idx % len(self.metadata)
        video, start_time = get_video_frames_as_tensor(config=self.config,
                                                       train_or_test=self.train_or_test,
                                                       idx=idx,
                                                       metadata=self.metadata)
        label = self.metadata['label'][idx]

        return video, label, (self.metadata['full_path'][idx], start_time)


def get_datasets(config: Config):
    """
    Instantiates and returns train and test datasets

    Parameters
    ----------
    """
    datasets = {}

    for train_or_test in ['train', 'test']:
        print(f'get error detector loader for {train_or_test}')
        datasets[train_or_test] = CriticalDataset(
            config=config,
            test=train_or_test == 'test'
        )

    return datasets


def get_loaders(datasets: typing.Dict[str, CriticalDataset],
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
        weight = [0, 0]
        weight[1] = 1 / datasets[split].num_positive_class
        weight[0] = 1 / datasets[split].num_negative_class
        samples_weight = np.array(
            [weight[idx]
             for idx in np.where(datasets[split].metadata['label'] == 1, 1, 0)])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

        loaders[split] = torch.utils.data.DataLoader(
            dataset=datasets[split],
            batch_size=batch_size,
            sampler=sampler if split == 'train' else None,
            num_workers=4,
            pin_memory=True
        )
    return loaders


def normalize(image, mean=None, std=None):
    if mean is not None and std is not None:
        return (image - mean) / std
    return image / 255.0


def resize(image, height, width):
    return cv2.resize(image, (width, height))


def train_transform(video, height, width, mean, std, min_size=256, max_size=320):
    transformed_video = []
    for frame in video:
        frame = normalize(frame, mean, std)
        frame = resize(frame, height, width)
        # frame = random_short_side_scale(frame, min_size, max_size)
        # frame = random_crop(frame, height, width)
        transformed_video.append(frame)
    return np.array(transformed_video)


def test_transform(video, height, width, mean, std):
    transformed_video = []
    for frame in video:
        frame = normalize(frame, mean, std)
        frame = resize(frame, height, width)
        transformed_video.append(frame)
    return np.array(transformed_video)


def dataset_transforms(video_array: typing.Union[torch.Tensor, np.array],
                       train_or_test: str,
                       img_size: int,
                       model_name: str) -> torch.tensor:
    """
    Returns the transforms required for the VIT for training or test datasets
    """
    mean, std = None, None
    if model_name == 'YOLOv1_video' or model_name is None:
        transform = v2.Compose([
            v2.Resize((img_size, img_size), Image.NEAREST),
            v2.ToTensor(),
        ])

        if video_array.ndim == 3:
            return transform(video_array)

        frames = []
        for frame in video_array:
            img = Image.fromarray(frame)
            img_tensor = transform(img)
            frames.append(img_tensor)
        return torch.stack(frames)

    elif model_name == 'Swin3D':
        transform = v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if video_array.ndim == 3:
            return transform(video_array)

        frames = []
        for frame in video_array:
            img = Image.fromarray(frame)
            img_tensor = transform(img)
            frames.append(img_tensor)
        return torch.stack(frames)

    elif model_name == 'VideoMAE':
        model_ckpt = "MCG-NJU/videomae-base"
        image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        mean = image_processor.image_mean
        std = image_processor.image_std
        if "shortest_edge" in image_processor.size:
            height = width = image_processor.size["shortest_edge"]
        else:
            height = image_processor.size["height"]
            width = image_processor.size["width"]

    else:
        height, width = img_size, img_size

    if train_or_test == 'train':
        return train_transform(video_array, height, width, mean, std)
    else:
        return test_transform(video_array, height, width, mean, std)
