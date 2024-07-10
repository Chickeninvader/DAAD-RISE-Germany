import os
import random
import typing

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor

from critical_classification.src import utils


# random.seed(42)
# np.random.seed(42)


def get_video_frames_as_tensor(train_or_test: str,
                               index: int,
                               metadata: pd.DataFrame,
                               model_name: str,
                               img_representation: str,
                               img_size: int,
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

    # Decide to return positive or negative label: if the video is dashcam, label set to 1
    label = 0 if metadata['video_type'][index] != 'Dashcam' or not isinstance(critical_time, str) else 1
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
            frames.append(frame)
        else:
            raise ValueError("Error: Frame not read!")

    cap.release()

    frames_tensor = np.stack(frames, axis=0)
    frames_tensor = dataset_transforms(video_tensor=frames_tensor,
                                       train_or_test=train_or_test,
                                       img_size=img_size,
                                       model_name=model_name)
    if img_representation == 'HWC':
        # Final shape: (num_frames, height, width, channel)
        return frames_tensor, start_time, label
    else:
        # Final shape: (num_frames, channel, height, width)

        return frames_tensor.transpose((0, 3, 1, 2)), start_time, label


def get_critical_mid_time(critical_driving_time,
                          sample_time,
                          label: int):
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

    if not isinstance(critical_driving_time, str) and label == 0:
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
        # Avoid getting video error at the end of the video
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
                 duration: int,
                 img_size: int,
                 test: bool,
                 img_representation: str,
                 model_name: str):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if representation == 'rectangle':
        #     folder_path = 'bounding_box_mask_video'
        #     mask_str = '_mask'
        #     print('use rectangle representation')
        # elif representation == 'gaussian':
        #     folder_path = 'gaussian_mask_video'
        #     mask_str = '_mask'
        #     print('use gaussian representation')
        # else:

        frame_rate = 30
        folder_path = 'original_video'
        print('use original representation')

        self.metadata = metadata[metadata['train_or_test'] == 'test'] if test \
            else metadata[metadata['train_or_test'] == 'train']

        self.img_representation = img_representation

        self.metadata['full_path'] = [
            os.path.join(os.getcwd(),
                         f"critical_classification/dashcam_video/{folder_path}", f'{filename}')
            for filename in self.metadata['path']
        ]

        # Filter metadata based on path existence
        valid_indices = [True if os.path.exists(path) else False for path in self.metadata['full_path']]
        self.metadata = self.metadata[valid_indices]

        self.metadata['duration'] = [get_video_duration_opencv(path) for path in self.metadata['full_path']]
        self.metadata = self.metadata.reset_index()
        self.duration = duration
        self.frame_rate = frame_rate
        self.train_or_test = 'test' if test else 'train'
        self.model_name = model_name
        self.img_size = img_size

        invalid_rows = self.metadata['critical_driving_time'].apply(is_valid_time_format)
        if not invalid_rows.all():
            raise ValueError(f"Invalid time format found in rows with data:\n{self.metadata['path'][~invalid_rows]}")

        print(f'{self.train_or_test} dataset contain: ')
        print(utils.green_text(f"{sum(self.metadata['video_type'] == 'Dashcam')} video contain critical data"))
        print(utils.red_text(f"{sum(self.metadata['video_type'] != 'Dashcam')} video contain non critical data"))

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
                                                              model_name=self.model_name,
                                                              img_representation=self.img_representation,
                                                              img_size=self.img_size)

        if self.model_name == 'YOLOv1':
            video = video.transpose((0, 3, 1, 2))

        return video, label, (self.metadata['full_path'][idx], start_time)


def get_datasets(metadata: pd.DataFrame,
                 img_representation: str,
                 sample_duration: float,
                 model_name: str,
                 img_size: int = 224):
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
            img_representation=img_representation,
            test=train_or_test == 'test',
            model_name=model_name,
            img_size=img_size
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


# def random_short_side_scale(image, min_size=256, max_size=320):
#     h, w = image.shape[:2]
#     short_side = random.randint(min_size, max_size)
#     if h < w:
#         new_h = short_side
#         new_w = int((short_side / h) * w)
#     else:
#         new_w = short_side
#         new_h = int((short_side / w) * h)
#     return cv2.resize(image, (new_w, new_h))


# def random_crop(image, height, width):
#     h, w = image.shape[:2]
#     top = random.randint(0, h - height)
#     left = random.randint(0, w - width)
#     return image[top:top + height, left:left + width]


def normalize(image, mean=None, std=None):
    if mean and std is not None:
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


def dataset_transforms(video_tensor: np.array,
                       train_or_test: str,
                       img_size: int,
                       model_name: str = 'VideoMAE') -> torch.tensor:
    """
    Returns the transforms required for the VIT for training or test datasets
    """
    mean, std = None, None
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
    else:
        height, width = img_size, img_size

    if train_or_test == 'train':
        return train_transform(video_tensor, height, width, mean, std)
    else:
        return test_transform(video_tensor, height, width, mean, std)
