import cv2
import os
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.utils.data
import typing
import random
from torchvision import transforms
from torch.utils.data import Dataset


# random.seed(42)
# np.random.seed(42)


def get_video_frames_as_tensor(video_path,
                               start_time,
                               duration=5,
                               frame_rate=30,
                               transform=None):
    """
    This function reads a video at a specific time and captures frames for a
    given duration, returning them as a NumPy array.

    Args:
      video_path: Path to the video file.
      start_time: Time in seconds from where to start capturing frames.
      duration: Duration of video
      frame_rate: Optional frame rate of the video (default 30fps).

    Returns:
      A NumPy array with captured video frames stacked together.

    Raises:
      ValueError: If video cannot be opened or frame is not read successfully.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file!")

    start_time_in_ms = start_time * 1000
    duration_in_ms = int(1000 * duration)  # Capture 5 seconds (adjustable)

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_in_ms)

    frames = []
    for i in range(duration_in_ms // int(1000 / frame_rate)):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (448, 448))
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(transform(frame))
        else:
            raise ValueError("Error: Frame not read!")

    cap.release()
    cv2.waitKey(1)

    # Convert frames to Torch.tensor
    frames_tensor = torch.from_numpy(np.stack(frames, axis=1)).float()
    return frames_tensor


def get_critical_label(critical_driving_time,
                       chosen_time):
    for start_end_time in critical_driving_time.split(","):
        start_time, end_time = start_end_time.split('-')
        start_time_in_second = sum(x * int(t) for x, t in zip([60, 1], start_time.split(":")))
        end_time_in_second = sum(x * int(t) for x, t in zip([60, 1], end_time.split(":")))
        if start_time_in_second < chosen_time < end_time_in_second:
            return 1

    return 0


def get_video_duration_opencv(video_path):
    """Gets the duration of a video using OpenCV."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file!")

    # Get frame rate
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Get total number of frames (may not be accurate for all video formats)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calculate duration in seconds (assuming frame count is accurate)
    duration_in_seconds = num_frames / frame_rate

    cap.release()
    return duration_in_seconds


class DashcamVideoDataset(Dataset):
    def __init__(self,
                 metadata: pd.DataFrame,
                 duration,
                 frame_rate,
                 test: bool,
                 transform=None, ):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = metadata[metadata['train_or_test'] == 'test'] if test \
            else metadata[metadata['train_or_test'] == 'train']

        self.metadata['full_path'] = [
            os.path.join(os.getcwd(), "critical_classification/dashcam_video/bounding_box_mask_video", f'{filename[:-4]}_mask.mp4')
            for filename in self.metadata['path']
        ]

        self.metadata['duration'] = [get_video_duration_opencv(path) for path in self.metadata['full_path']]
        self.transform = transform
        self.duration = duration
        self.frame_rate = frame_rate

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        start_time = random.randint(0, int(self.metadata['duration'][idx] - 1) * 2) / 2.0
        video = get_video_frames_as_tensor(video_path=self.metadata['full_path'][idx],
                                           start_time=start_time,
                                           duration=self.duration,
                                           frame_rate=self.frame_rate,
                                           transform=self.transform)
        critical_time = self.metadata['critical_driving_time'][idx]

        return video, get_critical_label(critical_time, start_time), start_time


def get_datasets(metadata):
    """
    Instantiates and returns train and test datasets

    Parameters
    ----------
    """
    datasets = {}

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for train_or_test in ['train', 'test']:
        print(f'get error detector loader for {train_or_test}')
        datasets[train_or_test] = DashcamVideoDataset(
            metadata=metadata,
            transform=transform,
            duration=0.5,
            frame_rate=30,
            test=train_or_test == 'test'
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
            num_workers=4,
        )
    return loaders
