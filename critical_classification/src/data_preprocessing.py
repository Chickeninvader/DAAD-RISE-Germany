import cv2
import os
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.utils.data
import typing
import random
from torch.utils.data import Dataset
from moviepy.editor import VideoFileClip


# random.seed(42)
# np.random.seed(42)


def get_video_frames_as_tensor(video_path,
                               start_time,
                               duration=5,
                               frame_rate=30):
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
    duration_in_ms = 1000 * duration  # Capture 5 seconds (adjustable)

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_in_ms)

    frames = []
    for i in range(duration_in_ms // int(1000 / frame_rate)):
        ret, frame = cap.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)
            print(f'finish frame {i}')
            cv2.imshow(f'frame {i}', frame)
        else:
            raise ValueError("Error: Frame not read!")

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # Convert frames to NumPy array
    frames_np = np.stack(frames, axis=0)
    return frames_np


def get_critical_label(critical_driving_time,
                       chosen_time):
    for start_end_time in critical_driving_time.split(","):
        start_time, end_time = start_end_time.split('-')
        start_time_in_second = sum(x * int(t) for x, t in zip([60, 1], start_time.split(":")))
        end_time_in_second = sum(x * int(t) for x, t in zip([60, 1], end_time.split(":")))
        if start_time_in_second < chosen_time < end_time_in_second:
            return 1

    return 0


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
            os.path.join(os.getcwd(), "dashcam_video/bounding_box_mask_video", filename)
            for filename in self.metadata['path']
        ]

        # self.metadata['duration'] = [VideoFileClip(path).duration for path in self.metadata['full_path']]
        self.transform = transform
        self.duration = duration
        self.frame_rate = frame_rate

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        start_time = random.randint(0, self.metadata['duration'][idx] * 2) / 2.0
        video = get_video_frames_as_tensor(video_path=self.metadata['full_path'][idx],
                                           start_time=start_time,
                                           duration=self.duration,
                                           frame_rate=self.frame_rate)
        critical_time = self.metadata['critical_driving_time'][idx]

        if self.transform:
            video = self.transform(video)

        return video, get_critical_label(critical_time, start_time)


def get_datasets(metadata) -> \
        (typing.Dict[str, torchvision.datasets]):
    """
    Instantiates and returns train and test datasets

    Parameters
    ----------
    """
    datasets = {}

    for train_or_test in ['train', 'test']:
        print(f'get error detector loader for {train_or_test}')
        datasets[train_or_test] = DashcamVideoDataset(
            metadata=metadata,
            transform=None,
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


if __name__ == '__main__':
    # Example usage
    video_path = "/critical_classification/dashcam_video/video_1.mp4"
    frames_tensor = get_video_frames_as_tensor(video_path, 0, 2)

    print(frames_tensor.shape)  # Output: (number_of_frames, height, width, channels)
