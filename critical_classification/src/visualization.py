import argparse

import cv2

from critical_classification.src import backbone_pipeline
from critical_classification.config import Config
import numpy as np


def unnormalize_img(img, std, mean):
    """Un-normalizes the image pixels."""
    if std is None or mean is None:
        return np.round(img * 255).astype('uint8')
    img = (img * std) + mean
    img = (img * 255)
    return np.round(img).astype('uint8')


def create_gif(video_tensor, std, mean):
    """Prepares a GIF from a video tensor.

    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    video_len = video_tensor.shape[0]
    for video_frame_idx in range(video_len):
        frame_unnormalized = unnormalize_img(video_tensor[video_frame_idx].permute(1, 2, 0).numpy(),
                                             std=std,
                                             mean=mean)
        frames.append(frame_unnormalized)
    return frames


def save_output(data_tensor, std=None, mean=None,
                base_folder: str = 'critical_classification/dashcam_video/temp_video/',
                idx: int = 0):
    """Prepares and displays a GIF from a video tensor."""
    data_tensor = data_tensor[0]
    if data_tensor.shape[1] != 3:
        data_tensor = data_tensor.permute((0, 3, 1, 2))
    frames = create_gif(data_tensor, std, mean)

    if data_tensor.shape[0] == 1:
        cv2.imwrite(f'{base_folder}nothing.jpg', frames[0])
        return
    # Create a video stream to display frames using OpenCV
    height, width, _ = frames[0].shape
    video_stream = cv2.VideoWriter(f'{base_folder}nothing_{idx}.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                   (width, height))
    for frame in frames:
        # video_stream.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_stream.write(frame)


if __name__ == '__main__':
    # model_ckpt = "MCG-NJU/videomae-base"
    # image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    # mean = image_processor.image_mean
    # std = image_processor.image_std

    parser = argparse.ArgumentParser(description="Fine-tuning pipeline")
    parser.add_argument('--data_location', type=str, help='Path to the data location',
                        default='critical_classification/dashcam_video/original_video/')
    # Add more arguments as needed

    args = parser.parse_args()
    config = Config()
    config.data_location = args.data_location
    config.model_name = None

    fine_tuner, loaders, device = (
        backbone_pipeline.initiate(config)
    )

    label = None
    video_tensor = None
    metadata = None

    for idx, (video_tensor, label, metadata) in enumerate(loaders['train']):
        if metadata[0].endswith('.mov'):
            print(f'image has label {label}')
            print(metadata)
            save_output(video_tensor, idx)
