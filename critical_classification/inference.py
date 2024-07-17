import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

sys.path.append(os.getcwd())

from critical_classification.src import backbone_pipeline, data_preprocessing, utils
from critical_classification.config import Config


class FullVideoDataset:
    def __init__(self, config):
        metadata = config.metadata
        img_representation = config.img_representation
        duration = config.sample_duration
        model_name = config.model_name
        img_size = config.img_size
        data_location = config.data_location
        frame_rate = config.FRAME_RATE
        folder_path = ''
        print('use original representation')

        self.metadata = metadata[metadata['train_or_test'] == 'infer']
        self.img_representation = img_representation

        self.metadata['full_path'] = [
            os.path.join(os.getcwd(), f"{data_location}{folder_path}", f'{filename}')
            for filename in self.metadata['path']
        ]

        # Filter metadata based on path existence
        valid_indices = [i for i, path in enumerate(self.metadata['full_path']) if os.path.exists(path)]
        self.metadata = self.metadata.iloc[valid_indices]
        self.metadata['duration'] = [data_preprocessing.get_video_duration_opencv(path)
                                     for path in self.metadata['full_path']]

        self.metadata = self.metadata.reset_index(drop=True)
        self.duration = duration
        self.frame_rate = frame_rate
        self.model_name = model_name
        self.img_size = img_size

        print(f'Infer dataset contain: ')
        print(utils.blue_text(f"{len(self.metadata)} videos contain critical data"))

        self.current_idx = 0
        self.cap = None

    def __len__(self):
        return len(self.metadata)

    def infer_and_save_result(self,
                              fine_tuner: torch.nn.Module,
                              idx: int,
                              config: Config,
                              device: torch.device,
                              base_folder: str = 'critical_classification/output/',
                              ):
        video_path = self.metadata['full_path'][idx]
        file_name = self.metadata['path'][idx]

        print(f'do inference on {video_path}')

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Error opening video file!")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
        pbar = tqdm(total=total_frames, desc='Processing Video', unit='frame')  # Initialize progress bar

        cap.set(cv2.CAP_PROP_POS_MSEC, 0)

        frames = deque(maxlen=15)
        ret = True
        frame_idx = 0
        prediction_list = []
        current_time_list = []

        while ret:
            frame_idx += 1
            ret, frame = cap.read()
            if not ret:
                break
            frame = data_preprocessing.dataset_transforms(video_array=frame,
                                                          train_or_test='test',
                                                          img_size=self.img_size,
                                                          model_name=self.model_name)
            frames.append(frame)

            if len(frames) != 15 or frame_idx % 4 != 0:
                continue

            video_tensor_frame = torch.tensor(np.stack(frames, axis=0))

            with torch.no_grad():
                prediction_list.append(float(fine_tuner(video_tensor_frame.to(device))))

            current_time_list.append(frame_idx / config.FRAME_RATE)
            pbar.update(1)  # Update the progress bar

        cap.release()
        pbar.close()  # Close the progress bar

        # Plot and save the figure
        plt.figure()
        plt.plot(current_time_list, prediction_list)
        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.title('Critical prediction over time')
        plt.xlabel('Time (s)')
        plt.ylabel('Prediction')
        plt.savefig(f'{base_folder}{str(file_name[:-4])}_{config.additional_saving_info}.png')


def unnormalize_img(img):
    """Un-normalizes the image pixels."""
    return np.round(img * 255).astype('uint8')


def create_gif(video_tensor):
    """Prepares a GIF from a video tensor.

    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    video_len = video_tensor.shape[0]
    for video_frame_idx in range(video_len):
        frame_unnormalized = unnormalize_img(video_tensor[video_frame_idx].permute(1, 2, 0).numpy())
        frames.append(frame_unnormalized)
    return frames


def save_output(video_tensor,
                file_name,
                start_time,
                config,
                prediction_list=None,
                base_folder: str = 'critical_classification/dashcam_video/temp_video/', ):
    """Prepares and displays a GIF from a video tensor."""
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    frames = create_gif(video_tensor)

    # Create a video stream to display frames using OpenCV
    height, width, _ = frames[0].shape
    save_video_file_name = f'{base_folder}{str(file_name[:-4])}_{int(start_time)}.mp4'
    video_stream = cv2.VideoWriter(save_video_file_name,
                                   cv2.VideoWriter_fourcc(*"mp4v"),
                                   config.FRAME_RATE,
                                   (width, height))
    for idx, frame in enumerate(frames):
        # since the model is trained with num_frame = 15, only after 15 frames we get the prediction
        frame = frame.copy()
        if prediction_list is not None:
            cv2.putText(frame,
                        text='Critical' if idx >= 15 and prediction_list[idx - 15] == 1 else 'Non critical',
                        org=(100, 100),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(0, 0, 255) if idx >= 15 and prediction_list[idx - 15] == 1 else (0, 255, 0),
                        thickness=2)
        video_stream.write(frame)


def inference_with_model(num_frame,
                         video_tensor,
                         fine_tuner,
                         device):
    prediction_list = []
    for video_tensor_frame_idx in range(num_frame - 15):
        print(f'frame {video_tensor_frame_idx}')
        with torch.no_grad():
            video_tensor_frame = video_tensor[video_tensor_frame_idx:video_tensor_frame_idx + 15].to(device)
            prediction_list.append(0 if float(fine_tuner(video_tensor_frame)) < 0.5 else 1)

            del video_tensor_frame
    return prediction_list


def main():
    parser = argparse.ArgumentParser(description="Inference pipeline")
    parser.add_argument('--data_location', type=str, help='Path to the data location',
                        default='critical_classification/dashcam_video/original_video/')
    parser.add_argument('--pretrained_path', type=str, help='Path to model location',
                        default=None)
    parser.add_argument('--all_frames', action='store_true', help='Do inference for infer video')

    args = parser.parse_args()

    config = Config()
    config.data_location = args.data_location
    config.pretrained_path = args.pretrained_path
    config.infer_all_video = args.all_frames
    config.sample_duration = 4
    config.print_config()

    fine_tuner, loaders, device = (
        backbone_pipeline.initiate(config)
    )
    if config.model_name is not None:
        fine_tuner.to(device)
        fine_tuner.eval()

    if config.infer_all_video:
        video_dataset = FullVideoDataset(config)
        for idx in range(len(video_dataset)):
            video_dataset.infer_and_save_result(fine_tuner=fine_tuner,
                                                idx=3,
                                                config=config,
                                                device=device)
            break
        return

    for idx, (video_tensor_batch, label, metadata) in enumerate(loaders['test']):
        video_tensor = video_tensor_batch[0]
        file_name, start_time = metadata
        file_name = file_name[0]
        num_frame = video_tensor.shape[0]
        print(f'start doing inference for {file_name}')

        if config.model_name is not None:
            prediction_list = inference_with_model(num_frame=num_frame,
                                                   video_tensor=video_tensor,
                                                   fine_tuner=fine_tuner,
                                                   device=device)

            print(f'{file_name} has prediction: {prediction_list}')
            save_output(video_tensor, file_name, start_time, config, prediction_list)
            del video_tensor, video_tensor_batch, prediction_list

        else:
            save_output(video_tensor, file_name, start_time, config)


if __name__ == '__main__':
    main()
