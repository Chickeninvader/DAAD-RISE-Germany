import os
import pickle
import sys
from collections import deque

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())

from critical_classification.src import backbone_pipeline, data_preprocessing, utils
from critical_classification.config import Config, GetConfig
from critical_classification.src.data_preprocessing import get_video_duration_opencv


class FullVideoDataset:
    def __init__(self, config: Config):
        self.metadata = config.metadata
        self.img_representation = config.img_representation
        self.model_name = config.model_name
        self.img_size = config.img_size
        self.data_location = config.data_location
        self.dataset_name = config.dataset_name
        self.frame_rate = 30
        self.config = config

        self.metadata = self.metadata[self.metadata['train_or_test'] == 'infer']

        base_path_mapping = {
            'Dashcam': os.path.join(config.data_location, 'Dashcam_video/'),
            'Carcrash': os.path.join(config.data_location, 'Car_crash_video/'),
            'Bdd100k': os.path.join(config.data_location, 'Bdd100k_video/')
        }

        self.metadata['full_path'] = [
            os.path.join(base_path_mapping.get(video_type, ''), f'{filename}')
            for video_type, filename in zip(self.metadata['video_type'], self.metadata['path'])
        ]

        # Filter metadata based on path existence
        valid_indices = [True if os.path.exists(path) else False for path in self.metadata['full_path']]
        self.metadata = self.metadata[valid_indices]

        self.metadata['duration'] = [int(get_video_duration_opencv(path)) for path in self.metadata['full_path']]
        self.metadata = self.metadata.reset_index(drop=True)

        print(f'Infer dataset contain: ')
        print(utils.blue_text(f"{len(self.metadata)} videos contain critical data"))

    def __len__(self):
        return len(self.metadata)

    def infer_and_save_result(self,
                              fine_tuner: torch.nn.Module,
                              idx: int,
                              config: Config,
                              device: torch.device,
                              base_folder: str = 'critical_classification/output/inference_results',
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

        frames = deque(maxlen=config.image_batch_size)
        ret = True
        frame_idx = 0
        prediction_list = []
        current_time_list = []

        while ret:
            frame_idx += 1
            ret, frame = cap.read()
            if not ret:
                break
            frame = data_preprocessing.dataset_transforms(video_array=torch.Tensor(frame).permute(2, 0, 1),
                                                          train_or_test='test',
                                                          img_size=self.img_size,
                                                          model_name=self.model_name)
            frames.append(frame)
            pbar.update(1)  # Update the progress bar

            if len(frames) != config.image_batch_size or frame_idx % 5 != 0:
                continue

            if config.model_name == 'YOLOv1_video':
                video_tensor_frame = torch.tensor(np.stack(frames, axis=0))
            elif config.model_name == 'Swin3D':
                video_tensor_frame = torch.tensor(np.expand_dims(np.stack(frames, axis=0), axis=0))
            else:
                raise NotImplementedError()
            with torch.no_grad():
                prediction_list.append(float(fine_tuner(video_tensor_frame.to(device))))

            current_time_list.append(frame_idx / 30)

        cap.release()
        pbar.close()  # Close the progress bar

        # Construct the full path
        directory = f'{base_folder}{config.model_name}_{config.additional_saving_info}'
        file_path = f'{directory}/{str(file_name[:-4])}.pkl'

        # Create the directory if it does not exist
        os.makedirs(directory, exist_ok=True)

        temp_dict = {
            'name': f'{str(file_name[:-4])}',
            'current_time_list': current_time_list,
            'prediction_list': prediction_list
        }

        with open(file_path, 'wb') as file:
            pickle.dump(temp_dict, file)


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
                prediction_list=None,
                base_folder: str = 'critical_classification/critical_dataset/temp_video/', ):
    """Prepares and displays a GIF from a video tensor."""
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    frames = create_gif(video_tensor)

    # Create a video stream to display frames using OpenCV
    height, width, _ = frames[0].shape
    save_video_file_name = f'{base_folder}{str(file_name[:-4])}_{int(start_time)}.mp4'
    video_stream = cv2.VideoWriter(save_video_file_name,
                                   cv2.VideoWriter_fourcc(*"mp4v"),
                                   30,
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
    config = GetConfig()
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
                                                idx=idx,
                                                config=config,
                                                device=device)
        return

    positive_video_counts = 0
    negative_video_counts = 0

    for idx, (video_tensor_batch, label, metadata) in enumerate(loaders['train']):
        video_tensor = video_tensor_batch[0]
        full_path, start_time = metadata
        full_path = full_path[0]
        file_name = os.path.basename(full_path)
        num_frame = video_tensor.shape[0]

        if config.model_name is not None:
            prediction_list = inference_with_model(num_frame=num_frame,
                                                   video_tensor=video_tensor,
                                                   fine_tuner=fine_tuner,
                                                   device=device)

            print(f'{file_name} has prediction: {prediction_list}')
            save_output(video_tensor, file_name, start_time, prediction_list)
            del video_tensor, video_tensor_batch, prediction_list
            continue

        # if there is no model, simply save video
        if int(label) == 0:
            negative_video_counts += 1
        else:
            positive_video_counts += 1

            # Check if we have enough samples to stop
        if positive_video_counts >= 5 and negative_video_counts >= 5:
            break

            # Skip further processing if counts are already sufficient
        if (int(label) == 0 and negative_video_counts > 5) or (int(label) == 1 and positive_video_counts > 5):
            continue

        print(f'{file_name} has ground truth: {label}')
        save_output(video_tensor, file_name, start_time, config)


if __name__ == '__main__':
    main()
