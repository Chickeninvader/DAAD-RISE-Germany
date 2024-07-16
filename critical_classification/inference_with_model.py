import argparse

import cv2
import numpy as np

from critical_classification.src import backbone_pipeline
from critical_classification.config import Config


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
                prediction_list,
                file_name,
                start_time,
                config,
                base_folder: str = 'critical_classification/dashcam_video/temp_video/', ):
    """Prepares and displays a GIF from a video tensor."""

    frames = create_gif(video_tensor)

    # Create a video stream to display frames using OpenCV
    height, width, _ = frames[0].shape
    video_stream = cv2.VideoWriter(f'{base_folder}{file_name}_{start_time}.mp4',
                                   cv2.VideoWriter_fourcc(*"mp4v"),
                                   config.FRAME_RATE,
                                   (width, height))
    for idx, frame in enumerate(frames):
        # since the model is trained with num_frame = 15, only after 15 frames we get the prediction
        if idx >= 15 and prediction_list[idx - 15].item() == 1:
            cv2.putText(frame,
                        text='Critical',
                        org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2)
        video_stream.write(frame)


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning pipeline")
    parser.add_argument('--data_location', type=str, help='Path to the data location',
                        default='critical_classification/dashcam_video/original_video/')
    parser.add_argument('--pretrained_path', type=str, help='Path to model location',
                        default='critical_classification/save_models/'
                                'MYOLOv1_video_lr1e-05_lossBCE_e20_scosine_Aexperiment_20240715_165949.pth')
    args = parser.parse_args()

    config = Config()
    config.data_location = args.data_location
    config.pretrained_path = args.pretrained_path
    config.sample_duration = 4

    fine_tuner, loaders, device = (
        backbone_pipeline.initiate(config)
    )

    fine_tuner.eval()

    for idx, (video_tensor_batch, label, metadata) in enumerate(loaders['train']):
        if idx > 5:
            break
        video_tensor = video_tensor_batch[0]
        file_name, start_time = metadata[0]
        prediction_list = fine_tuner.infer_from_video(video_tensor)
        save_output(video_tensor, prediction_list, file_name, start_time, config)


if __name__ == '__main__':
    main()
