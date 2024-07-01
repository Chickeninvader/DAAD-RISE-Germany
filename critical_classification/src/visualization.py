import cv2
from critical_classification.src import backbone_pipeline
from critical_classification import config
from transformers import VideoMAEImageProcessor


def unnormalize_img(img, std, mean):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


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


def save_video(video_tensor, std, mean,
               base_folder: str = 'critical_classification/dashcam_video/temp_video/'):
    """Prepares and displays a GIF from a video tensor."""
    video_tensor = video_tensor[0].permute(1, 0, 2, 3)
    frames = create_gif(video_tensor, std, mean)

    # Create a video stream to display frames using OpenCV
    height, width, _ = frames[0].shape
    video_stream = cv2.VideoWriter(f'{base_folder}nothing.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                   (width, height))
    for frame in frames:
        video_stream.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    model_ckpt = "MCG-NJU/videomae-base"
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    mean = image_processor.image_mean
    std = image_processor.image_std

    fine_tuner, loaders, device = (
        backbone_pipeline.initiate(metadata=config.metadata,
                                   batch_size=1,
                                   model_name=config.model_name,
                                   pretrained_path=config.pretrained_path,
                                   representation=config.representation,
                                   sample_duration=2)
    )

    label = None
    video_tensor = None

    for video_tensor, label, metadata in loaders['train']:
        if label == 1:
            break

    print(f'image has label {label}')
    save_video(video_tensor,
               std=std,
               mean=mean)
