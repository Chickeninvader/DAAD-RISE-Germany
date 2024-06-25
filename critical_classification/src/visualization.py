# Visualization now work only for
import imageio
from IPython.display import Image

from critical_classification.src import backbone_pipeline
from critical_classification import config
from transformers import VideoMAEImageProcessor


def unnormalize_img(img, std, mean):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def create_gif(video_tensor,
               std,
               mean,
               duration,
               filename: str = "critical_classification/dashcam_video/temp_video/sample.gif"):
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
    kargs = {"fps": duration / video_len, "duration": 2}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename


def display_gif(video_tensor,
                std,
                mean,
                duration):
    """Prepares and displays a GIF from a video tensor."""
    video_tensor = video_tensor[0].permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, std, mean, duration)
    return Image(filename=gif_filename)


if __name__ == '__main__':
    model_ckpt = "MCG-NJU/videomae-base"
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    mean = image_processor.image_mean
    std = image_processor.image_std
    sample_duration = 2

    fine_tuner, loaders, device = (
        backbone_pipeline.initiate(metadata=config.metadata,
                                   batch_size=1,
                                   model_name=config.model_name,
                                   pretrained_path=config.pretrained_path,
                                   representation=config.representation,
                                   sample_duration=sample_duration)
    )
    for video_tensor, label, metadata in loaders['train']:
        if label == 1:
            break

    print(f'image has label {label}')
    display_gif(video_tensor,
                std=std,
                mean=mean,
                duration=sample_duration)
