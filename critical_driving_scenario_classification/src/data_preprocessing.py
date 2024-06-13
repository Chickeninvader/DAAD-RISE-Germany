import time

import cv2
import numpy as np
import torch


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
            frames.append(frame)
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

if __name__ == '__main__':
    # Example usage
    video_path = "/Users/khoavo2003/PycharmProjects/DAAD-RISE-Germany/youtube_video/video_1.mp4"
    frames_tensor = get_video_frames_as_tensor(video_path, 0, 2)

    print(frames_tensor.shape)  # Output: (number_of_frames, height, width, channels)
