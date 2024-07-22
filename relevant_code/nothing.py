import cv2


def get_video_frame_rate(video_path: str) -> float:
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file!")

    # Get the frame rate
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    cap.release()

    return frame_rate


# Example usage
video_path = '/data/nvo/Car_crash_video/000062.mp4'
frame_rate = get_video_frame_rate(video_path)
print(f"Frame rate of the video is: {frame_rate} FPS")
