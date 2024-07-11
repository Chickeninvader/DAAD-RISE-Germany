import cv2


def read_and_append_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file '{video_path}'")
        return

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            frame_count += 1
            # Display frame dimensions (height, width, channels)
            print(f"Frame {frame_count}: {frame.shape}")
        else:
            break

    cap.release()

    print(f"Total frames read: {frame_count}")

    return frames


if __name__ == "__main__":
    # Replace with the path to your .mov file
    video_path = 'critical_classification/dashcam_video/original_video/cc7b0a4d-e8dfd84a.mov'

    frames = read_and_append_frames(video_path)

    # Optionally, display or process frames further
    for i, frame in enumerate(frames):
        cv2.imshow(f"Frame {i}", frame)
        cv2.waitKey(100)  # Adjust delay as needed
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
