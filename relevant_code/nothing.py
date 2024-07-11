from moviepy.editor import VideoFileClip

def convert_mov_to_mp4(input_path, output_path):
    # Load the video file
    clip = VideoFileClip(input_path)

    # Export the video in .mp4 format
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # Close the clip
    clip.close()

if __name__ == "__main__":
    # Replace with your input and output file paths
    input_path = 'critical_classification/dashcam_video/original_video/cc7b0a4d-e8dfd84a.mov'
    output_path = 'critical_classification/dashcam_video/original_video/cc7b0a4d-e8dfd84a.mp4'

    # Convert .mov to .mp4
    convert_mov_to_mp4(input_path, output_path)
