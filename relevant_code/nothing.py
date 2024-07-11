import os
from moviepy.editor import VideoFileClip


def convert_mov_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        if file.endswith(".mov"):
            # Generate input and output file paths
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace(".mov", ".mp4"))

            # Check if output file already exists
            if os.path.exists(output_path):
                print(f"Skipping {input_path} as {output_path} already exists.")
                continue

            # Convert .mov to .mp4
            convert_mov_to_mp4(input_path, output_path)


def convert_mov_to_mp4(input_path, output_path):
    # Load the video file
    clip = VideoFileClip(input_path)

    # Export the video in .mp4 format
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # Close the clip
    clip.close()


def convert_mov_to_mp4(input_path, output_path):
    # Load the video file
    clip = VideoFileClip(input_path)

    # Export the video in .mp4 format
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # Close the clip
    clip.close()


if __name__ == "__main__":
    # Specify input and output folders
    input_folder = 'critical_classification/dashcam_video/original_video/'
    output_folder = 'critical_classification/dashcam_video/original_video/'

    # Convert all .mov files in input_folder to .mp4 in output_folder
    convert_mov_files(input_folder, output_folder)
