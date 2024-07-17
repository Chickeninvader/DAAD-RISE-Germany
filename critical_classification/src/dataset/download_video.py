import os
import sys

from pytubefix import YouTube
import pandas as pd

sys.path.append(os.getcwd())


def download_videos_from_file(file_path, output_dir, df):
    """Downloads YouTube videos from URLs in a text file and names them as video_{number}.mp4.

    Args:
        file_path (str): Path to the text file containing YouTube video URLs.
        output_dir (str): Directory to save the downloaded videos.
    """
    # Resolve the output directory path relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory\

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_number = 0

    with open(file_path, 'r') as file:
        for url in file.readlines():
            url = url.strip()  # Remove any leading/trailing whitespace
            try:
                video_number += 1
                yt = YouTube(url)
                stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                downloaded_path = stream.get_file_path(output_path=output_dir)
                filename = os.path.basename(downloaded_path)
                # Assign the path to the last row using df.loc
                if not df['path'].isin([filename]).any():
                    print(f'add {filename} to metadata')
                    df.at[df.shape[0] + video_number, 'path'] = filename

                if os.path.exists(downloaded_path):
                    print(f'video {url} already exist')
                    continue
                print(f"Downloaded video: {url}")
                stream.download(output_path=output_dir)

            except Exception as e:
                print(f"Error downloading video from {url}: {e}")
    return df


if __name__ == '__main__':
    file_path = 'critical_classification/src/dataset/critical_driving_scenario_video_urls.txt'
    output_dir = '/data/nvo/original_video'  # Replace with your desired output directory
    df = pd.read_excel('critical_classification/dashcam_video/metadata.xlsx')

    update_df = download_videos_from_file(file_path, output_dir, df)
    # update_df.to_excel('critical_classification/dashcam_video/metadata.xlsx', index=False)
