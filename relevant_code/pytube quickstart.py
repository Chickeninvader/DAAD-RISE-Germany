from pytube import YouTube
import os

# yt = YouTube('http://youtube.com/watch?v=2lAe1cqCOXo')
# yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
def download_videos_from_file(file_path, output_dir):
    """Downloads YouTube videos from URLs in a text file and names them as video_{number}.mp4.

    Args:
        file_path (str): Path to the text file containing YouTube video URLs.
        output_dir (str): Directory to save the downloaded videos.
    """
    # Resolve the output directory path relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
    output_dir = os.path.join(script_dir[:-3], output_dir)  # Join with relative path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_number = 1

    with open(file_path, 'r') as file:
        for url in file.readlines():
            url = url.strip()  # Remove any leading/trailing whitespace
            filename = f"video_{video_number}.mp4"

            if os.path.exists(os.path.join(output_dir, filename)):
                print(f"Skipping download (file already exists): {url}")
                video_number += 1
                continue
            try:
                yt = YouTube(url)
                stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                stream.download(output_path=output_dir,
                                filename=filename)
                print(f"Downloaded video {video_number}: {url}")
                video_number += 1
            except Exception as e:
                print(f"Error downloading video from {url}: {e}")


if __name__ == '__main__':
    file_path = 'critical_driving_scenario_video_urls.txt'  # Replace with the actual path to your file
    output_dir = 'youtube_video'  # Replace with your desired output directory

    download_videos_from_file(file_path, output_dir)
