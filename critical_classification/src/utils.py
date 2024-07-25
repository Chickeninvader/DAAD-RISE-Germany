import os
import pickle
import sys
import pathlib
import datetime
import math
import typing

from matplotlib import pyplot as plt


def format_seconds(seconds: int):
    # Create a timedelta object with the given seconds
    time_delta = datetime.timedelta(seconds=seconds)

    # Use the total_seconds() method to get the total number of seconds
    total_seconds = time_delta.total_seconds()

    # Use divmod to get the hours and minutes
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Create the formatted string
    if hours > 0:
        return f"{math.floor(hours)} hour{'s' if hours > 1 else ''}"
    elif minutes > 0:
        return f"{math.floor(minutes)} minute{'s' if minutes > 1 else ''}"
    else:
        return f"{math.floor(seconds)} second{'s' if seconds > 1 else ''}"


# Function to create a directory if it doesn't exist
def create_directory(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created {directory}')


def is_running_in_colab() -> bool:
    """
    Check if the code is running in Google Colab.
    Returns:
        True if running in Google Colab, False otherwise.
    """

    return 'google.colab' in sys.modules


def is_local() -> bool:
    return pathlib.Path(__file__).parent.parent.name == 'PycharmProjects'


def is_debug_mode():
    # Check if the script was launched with the -d or --debug flag
    return is_local() and (any(arg in sys.argv for arg in ['-d', '--debug']) or sys.gettrace() is not None)


def colored_text(color: str):
    index = {'red': 1,
             'green': 2,
             'blue': 4}[color]

    return lambda s: f"\033[9{index}m{s}\033[0m"


def green_text(s: typing.Union[str, float]) -> str:
    return colored_text('green')(s)


def red_text(s: typing.Union[str, float]) -> str:
    return colored_text('red')(s)


def blue_text(s: typing.Union[str, float]) -> str:
    return colored_text('blue')(s)


def format_integer(n):
    if n == 0:
        return "0"

    # Extracting the sign
    sign = "-" if n < 0 else ""
    n = abs(n)

    # Finding 'a' and 'b'
    b = 0
    while n >= 10:
        n /= 10
        b += 1
    a = int(n)

    # Constructing the string representation
    if b == 0:
        return f"{sign}{a}"
    else:
        return f"{sign}{a} * 10^{b}"


def plot_figure_and_save_dict(data_dict, save_fig_path: str, save_dict_path: str, train_or_test: str):
    """
    Plots and saves a graph of multiple values over epochs.

    Args:
    data_dict (dict of list of float): Dictionary containing lists of values to plot, keyed by the result type (e.g., 'train_acc', 'train_loss').
    save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 6))

    for key, values in data_dict.items():
        epochs = list(range(1, len(values) + 1))
        plt.plot(epochs, values, marker='o', linestyle='-', label=key)

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'Metrics over Epochs on {train_or_test}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_fig_path)
    plt.close()

    with open(save_dict_path, 'wb') as file:
        pickle.dump(data_dict, file)


def _format_time(seconds):
    """
    Format a time given in seconds into a string "MM:SS".
    """
    minutes = seconds // 60
    seconds = seconds % 60
    return f'{minutes}:{seconds:02d}'


def _calculate_non_critical_times(critical_times, total_duration):
    """
    Calculate non-critical times based on critical times and total duration.
    """
    critical_ranges = [_parse_time_range(tr) for tr in critical_times.split(', ')]
    non_critical_ranges = []
    last_end = 0
    for start, end in critical_ranges:
        if last_end < start:
            non_critical_ranges.append(f'{_format_time(last_end)}-{_format_time(start)}')
        last_end = end
    if last_end < total_duration:
        non_critical_ranges.append(f'{_format_time(last_end)}-{_format_time(total_duration)}')
    return non_critical_ranges


def _parse_time_range(time_range):
    """
    Parse a time range string into start and end times in seconds.
    """
    start_str, end_str = time_range.split('-')
    start = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start_str.split(':'))))
    end = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end_str.split(':'))))
    return start, end


def add_row_metadata(expanded_metadata: list,
                     dataset_name: str,
                     video_path: str,
                     video_duration: int,
                     critical_times: str):
    if dataset_name == 'Dashcam':
        for time_range in critical_times.split(', '):
            expanded_metadata.append({
                'full_path': video_path,
                'sample_duration': time_range,
                'video_duration': video_duration,
                'label': 1
            })
        # Add remaining times as non-critical
        non_critical_times = _calculate_non_critical_times(critical_times, video_duration)
        for time_range in non_critical_times:
            start_time, end_time = _parse_time_range(time_range)

            # Split non-critical times into 4-second segments
            for start in range(start_time, end_time, 4):
                end = min(start + 4, end_time)
                expanded_metadata.append({
                    'full_path': video_path,
                    'sample_duration': f'{_format_time(start)}-{_format_time(end)}',
                    'video_duration': video_duration,
                    'label': 0
                })
    elif dataset_name == 'Bdd100k':
        # Sample 4s for 40s for all the video
        for start in range(0, video_duration, 4):
            end = min(start + 4, video_duration)
            expanded_metadata.append({
                'full_path': video_path,
                'sample_duration': f'{_format_time(start)}-{_format_time(end)}',
                'video_duration': video_duration,
                'label': 0
            })
    elif dataset_name == 'Carcrash':
        # For the dataset, critical label is used only
        critical_labels = [int(x) for x in critical_times.split(', ')]

        expanded_metadata.append({
            'full_path': video_path,
            'sample_duration': critical_labels,
            'video_duration': video_duration,
            'label': 1
        })
    return expanded_metadata
