import pickle
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

# Define folder names and common file name
folder_names = [
    'Swin3D_experiment_20240802_043855',
    'YOLOv1_video_experiment_20240802_135545_no_fc'
]
common_file_name = 'Vollbremsungen RTW blockieren Anhänger schaukelt sich auf und Zufälle DDG Dashcam Germany  285.pkl'

# Create a dictionary to store the data with labels
data = {
    'Swin3D': folder_names[0] + '/' + common_file_name,
    'YOLOv1_video': folder_names[1] + '/' + common_file_name
}

# Ground truth time range (when ground truth is 1)
ground_truth_time_range = (43, 44)


# Function to apply sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Load data from all files
data_frames = []
labels = []

for label, file_path in data.items():
    with open(file_path, 'rb') as file:
        loaded_list = pickle.load(file)

    current_time_list = loaded_list['current_time_list']
    prediction_list = sigmoid(np.array(loaded_list['prediction_list']))

    df = pd.DataFrame({
        'time': current_time_list,
        # 'time': [item * 3 for item in current_time_list],
        'prediction': prediction_list,
        'label': label
    })

    data_frames.append(df)
    labels.append(label)

# Combine all data frames
combined_df = pd.concat(data_frames, ignore_index=True)

# Calculate the total duration and set initial range
total_duration = combined_df['time'].max()
initial_range = [0, min(10, total_duration // 10)]  # Adjust initial range as 10% of the video or up to 10 seconds
dtick = max(1, total_duration // 20)  # Adjust dtick dynamically
marks = int(total_duration // 20) if int(total_duration // 20) != 0 else 1

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    dcc.Graph(id='interactive-plot'),
    dcc.RangeSlider(
        id='time-slider',
        min=combined_df['time'].min(),
        max=combined_df['time'].max(),
        step=1,
        marks={i: str(i) for i in range(int(combined_df['time'].min()), int(combined_df['time'].max()) + 1, marks)},
        tooltip={"placement": "bottom", "always_visible": True}
    )
])

# Create the ground truth data series
combined_df['ground_truth'] = combined_df['time'].apply(
    lambda t: 1 if ground_truth_time_range[0] <= t <= ground_truth_time_range[1] else 0
)


# Callback to update the plot based on the slider input
@app.callback(
    Output('interactive-plot', 'figure'),
    Input('time-slider', 'value')
)
def update_graph(selected_range):
    filtered_df = filtered_df = combined_df[(combined_df['time'] >= selected_range[0]) & (combined_df['time'] <= selected_range[1])]

    fig = go.Figure()

    for label in labels:
        df_label = filtered_df[filtered_df['label'] == label]
        fig.add_trace(go.Scatter(x=df_label['time'], y=df_label['prediction'], mode='lines', name=label))

    # Add a horizontal line for the prediction threshold
    fig.add_hline(y=0.5, line_dash='dash', line_color='red', name='Threshold')

    # Add the ground truth line
    fig.add_trace(go.Scatter(
        x=filtered_df[filtered_df['label'] == labels[0]]['time'],
        y=filtered_df[filtered_df['label'] == labels[0]]['ground_truth'],
        mode='lines',
        name='Ground Truth',
    ))

    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Prediction score',
        xaxis=dict(tickmode='linear', tick0=0, dtick=dtick),
        yaxis=dict(range=[0, 1]),
        width=800,
        height=400
    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
