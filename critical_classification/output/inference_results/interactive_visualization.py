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
    'Swin3D_experiment_20240724_235746'
]
common_file_name = 'cd2af7d8-a65bb530.pkl'

# Create a dictionary to store the data with labels
data = {
    'Swin3D': folder_names[0] + '/' + common_file_name,
    'Swin3D_past': folder_names[1] + '/' + common_file_name,
    # 'YOLOv1_video': folder_names[1] + '/' + common_file_name,
    # 'YOLOv1_video_no_fc': folder_names[2] + '/' + common_file_name
}


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
        'prediction': prediction_list,
        'label': label
    })

    data_frames.append(df)
    labels.append(label)

# Combine all data frames
combined_df = pd.concat(data_frames, ignore_index=True)

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
        value=[5, 10],
        marks={i: str(i) for i in range(int(combined_df['time'].min()), int(combined_df['time'].max()) + 1, 5)},
        tooltip={"placement": "bottom", "always_visible": True}
    )
])


# Callback to update the plot based on the slider input
@app.callback(
    Output('interactive-plot', 'figure'),
    Input('time-slider', 'value')
)
def update_graph(selected_range):
    filtered_df = combined_df[(combined_df['time'] >= selected_range[0]) & (combined_df['time'] <= selected_range[1])]

    fig = go.Figure()

    for label in labels:
        df_label = filtered_df[filtered_df['label'] == label]
        fig.add_trace(go.Scatter(x=df_label['time'], y=df_label['prediction'], mode='lines', name=label))

    fig.add_hline(y=0.5, line_dash='dash', line_color='red', name='Threshold')

    fig.update_layout(
        title='Predictions Over Time',
        xaxis_title='Time (s)',
        yaxis_title='Prediction',
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        yaxis=dict(range=[0, 1]),
        width=400,
        height=400
    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
