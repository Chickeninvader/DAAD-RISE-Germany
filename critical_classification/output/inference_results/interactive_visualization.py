import pickle

import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd

file_name = '/critical_classification/output/inference_results/Swin3D_experiment_20240724_235746/Rettungsgassenmissbrauch verrücktes Überholen und kaputter LKW DDG Dashcam Germany  284.pkl'

with open(file_name, 'rb') as file:
    loaded_list = pickle.load(file)

# Sample data for testing
current_time_list = loaded_list['current_time_list']
prediction_list = loaded_list['prediction_list']

app = dash.Dash(__name__)

df = pd.DataFrame({
    'time': current_time_list,
    'prediction': prediction_list
})

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['time'], y=df['prediction'], mode='lines', name='Prediction'))
fig.add_hline(y=0.5, line_dash='dash', line_color='red', name='Threshold')

fig.update_layout(
    title=loaded_list['name'],
    xaxis_title='Time (s)',
    yaxis_title='Prediction',
    xaxis=dict(tickmode='linear', tick0=0, dtick=10)
)

app.layout = html.Div([
    dcc.Graph(id='interactive-plot', figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
