import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd

# Sample data for testing
current_time_list = [i for i in range(660)]
prediction_list = [0.5 + 0.1 * (-1) ** (i // 50) * (i % 50) / 50 for i in range(660)]
ground_truth = [0.5 + 0.1 * 1 ** (i // 50) * (i % 50) / 50 for i in range(660)]

app = dash.Dash(__name__)

df = pd.DataFrame({
    'time': current_time_list,
    'prediction': prediction_list,
    'ground_truth': ground_truth
})

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['time'], y=df['prediction'], mode='lines', name='Prediction'))
fig.add_trace(go.Scatter(x=df['time'], y=df['ground_truth'], mode='lines', name='Ground Truth'))
fig.add_hline(y=0.5, line_dash='dash', line_color='red', name='Threshold')

fig.update_layout(
    title='Critical prediction over time',
    xaxis_title='Time (s)',
    yaxis_title='Prediction',
    xaxis=dict(tickmode='linear', tick0=0, dtick=60),
    width=800,  # Specify the width of the figure
    height=400  # Specify the height of the figure
)

app.layout = html.Div([
    dcc.Graph(id='interactive-plot', figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
