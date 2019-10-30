import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import pandas as pd

# read and initialize data for plots
data = pd.read_csv('../data/results/2019-10-29.csv',index_col=0,date_parser=pd.to_datetime)
data = data.iloc[14:]

# generate figure to plot
fig = make_subplots(
    rows=1, cols=2, subplot_titles=('Aggregate', 'Cluster 1')
)
fig.add_trace(go.Scatter(x=data.index, y=data['aggregate_actual'].values), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index[:10], y=data['aggregate_sarimax'].values[:10]), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['cluster1_actual'].values), row=1, col=2)
fig.add_trace(go.Scatter(x=data.index[:10], y=data['cluster1_sarimax'].values[:10]), row=1, col=2)
fig.update_layout(showlegend=False)


# initialize Dash app
app = dash.Dash()

styleDict = {
    'fontFamily':['Avenir','Arial'],
    'backgroundColor':'#FFFFFF',
    'color': '#111111',
    'textAlign':'center'
}

app.layout = html.Div(style=styleDict, children=[
    html.H1(
        children='University Load Forecasting:',
    ),
    html.Div(
        children="Applications of Deep Learning and Load-Shape Clustering"
    ),
    dcc.Graph(
        id='Graph1',
        # style={'width': 600},
        figure=fig,
        config={
            'displaylogo': False,
            # 'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
            'modeBarButtons': [['pan2d', 'zoom2d']],
            # 'displayModeBar':False
        },
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)

# storing old graph implementation

# dcc.Graph(
#     id='Graph1',
#     style={'width': 600},
#     config={
#         'displaylogo':False,
#         # 'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
#         'modeBarButtons': [['pan2d','zoom2d']],
#         # 'displayModeBar':False
#     },
#     figure=go.Figure(
#         data=[
#             go.Scatter(
#                 x=X_index,
#                 y=y_actual,
#                 name='Actual',
#             ),
#             go.Scatter(
#                 x=X_index,
#                 y=y_sarimax,
#                 name='SARIMAX Forecast',
#             ),
#         ],
#         layout=go.Layout(
#             title='Day Ahead Forecasted Demand',
#             showlegend=True,
#             legend=go.layout.Legend(
#                 x=0.8,
#                 y=0.1
#             ),
#             margin=go.layout.Margin(l=20, r=20, t=40, b=20)
#         )
#     )
# )
