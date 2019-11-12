import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import pandas as pd
import datetime as dt

# read and initialize data for plots
date = dt.datetime.now().date()
time = dt.datetime.now().hour

results = pd.read_csv('../data/forecasts/'+str(date)+'.csv', index_col=0, header=[0,1], date_parser=pd.to_datetime)
data = results.iloc[14:]

# generate figure to plot
fig = make_subplots(rows=1, cols=2, subplot_titles=('Aggregate', 'Cluster 1'))

fig.add_trace(go.Scatter(x=data.index[:time],
                         y=data['aggregate','actual'].values[:time],
                         mode='lines',hoverinfo='x+y'), row=1, col=1)

fig.add_trace(go.Scatter(x=data.index,
                         y=data['aggregate','sarimax'].values,
                         mode='lines',hoverinfo='x+y'), row=1, col=1)

fig.add_trace(go.Scatter(x=data.index[:time],
                         y=data['cluster1','actual'].values[:time],
                         mode='lines',hoverinfo='x+y'), row=1, col=2)

fig.add_trace(go.Scatter(x=data.index,
                         y=data['cluster1','sarimax'].values,
                         mode='lines',hoverinfo='x+y'), row=1, col=2)

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
            'modeBarButtons': [['pan2d', 'zoom2d']],
            # 'displayModeBar':False
        },
    ),
    dcc.Graph(
        id='example-graph-2',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)