import numpy as np
import pandas as pd
import datetime as dt

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from loadforecast.src.utils import MAPE, LoadShapePlot

# read and initialize data for plots
date = dt.datetime.now().date()
time = dt.datetime.now().hour

location = 'arizona'

results = pd.read_csv('data/forecasts/'+location+'/'+str(date)+'.csv', index_col=0, header=[0,1], date_parser=pd.to_datetime)
data = results.iloc[14:]/1000

# get list of clusters and list of models from column names
clusters = list(data.columns.get_level_values(0).unique())
models = list(data.columns.get_level_values(1).unique())
models.remove('actual')

# calculate and store mapes for all forecasts
mapes = {}
for cluster in clusters:
    mapes[cluster] = {}
    for model in models:
        mapes[cluster][model] = np.round(MAPE(data[cluster]['actual'],data[cluster][model]),2)

# create a figure for each actual cluster
clusters.remove('aggregate')
clusters.remove('clustersum')
clusterDivs = {}
for cluster in clusters:
    clusterDivs[cluster] = html.Div(children=[
        dcc.Graph(
            id=cluster+'Graph',
            config={
                'displaylogo': False,
                'showTips': False,
                'scrollZoom': False,
                'displayModeBar':False
            },
            figure = go.Figure(
                data=[
                    LoadShapePlot('Actual', data[cluster,'actual'].iloc[:time]),
                    LoadShapePlot('SARIMA', data[cluster,'sarimax']),
                    LoadShapePlot('MLP', data[cluster,'mlp']),
                ],
                layout={
                    'margin':{'t':40,'b':40,'l':40,'r':40},
                    'yaxis':{
                        # 'title':'Electricity Load [MW]',
                        'ticks':'outside',
                        'ticksuffix':' MW',
                        'fixedrange':True,
                        'hoverformat':'.2f',
                    },
                    'xaxis':{
                        'ticks':'outside',
                        'nticks':8,
                        'fixedrange':True,
                        'tickformat':'%-I %p',
                    },
                    'legend':{'x':0.78,'y':0.05},
                }
            )
        ),
        dcc.Markdown(
            id=cluster+'Markdown',
            style={'textAlign':'center'},
            children=f"""
            SARIMAX Error: {mapes[cluster]['sarimax']}%
            MLP Error: {mapes[cluster]['mlp']}%
            """
        )],
        style={'width':'50%', 'display':'inline-block', 'padding-top':20, 'vertical-align':'top'},
    )


# generate primary (aggregate load) figure
aggfig = go.Figure(
    data=[
        LoadShapePlot('Actual', data['aggregate','actual'].iloc[:time]),
        LoadShapePlot('Aggregate SARIMAX', data['aggregate','sarimax']),
        LoadShapePlot('Aggregate MLP', data['aggregate','mlp'])
    ],
    layout={
        'margin':{'t':20,'b':20,'l':40,'r':30},
        'yaxis':{'title':'Electricity Load [MW]','ticks':'outside'},
        'xaxis':{'ticks':'outside'},
        'legend':{'x':0.72,'y':0.05}
    }
)

# text to accompany aggregate plot
aggtext = f"""
The plot shows the aggregate load of all buildings on the University of Arizona campus.
The legend describes each series.
Here are the mean absolute percentage error (MAPE) for each forecasting strategy:
* Aggregate SARIMAX: {mapes['aggregate']['sarimax']}
* Aggregate MLP: {mapes['aggregate']['mlp']}
* Cluster Sum SARIMAX: {mapes['clustersum']['sarimax']}
* Cluster Sum MLP: {mapes['clustersum']['mlp']}
"""

# initialize Dash app and include external css
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div(children=[
    html.H1(
        children='University Load Forecasting:',
        style={'textAlign':'center'}
    ),
    html.Div(
        children="Applications of Deep Learning and Load-Shape Clustering",
        style={'textAlign':'center','padding-bottom':20}
    ),
    html.Div(children=[
        dcc.Graph(
            id='AggGraph',
            figure=aggfig,
            style={'width':'60%', 'height':'40%', 'display':'inline-block'},
            config={
                'displaylogo': False,
                'showTips': False,
                'scrollZoom': False,
                'displayModeBar':False
            },
        ),
        dcc.Markdown(
            id='AggText',
            children=aggtext,
            style={'width':'40%', 'display':'inline-block', 'padding-top':20, 'vertical-align':'top'}
        )
    ]),

    html.Div(
        children=[clusterDivs[cluster] for cluster in clusters]
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
