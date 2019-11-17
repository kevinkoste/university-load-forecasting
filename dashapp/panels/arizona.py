import numpy as np
import pandas as pd
import datetime as dt

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from loadforecast.src.utils.plot_utils import MAPE, LoadShapePlot

# read and initialize data for plots
date = dt.datetime.now().date()
time = dt.datetime.now().hour

results = pd.read_csv('../../../data/forecasts/'+str(date)+'.csv', index_col=0, header=[0,1], date_parser=pd.to_datetime)
data = results.iloc[14:]

clusters = list(data.columns.get_level_values(0).unique())
models = list(data.columns.get_level_values(1).unique())
models.remove('actual')

# calculate mean absolute percentage errors for each forecast
mapes = {}
for cluster in clusters:
    mapes[cluster] = {}
    for model in models:
        mapes[cluster][model] = np.round(MAPE(data[cluster]['actual'],data[cluster][model]),3)

# generate primary (aggregate load) figure
aggfig = go.Figure([LoadShapePlot(data['aggregate','actual'].iloc[:time]),
    LoadShapePlot(data['aggregate','sarimax']),
    LoadShapePlot(data['aggregate','mlp'])
    ])

aggfig.update_layout(
    showlegend=False,
    margin={'t':40,'b':40,'l':40,'r':40},
    )

# text to accompany aggegate plot
aggtext = f"""
The plot shows the aggregate load of all buildings on the University of Arizona campus.
The legend describes each series.
Here are the mean absolute percentage error (MAPE) for each forecasting strategy:
* Aggregate SARIMAX: {mapes['aggregate']['sarimax']}
* Aggregate MLP: {mapes['aggregate']['mlp']}
* Cluster Sum SARIMAX: {mapes['clustersum']['sarimax']}
* Cluster Sum MLP: {mapes['clustersum']['mlp']}
"""

mapefig = go.Figure(
    data=go.Table(
        header=dict(
            values=['Forecasting Strategy','Mean Absolute Percent Error'],
            align='left'
        ),
        cells=dict(
            values=[
                ['Aggregate SARIMAX','Aggregate MLP',
                'Cluster Sum SARIMAX','Aggregate MLP'],
                [mapes['aggregate']['sarimax'],mapes['aggregate']['mlp'],
                mapes['clustersum']['sarimax'], mapes['clustersum']['mlp']]
                ],
            align='left')
    )
)
mapefig.update_layout(margin={'t':20,'b':20,'l':20,'r':20})

# initialize Dash app and add external css
app = dash.Dash(__name__, external_stylesheets='https://codepen.io/chriddyp/pen/bWLwgP.css')
# app.css.append_css({
#     "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
# })

styleDict = {
#     'fontFamily':['Avenir','Arial'],
#     'backgroundColor':'#FFFFFF',
#     'color':'#111111',
#     'display':'inline-block',
#     'textAlign':'left',
#     'vertical-align':'top'
}

app.layout = html.Div(children=[
    html.H1(
        children='University Load Forecasting:',
        style={'textAlign':'center'}
    ),
    html.Div(
        children="Applications of Deep Learning and Load-Shape Clustering",
        style={'textAlign':'center'}
    ),
    dcc.Graph(
        id='AggGraph',
        figure=aggfig,
        style={'width':'60%', 'height':'40%', 'display':'inline-block'},
        config={
            # 'staticPlot': True,
            # 'modeBarButtons': [['pan2d','zoom2d']],
            'displaylogo': False,
            'showTips': False,
            'scrollZoom': False,
            'displayModeBar':False
        },
    ),

    dcc.Markdown(
        # id='AggText',
        children=aggtext,
        style={'width':'39%', 'display':'inline-block', 'padding-top':20,'vertical-align':'top'}
    ),

    dcc.Graph(
        # id='MapeTable',
        figure=mapefig,
        style={'width':'39%', 'display':'inline-block','vertical-align':'top'},
        config={
            'staticPlot': True,
            'displayModeBar':False,
            'displaylogo': False,
            'showTips': False,
        },
    )
])

if __name__ == '__main__':
    app.server.run(port=8000, host='127.0.0.1', debug=True)
    # app.run_server(debug=True)