import numpy as np
import plotly.graph_objs as go

def MAPE(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true))*100

def LoadShapePlot(name, plotdata, **kwargs):
    """
    Takes a pandas.Series and returns a Plotly Scatter object with custom attributes
    kwargs are passed to plotly.graph_objs.Scatter
    """
    trace = go.Scatter(
        x=plotdata.index,
        y=plotdata.values,
        mode='lines',
        name=name,
        hovertemplate = '%{fullData.name}<br>Load: %{y:.2f} MW<br>Time: %{x}<extra></extra>',
        **kwargs)

    return trace