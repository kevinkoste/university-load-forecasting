import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def LoadShapeCluster(df, num_clusters, method='average'):
    """
    Takes an hourly indexed df with one column per load node
    Returns a df with the same index, one column per cluster (sum of nodes)
    """
    scaler = MinMaxScaler()
    loadShape = df.groupby(df.index.hour).sum()
    loadShapeNorm = pd.DataFrame(scaler.fit_transform(loadShape.values),columns=loadShape.columns)

    # create the hour-of-day linkage matrix
    linkageMatrix = linkage(loadShapeNorm.values.T, method=method)

    # calculate the cophenetic correlation coefficient, a clustering evaluation metric (>0.75 is good)
    clusterScore, coph_dists = cophenet(linkageMatrix, pdist(loadShapeNorm.values.T))

    # create clusters based on dendrograms, middle argument gives number of clusters
    clusterMap = fcluster(linkageMatrix, num_clusters, criterion='maxclust')
    clusterDict = dict(zip(df.columns, clusterMap))
    
    # return dictionary of clusterMap and clusterScore
    return clusterDict, clusterScore

def PlotDendrogram(df, method='average'):
    # calculate load shapes
    scaler = MinMaxScaler()
    loadShape = df.groupby(df.index.hour).sum()
    loadShapeNorm = pd.DataFrame(scaler.fit_transform(loadShape.values),columns=loadShape.columns)

    # plot the dendrogram
    plt.figure(figsize=(18, 5))
    plt.title('Dendrogram of Nodal Daily Load Shapes')
    dendrogram(linkageMatrix, orientation='top', labels=loadShapeNorm.columns, distance_sort='descending')
    return loadShapeNorm