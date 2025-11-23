"""
Utility functions for iVAE analysis and evaluation.

This module provides utility functions for processing results, computing
evaluation metrics, and analyzing latent representations from iVAE models.
"""

import numpy as np
from numpy import ndarray
import pandas as pd
import scib
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.sparse import csr_matrix 
from scipy.sparse import csgraph

def get_dfs(
    mode, 
    agent_list
):
    """
    Aggregate and summarize scores from multiple agent runs.
    
    Parameters
    ----------
    mode : str
        Aggregation mode, either 'mean' or 'std'.
    agent_list : list
        List of trained agent objects with score histories.
    
    Returns
    -------
    generator
        Generator yielding DataFrames with aggregated scores.
        Columns are: ARI, NMI, ASW, C_H, D_B, P_C.
    """
    if mode == 'mean':
        ls = list(map(lambda x: zip(*(np.array(b).mean(axis=0) for b in zip(*((zip(*a.score)) for a in x)))), list(zip(*agent_list))))
    else:
        ls = list(map(lambda x: zip(*(np.array(b).std(axis=0) for b in zip(*((zip(*a.score)) for a in x)))), list(zip(*agent_list))))
    return (map(lambda x:pd.DataFrame(x, columns=['ARI', 'NMI', 'ASW', 'C_H', 'D_B', 'P_C']),ls))

def moving_average(
    a, 
    window_size
):
    """
    Compute moving average with boundary handling.
    
    This function computes a moving average while properly handling boundary
    conditions at the start and end of the array.
    
    Parameters
    ----------
    a : numpy.ndarray
        Input array to smooth.
    window_size : int
        Size of the moving average window.
    
    Returns
    -------
    numpy.ndarray
        Smoothed array of the same length as input.
    """
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def fetch_score(
    adata, 
    latent, 
    label_true, 
    label_mode='KMeans', 
    batch=False
):
    """
    Compute comprehensive evaluation metrics for latent representations.
    
    This function evaluates the quality of latent representations by computing
    clustering metrics, graph connectivity, and batch integration metrics.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (will be modified in place to add latent embeddings).
    latent : numpy.ndarray
        Latent representations of shape (n_cells, latent_dim).
    label_true : array-like
        True cluster labels for cells.
    label_mode : str, optional
        Method for assigning labels from latent space:
        - 'KMeans': Apply K-means clustering (default)
        - 'Max': Use argmax of latent dimensions
        - 'Min': Use argmin of latent dimensions
    batch : bool, optional
        If True, compute batch integration metrics (requires 'batch' in adata.obs).
        Default is False.
    
    Returns
    -------
    tuple
        If batch=False: (NMI, ARI, ASW, C_H, D_B, G_C, clisi)
        If batch=True: (NMI, ARI, ASW, C_H, D_B, G_C, clisi, ilisi, bASW)
        where:
        - NMI: Normalized Mutual Information
        - ARI: Adjusted Rand Index  
        - ASW: Average Silhouette Width
        - C_H: Calinski-Harabasz score
        - D_B: Davies-Bouldin score
        - G_C: Graph connectivity
        - clisi: Cell-type Local Inverse Simpson's Index
        - ilisi: Batch Local Inverse Simpson's Index (batch integration)
        - bASW: Batch Average Silhouette Width (batch integration)
    """
    q_z = latent
    if label_mode == 'KMeans':
        labels = KMeans(q_z.shape[1]).fit_predict(q_z)
    elif label_mode == 'Max': 
        labels = np.argmax(q_z, axis=1)
    elif label_mode == 'Min':
        labels = np.argmin(q_z, axis=1)
    else:
        raise ValueError('Mode must be in one of KMeans, Max and Min')
        
    adata.obsm['X_qz'] = q_z
    adata.obs['label'] = pd.Categorical(labels)
    
    NMI = normalized_mutual_info_score(label_true, labels)
    ARI = adjusted_mutual_info_score(label_true, labels)
    ASW = silhouette_score(q_z, labels) 
    if label_mode != 'KMeans':
        ASW = abs(ASW)
    C_H = calinski_harabasz_score(q_z, labels)
    D_B = davies_bouldin_score(q_z, labels)
    G_C = graph_connection(kneighbors_graph(adata.obsm['X_qz'], 15), adata.obs['label'].values)
    clisi = scib.metrics.clisi_graph(adata, 'label', 'embed', 'X_qz', n_cores=26)

    if batch:
        sub_adata = adata[np.random.choice(adata.obs_names, 5000, replace=False)].copy()
        ilisi = scib.metrics.ilisi_graph(sub_adata, 'batch', 'embed', 'X_qz', n_cores=26)
        bASW = scib.metrics.silhouette_batch(sub_adata, 'batch', 'label', 'X_qz')
        print('Completed')
        return NMI, ARI, ASW, C_H, D_B, G_C, clisi, ilisi, bASW
    print('Completed')
    return NMI, ARI, ASW, C_H, D_B, G_C, clisi

def graph_connection(
    graph: csr_matrix,
    labels: ndarray
):
    """
    Compute graph connectivity score for each cluster.
    
    This metric measures how well connected cells within each cluster are
    in the k-nearest neighbor graph. Higher values indicate better connectivity.
    
    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        Sparse adjacency matrix representing the k-nearest neighbor graph.
    labels : numpy.ndarray
        Cluster labels for each cell.
    
    Returns
    -------
    float
        Average connectivity score across all clusters (range: 0 to 1).
        Higher values indicate cells in the same cluster are well-connected.
    """
    cg_res = []
    for l in np.unique(labels):
        mask = np.where(labels==l)[0]
        subgraph = graph[mask, :][:, mask]
        _, lab = csgraph.connected_components(subgraph, connection='strong')
        tab = np.unique(lab, return_counts=True)[1]
        cg_res.append(tab.max() / tab.sum())
    return np.mean(cg_res)




