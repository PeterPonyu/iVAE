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
    if mode == 'mean':
        ls = list(map(lambda x: zip(*(np.array(b).mean(axis=0) for b in zip(*((zip(*a.score)) for a in x)))), list(zip(*agent_list))))
    else:
        ls = list(map(lambda x: zip(*(np.array(b).std(axis=0) for b in zip(*((zip(*a.score)) for a in x)))), list(zip(*agent_list))))
    return (map(lambda x:pd.DataFrame(x, columns=['ARI', 'NMI', 'ASW', 'C_H', 'D_B', 'P_C']),ls))

def moving_average(
    a, 
    window_size
):
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
    cg_res = []
    for l in np.unique(labels):
        mask = np.where(labels==l)[0]
        subgraph = graph[mask, :][:, mask]
        _, lab = csgraph.connected_components(subgraph, connection='strong')
        tab = np.unique(lab, return_counts=True)[1]
        cg_res.append(tab.max() / tab.sum())
    return np.mean(cg_res)




