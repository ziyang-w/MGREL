# import copy
# import math
# import os
# import os.path as osp
# import pickle
# import random
# import time
# from itertools import combinations
# from operator import itemgetter

import h5py
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
# import pandas as pd
# import scipy.io as sio
import scipy.sparse as sp
import torch

from utils import network_edge_threshold
#      (get_sparse_mat,  sparse_to_tuple,DiseaseGeneDataset, train_test_split_edges)

# import torch.nn as nn
# import torch.nn.functional as F
# import torch_geometric.transforms as T
# from sklearn import metrics
# from sklearn.metrics import (auc, average_precision_score, f1_score,
#                              precision_recall_curve, roc_auc_score, roc_curve)
# from torch.nn import BatchNorm1d, Linear
# from torch.utils.data import Dataset
# from torch_geometric.data import (Batch, ClusterData, ClusterLoader, Data,
#                                   DataLoader, Dataset,
#                                   GraphSAINTRandomWalkSampler,
#                                   GraphSAINTSampler, InMemoryDataset,
#                                   NeighborSampler)
# # from .utils import get_sparse_mat, sparse_to_tuple, network_edge_threshold
# from torch_geometric.datasets import Entities, Flickr
# from torch_geometric.nn import (ChebConv, FastRGCNConv, GCNConv,  # noqa
#                                 Node2Vec, RGCNConv, global_add_pool,
#                                 global_mean_pool)
# from torch_geometric.utils import (add_self_loops, degree, negative_sampling,
#                                    remove_self_loops,
#                                    structured_negative_sampling, to_undirected,
#                                    train_test_split_edges)
# from torch_scatter import scatter_mean


num_nodes = 15546
num_features = 37287

num_gene = 12331
num_disease = 3215


# gene interaction network
gene_phenes_path = 'data/data_prioritization/genes_phenes.mat'
f = h5py.File(gene_phenes_path, 'r')
gene_network_adj = sp.csc_matrix((np.array(f['GeneGene_Hs']['data']),
                                  np.array(f['GeneGene_Hs']['ir']), np.array(f['GeneGene_Hs']['jc'])),
                                 shape=(12331, 12331))
gene_network_adj = gene_network_adj.tocsr()
gene_network_adj_row = (gene_network_adj.tocoo()).row
gene_network_adj_col = (gene_network_adj.tocoo()).col
gene_network_adj_value = np.array(f['GeneGene_Hs']['data'])

disease_network_adj = sp.csc_matrix((np.array(f['PhenotypeSimilarities']['data']),
                                     np.array(f['PhenotypeSimilarities']['ir']),
                                     np.array(f['PhenotypeSimilarities']['jc'])),
                                    shape=(3215, 3215))
disease_network_adj = disease_network_adj.tocsr()

disease_network_adj_row = (disease_network_adj.tocoo()).row
disease_network_adj_col = (disease_network_adj.tocoo()).col
disease_network_adj_value = np.array(f['PhenotypeSimilarities']['data'])

# gene disease network
dg_ref = f['GenePhene'][0][0]
gene_disease_adj = sp.csc_matrix((np.array(f[dg_ref]['data']),
                                  np.array(f[dg_ref]['ir']), np.array(f[dg_ref]['jc'])),
                                 shape=(12331, 3215))
gene_disease_adj = gene_disease_adj.tocsr()

# novel disease network
novel_associations_adj = sp.csc_matrix((np.array(f['NovelAssociations']['data']),
                                        np.array(f['NovelAssociations']['ir']), np.array(f['NovelAssociations']['jc'])),
                                       shape=(12331, 3215))
novel_associations_adj_row = (novel_associations_adj.tocoo()).row
novel_associations_adj_col = (novel_associations_adj.tocoo()).col
novel_associations_adj_values = np.array(f['NovelAssociations']['data'])

# disease features
disease_tfidf_path = 'data/data_prioritization/clinicalfeatures_tfidf.mat'
f_disease_tfidf = h5py.File(disease_tfidf_path, "r")
disease_tfidf = np.array(f_disease_tfidf['F'])
disease_tfidf = np.transpose(disease_tfidf)
disease_tfidf = sp.csc_matrix(disease_tfidf)

# Gene feature1:microarray features
gene_feature_path = 'data/data_prioritization/GeneFeatures.mat'
f_gene_feature = h5py.File(gene_feature_path, 'r')
gene_feature_exp = np.array(f_gene_feature['GeneFeatures'])
gene_feature_exp = np.transpose(gene_feature_exp)
gene_network_exp = sp.csc_matrix(gene_feature_exp)

# Gene feature2:other species features
row_list = [3215, 1137, 744, 2503, 1143, 324, 1188, 4662, 1243]
gene_feature_list_other_spe = list()
for i in range(1, 9):
    dg_ref = f['GenePhene'][i][0]
    disease_gene_adj_tmp = sp.csc_matrix((np.array(f[dg_ref]['data']),
                                          np.array(f[dg_ref]['ir']), np.array(f[dg_ref]['jc'])),
                                         shape=(12331, row_list[i]))
    gene_feature_list_other_spe.append(disease_gene_adj_tmp)

# Combine the gene features: 4536 micro-array features and features from other species
gene_feat = sp.hstack(gene_feature_list_other_spe + [gene_feature_exp])

dis_feat = disease_tfidf

### Create the gene disease network from gene network and disease network
# gene Network
gene_network_adj_row = (gene_network_adj.tocoo()).row
gene_network_adj_col = (gene_network_adj.tocoo()).col
gene_network_adj_value = np.array(f['GeneGene_Hs']['data'])
gene_network_matrix = np.zeros((np.shape(gene_network_adj)))
for i, j in zip(gene_network_adj_row, gene_network_adj_col):
    gene_network_matrix[i, j] = 1

# disease Network
disease_network_adj = network_edge_threshold(disease_network_adj, 0.2)
disease_network_adj_row = (disease_network_adj.tocoo()).row
disease_network_adj_col = (disease_network_adj.tocoo()).col
disease_network_adj_value = np.array(f['PhenotypeSimilarities']['data'])
disease_network_matrix = np.zeros((np.shape(disease_network_adj)))
for i, j in zip(disease_network_adj_row, disease_network_adj_col):
    disease_network_matrix[i, j] = 1
disease_network_matrix = disease_network_matrix - np.diag(np.ones(len(disease_network_matrix)))

# gene-disease network
novel_associations_adj_row = (novel_associations_adj.tocoo()).row
novel_associations_adj_col = (novel_associations_adj.tocoo()).col
novel_associations_adj_values = np.array(f['NovelAssociations']['data'])

#disease:
gene_disease_adj_row = (gene_disease_adj.tocoo()).row
gene_disease_adj_col = (gene_disease_adj.tocoo()).col
disease_gene_matrix = np.zeros((np.shape(gene_disease_adj)))
for i, j in zip(gene_disease_adj_row, gene_disease_adj_col):
    disease_gene_matrix[i, j] = 1

#novel disease:
# disease_gene_matrix = np.zeros((np.shape(novel_associations_adj)))
# for i, j in zip(novel_associations_adj_row, novel_associations_adj_col):
#     disease_gene_matrix[i, j] = 1

comb1 = np.hstack((gene_network_matrix, disease_gene_matrix))
comb2 = np.hstack((disease_gene_matrix.T, disease_network_matrix))
comb = np.vstack((comb1, comb2))

### Combine the fetaure matrix
disease_network_adj_row = (disease_network_adj.tocoo()).row
disease_network_adj_col = (disease_network_adj.tocoo()).col
disease_network_adj_value = np.array(f['PhenotypeSimilarities']['data'])
disease_network_matrix = np.zeros((np.shape(disease_network_adj)))
for i, j in zip(disease_network_adj_row, disease_network_adj_col):
    disease_network_matrix[i, j] = 1
disease_network_matrix = disease_network_matrix - np.diag(np.ones(len(disease_network_matrix)))

# edge_index, label and the feature matrix that will be used
edge_index = torch.tensor(comb.nonzero()) # 已经返回的是连接的位置了, shape=(2,1376654)
label = torch.zeros(num_disease + num_gene)
label[0:num_gene] = 1
disease_feat_matrix = dis_feat.tocsc().toarray()
disease_feat_none = np.zeros((num_gene, np.shape(disease_feat_matrix)[1]))
gene_feat_matrix = gene_feat.tocsc().toarray()
gene_feat_none = np.zeros((num_disease,np.shape(gene_feat_matrix)[1]))

# feature matrix,shape=(15546, 34072)
feature_matrix = np.vstack((np.hstack((gene_feat_matrix, disease_feat_none)), np.hstack(
    (gene_feat_none, disease_feat_matrix))))

feature_matrix = torch.from_numpy(feature_matrix).double()

# save ndarray to .npy
np.save("./feature_matrix.txt",feature_matrix)
np.save("./label.txt",label)
np.save("./edge_index.txt",edge_index)
