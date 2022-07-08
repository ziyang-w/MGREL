## import the packages that might be used
from pgmpy import device
from torch_geometric.datasets import Entities,Flickr
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import matplotlib as mpl  
import matplotlib.pyplot as plt

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import os.path as osp
from torch_geometric.data import Dataset
from torch_geometric.data import NeighborSampler
from torch_geometric.data import Batch, ClusterData, ClusterLoader, DataLoader

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops, structured_negative_sampling,train_test_split_edges)

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, RGCNConv  # noqa
from torch_geometric.nn import Node2Vec

import math
import random
from torch_geometric.utils import to_undirected
from torch_geometric.data import GraphSAINTRandomWalkSampler, GraphSAINTSampler
from torch_geometric.utils import degree
from torch_geometric.nn import FastRGCNConv

import copy

from utils import DiseaseGeneDataset,train_test_split_edges

def set_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



set_seed(42)

num_nodes = 15546
num_features = 37287

num_gene = 12331
num_Disease = 3215

#read the data saved in Processing_data.py
feature_matrix = np.load("./feature_matrix.txt.npy")
label = np.load("./label.txt.npy")
edge_index = np.load("./edge_index.txt.npy")

feature_matrix = feature_matrix.reshape((num_nodes,-1))
edge_index = edge_index.reshape((2,-1))

feature_matrix = torch.from_numpy(feature_matrix)
label = torch.from_numpy(label).long()
edge_index = torch.from_numpy(edge_index)

# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

dataset_gd = DiseaseGeneDataset(#root = "/scratch/gilbreth/shu30/GraphDropout/disease_gene.dataset",
                                root = "./DiseaseGeneDataset",
                                feature_matrix=feature_matrix,
                                label = label,
                                edge_index = edge_index,)

data = dataset_gd.data
data.num_classes = 2
data.num_relations = 3
data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)
data_dropout = copy.deepcopy(data)

edge_index = data.train_pos_edge_index
num_gene = 12331
gg_index = []
dd_index = []
gd_index = []
for i in range(edge_index.shape[1]):
    if (edge_index[:,i][0] < num_gene and edge_index[:,i][1] >= num_gene) or (edge_index[:,i][0] >= num_gene and edge_index[:,i][1] <= num_gene):
        gd_index.append(i)
    elif (edge_index[:,i][0] < num_gene and edge_index[:,i][1] < num_gene):
        gg_index.append(i)
    else:
        dd_index.append(i)

# 构造边类型
# 0->gd_edge
# 1->gg_edge
# 2->dd_edge
edge_type_gd = torch.zeros(edge_index.shape[1]).long()
edge_type_gd[gd_index] = torch.zeros(len(gd_index)).long() 
edge_type_gd[gg_index] = torch.ones(len(gg_index)).long()
edge_type_gd[dd_index] = 2 * torch.ones(len(dd_index)).long()


data.edge_type = edge_type_gd
data_dropout.edge_type = edge_type_gd

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = RGCNConv(data.num_features, 16, data.num_relations,
                                num_bases=30)
        self.conv2 = RGCNConv(16, 16, data.num_relations,
                                num_bases=30)
        self.fc1 = Linear(16, 16)

        self.Dropconv1 = GCNConv(data.num_features, 16, cached=False).to(torch.double)
        self.Dropconv2 = GCNConv(16, 16, cached=False).to(torch.double)
        self.Dropfc1 = Linear(16, 16).to(torch.double)

    def reparameterize(self,logvar):
        std = torch.exp(-0.5*logvar)
        return std

    def getvar(self):
        x = F.relu(self.Dropconv1(data.x, data.train_pos_edge_index))
        x = self.Dropconv2(x, data.train_pos_edge_index)
        x = self.Dropfc1(x)
        return x

    def GCN(self,z=None):

        y = F.relu(self.conv1(data.x.to(torch.float32), data.train_pos_edge_index, data.edge_type))
        y = self.conv2(y, data.train_pos_edge_index, data.edge_type)
        y = self.fc1(y)
        if TRAIN:
            y_out = y.mul(torch.randn(y.size()).to(device).mul(z)+1) #没看懂在干嘛？
        else:
            y_out = y
        return y_out


    def forward(self, pos_edge_index, neg_edge_index):
        if TRAIN:
            z = self.reparameterize(self.getvar())
            out = self.GCN(z)
            total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            x_j = torch.index_select(out, 0, total_edge_index[0])
            x_i = torch.index_select(out, 0, total_edge_index[1])
            return torch.einsum("ef,ef->e", x_i, x_j),z,out
        else:
            out = self.GCN()
            total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            x_j = torch.index_select(out, 0, total_edge_index[0])
            x_i = torch.index_select(out, 0, total_edge_index[1])
            return torch.einsum("ef,ef->e", x_i, x_j),out

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, y_d):
        return F.binary_cross_entropy_with_logits(x, y) + 0.000001 * torch.sum(torch.sum(torch.log(y_d)**2))


def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                                neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def get_dis_gene_edges(edge_indices,edge_type):
    '''
    从给定的边索引中找到dg边,并且返回dg边的特征.
    '''
    edges = []
    for i in range(edge_indices.shape[1]):
        if (edge_indices[:,i][0] < num_gene and edge_indices[:,i][1] >= num_gene) or \
            (edge_indices[:,i][0] >= num_gene and edge_indices[:,i][1] < num_gene): # 确保index在dis_gene_matrix范围内
            edges.append(i)
    return edge_indices[:,edges],edge_type[edges]

train_pos_edge_index,train_edge_type = get_dis_gene_edges(data.train_pos_edge_index,data.edge_type)
data.train_pos_edge_index = train_pos_edge_index
data.edge_type = train_edge_type

def train():
    TRAIN = True
    model.train()
    optimizer.zero_grad()

    x, pos_edge_index = data.x, data.train_pos_edge_index # pos means dis-gene edge (56 edges)

    _edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                        num_nodes=x.size(0))

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
        num_neg_samples=3 * pos_edge_index.size(1)) # after sampling, neg edges is 56*3 edges

    neg_edge_type = torch.zeros(neg_edge_index.shape[1]).long()
    neg_edge_index, neg_edge_type = get_dis_gene_edges(neg_edge_index,neg_edge_type)
    # 然后从不连接边中, 找到dg不连接边, 作为最后的训练集传入模型。

    link_logits,y_d,node_embedding = model(pos_edge_index, neg_edge_index)
    
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)

    print("TrainROC:",roc_auc_score(link_labels.detach().cpu(), torch.sigmoid(link_logits.detach().cpu())),"TrainPRC:",average_precision_score(link_labels.detach().cpu(), torch.sigmoid(link_logits.detach().cpu())))

    criterion = My_loss()
    loss = criterion(link_logits, link_labels, y_d)
    loss.backward()
    optimizer.step()

    return loss



model, data, data_dropout = Net().to(device), data.to(device), data_dropout.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

# 找到test中属于dg边的集合, 根据结果来看,该值为6
test_pos = []
for i in range(data.test_pos_edge_index.shape[1]):
    if (data.test_pos_edge_index[:,i][0] < num_gene and data.test_pos_edge_index[:,i][1] >= num_gene) or \
        (data.test_pos_edge_index[:,i][0] >= num_gene and data.test_pos_edge_index[:,i][1] <= num_gene):
        test_pos.append(i)
test_pos_edge_index = data.test_pos_edge_index[:,test_pos].to(device)

test_neg = []
for i in range(data.test_neg_edge_index.shape[1]):
    if (data.test_neg_edge_index[:,i][0] < num_gene and data.test_neg_edge_index[:,i][1] >= num_gene) or \
        (data.test_neg_edge_index[:,i][0] >= num_gene and data.test_neg_edge_index[:,i][1] <= num_gene):
        test_neg.append(i)
test_neg_edge_index = data.test_neg_edge_index[:,test_neg].to(device)

@torch.no_grad()
def test():
    TRAIN = False
    model.eval()
    link_probs = torch.sigmoid(model(
                                     test_pos_edge_index, 
                                     test_neg_edge_index[:,range(test_pos_edge_index.shape[1])])[0])
                                     # 最后[0]表示返回model的第一个返回值:link_logits,用于sigmoid函数做为预测值
    link_labels = get_link_labels(test_pos_edge_index, test_neg_edge_index[:,range(test_pos_edge_index.shape[1])])
    link_probs = link_probs.detach().cpu().numpy()
    link_labels = link_labels.detach().cpu().numpy()
    print("TestROC:",roc_auc_score(link_labels, link_probs))
    print("TestPRC:",average_precision_score(link_labels, link_probs))
    print('='*20)

    return link_labels, link_probs

best_val_perf = test_perf = 0
resultDict={'epochs':[],'ytest':[],'yproba':[]}
for epoch in range(1, 701):
    TRAIN = True
    train_loss = train()
    TRAIN = False
    labels, probs = test()

    resultDict['epochs'].append(epoch)
    resultDict['ytest'].append(labels)
    resultDict['yproba'].append(probs)

    print("EPOCH:",epoch, train_loss)

import pandas as pd
pd.DataFrame(resultDict).to_csv('result.csv',index=False,encoding='utf-8-sig')
import pickle

with open('result.pickle','wb')as f:
    pickle.dump(resultDict,f)

# if __name__=='__main__':