## import the packages that might be used
from ast import arguments
import copy
import os
import pickle
import pandas as pd
import random
from datetime import datetime
import scipy.sparse as sp
import numpy as np
from regex import P
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from pgmpy import device
from sklearn.model_selection import KFold, StratifiedKFold
from torch.nn import Linear
from torch_geometric.datasets import Entities, Flickr
from torch_geometric.nn import GCNConv, RGCNConv  # noqa
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   remove_self_loops,
                                   structured_negative_sampling,
                                   train_test_split_edges)

from utils import DiseaseGeneDataset, PRF1,set_seed

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# General Arguments
parser.add_argument('-se', '--seed', default=42, type=int,
                    help='Global random seed')
parser.add_argument('-md', '--mode', default='train', type=str,
                    help='turing the mode of the model')
parser.add_argument('-maskType', '--maskType', type=str,
                    help='running mode of main. mode: gene | dis', default='gene') 
parser.add_argument('-skipGraph', '--skipGraph', type=bool,
                    help='set True to skip make graph. mode: gene | dis', default=False)   
parser.add_argument('-savePath', '--saved_path', type=str,
                    help='Path to save training results', default='result')

args = parser.parse_args()
##### Hyperparameters #####
nfold=5
device='cpu'
random_state=args.seed
skipGraph = True
##### begin #####

startDate = datetime.now().strftime('%m-%d')
hour = datetime.now().strftime('%H_')
# hour = random_state
logPath = os.path.join('result_LUPI_RGCN',args.saved_path)

print('===================={}===================='.format(logPath))
print('===================={}===================='.format(args.maskType))

if not os.path.exists(logPath):
    os.makedirs(logPath)
    os.makedirs(os.path.join(logPath, 'bestModel'))

set_seed(random_state)

num_nodes = 15546
num_features = 37287

num_gene = 12331
num_Disease = 3215

#read the data saved in Processing_data.py
feature_matrix = np.load("data/feature_matrix.npy")
label = np.load("data/label.npy")
if args.mode =='train':
    edge_index = np.load("data/edge_index.npy")
elif args.mode == 'findNew':
    # edge_index = sp.load_npz("data/edge_index_findNew.npz").toarray()
    # edge_index_test = sp.load_npz("data/edge_index_findNew_test.npz").toarray()
    edge_index = sp.load_npz("data/{}_edge_index_findNew.npz".format(args.maskType)).toarray()
    edge_index_test = np.load("data/{}_findNew_pos_index.npy".format(args.maskType))
    

feature_matrix = feature_matrix.reshape((num_nodes,-1))
edge_index = edge_index.reshape((2,-1))

feature_matrix = torch.from_numpy(feature_matrix)
label = torch.from_numpy(label).long()
edge_index = torch.from_numpy(edge_index)

# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def get_dis_gene_edges(edge_indices): #,edge_type):
#     print('edge_indices.shape',edge_indices.shape)
    edges = []
    for i in range(edge_indices.shape[1]):
        if (edge_indices[:,i][0] < num_gene and edge_indices[:,i][1] >= num_gene) :
#            or (edge_indices[:,i][0] >= num_gene and edge_indices[:,i][1] < num_gene):
            edges.append(i)
#     print('len(edges)',len(edges))
    return edge_indices[:,edges],edges #edge_type[edges] 

# 应用pyg构图
if args.mode == 'train':
    graphPath = 'data/LUPI_RGCN_PYG/graphData_LUPI_RGCN.graph'
elif args.mode == 'findNew':
    graphPath = 'data/LUPI_RGCN_PYG_findNew_{}/graphData_LUPI_RGCN.graph'.format(args.maskType)
if skipGraph :
    data = torch.load(graphPath)
    print('skipGraph!!!,\tload graph data from {}'.format(graphPath))
else:
    print('using pyg to make graph')
    if args.mode == 'train':
        rootPath = 'data/LUPI_RGCN_PYG'
    elif args.mode == 'findNew':
        rootPath = 'data/LUPI_RGCN_PYG_findNew_{}'.format(args.maskType)
    dataset_gd = DiseaseGeneDataset(#root = "/scratch/gilbreth/shu30/GraphDropout/disease_gene.dataset",
                                    root = rootPath,
                                    feature_matrix=feature_matrix,
                                    label = label,
                                    edge_index = edge_index,)
    data = dataset_gd.data
    data.num_classes = 2
    data.num_relations = 3
    # 获取pos，neg边集，用于构造交叉验证的数据集，修改源代码中的分割测试集方式
    edge_index = data.edge_index

    gg_index = []
    dd_index = []
    gd_index = []
    for i in range(edge_index.shape[1]):
        if (edge_index[:,i][0] < num_gene and edge_index[:,i][1] >= num_gene) or \
        (edge_index[:,i][0] >= num_gene and edge_index[:,i][1] <= num_gene):
            gd_index.append(i)
        elif (edge_index[:,i][0] < num_gene and edge_index[:,i][1] < num_gene):
            gg_index.append(i)
        else:
            dd_index.append(i)
            
    edge_type_gd = torch.zeros(edge_index.shape[1]).long()
    edge_type_gd[gd_index] = torch.zeros(len(gd_index)).long()
    edge_type_gd[gg_index] = torch.ones(len(gg_index)).long()
    edge_type_gd[dd_index] = 2 * torch.ones(len(dd_index)).long()


    data.edge_type_all = edge_type_gd

    # Positive dis-gene edges
    pos_edge_index,pos_edges_colIndex = get_dis_gene_edges(data.edge_index) # ,data.edge_type)
    data.pos_edge_index = pos_edge_index 

    # Negative dis_gene edges.
    row, col = data.edge_index
    neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0
    # 没有连接的边
    data.none_edge_index = neg_adj_mask.nonzero(as_tuple=False).t()

    if args.mode == 'findNew': # TODO:test!!! 感觉应该没啥问题？
        # Positive dis-gene edges
        pos_edge_index_test,pos_edges_colIndex_test = get_dis_gene_edges(edge_index_test)
        data.pos_edge_index_test = pos_edge_index_test

        # Negative dis_gene edges.
        row, col = edge_index_test # TODO:看看应该用什么变量
        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0
        # 没有连接的边
        data.none_edge_index = neg_adj_mask.nonzero(as_tuple=False).t()

    # 取到没有连接的dis_gene边
    neg_edge_index,neg_edges_colIndex = get_dis_gene_edges(data.none_edge_index) # ,data.edge_type)
    data.neg_edge_index = neg_edge_index 
    torch.save(data,graphPath)

# data_dropout = copy.deepcopy(data)
# data_dropout.edge_type = edge_type_gd

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
            out = self.GCN(z) # 15546(num_nodes) * 16
            total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            x_j = torch.index_select(out, 0, total_edge_index[0])
            x_i = torch.index_select(out, 0, total_edge_index[1])
            return torch.einsum("ef,ef->e", x_i, x_j),z,out # out is node embedding,shape=15546*16 num_nodes*chanels
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


def train():
    TRAIN = True
    model.train()
    optimizer.zero_grad()

    x, pos_edge_index = data.x, data.train_pos_edge_index

    _edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                       num_nodes=x.size(0))

    neg_edge_index = negative_sampling(
#         edge_index=data.neg_edge_index, #
        edge_index = pos_edge_index_with_self_loops, 
        num_nodes=x.size(0),
        num_neg_samples= 3 * pos_edge_index.size(1))

    neg_edge_index,_ = get_dis_gene_edges(neg_edge_index)
    
    link_logits,y_d,node_embedding = model(pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)

    prf1 = PRF1(link_labels.detach().cpu().numpy(), torch.sigmoid(link_logits.detach().cpu()).numpy())

    criterion = My_loss()
    loss = criterion(link_logits, link_labels, y_d)
    loss.backward()
    optimizer.step()
    return loss,prf1


@torch.no_grad()
def test(ratio = 1):
    TRAIN = False
    model.eval()
    test_pos_edge_index = data.test_pos_edge_index
    test_neg_edge_index = data.test_neg_edge_index
    link_probs = torch.sigmoid(model(
                                     test_pos_edge_index, 
                                     test_neg_edge_index[:,range(test_pos_edge_index.shape[1]*ratio)])[0])
                                     # 最后[0]表示返回model的第一个返回值:link_logits,用于sigmoid函数做为预测值
    link_labels = get_link_labels(test_pos_edge_index, test_neg_edge_index[:,range(test_pos_edge_index.shape[1])])
    link_probs = link_probs.detach().cpu().numpy()
    link_labels = link_labels.detach().cpu().numpy()
    # prf1 = PRF1(link_labels, link_probs)

    # print("TrainROC:{}, TrainAUPR:{}".format(prf1['AUC'], prf1['AUPR']))

    return link_labels, link_probs

#################################Train start##############################################
if args.mode == 'train':
    kf = KFold(n_splits=nfold, shuffle=True, random_state=random_state)
    fold = 1
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(data.pos_edge_index.t()),
                                                                            kf.split(data.neg_edge_index.t())):
        print('================={}-Cross Validation: Fold {}==================='.format(nfold, fold))
        print(train_pos_idx.shape,test_pos_idx.shape,train_neg_idx.shape,test_neg_idx.shape)
        
        data.train_pos_edge_index = data.pos_edge_index.t()[train_pos_idx].t()
        data.train_neg_edge_index = data.neg_edge_index.t()[train_neg_idx].t()
        data.test_pos_edge_index = data.pos_edge_index.t()[test_pos_idx].t()
        data.test_neg_edge_index = data.neg_edge_index.t()[test_neg_idx].t()
        # 构造edge_type, 为dg边0
        data.edge_type = torch.zeros(data.train_pos_edge_index.shape[1]).long()
        
        data_dropout = copy.deepcopy(data) 
        model, data, data_dropout = Net().to(device), data.to(device), data_dropout.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

        bestAuc = 0
        bestLoss = 10
        resultDict={'epochs':[],'trainLoss':[],'ytest':[],'yproba':[]}

        for epoch in range(1, 701):
            TRAIN = True
            train_loss, train_prf1 = train()
            TRAIN = False
            labels, probs = test()
            test_prf1= PRF1(labels.flatten(), probs.flatten())        
            if epoch%100==0:
                print("EPOCH:{} \tLoss:{} \t| TrainROC:{}, TrainAUPR:{} \t| TestROC:{}, TestAUPR:{}".format(
                    epoch, train_loss, train_prf1['AUC'], train_prf1['AUPR'], test_prf1['AUC'], test_prf1['AUPR']))
            resultDict['epochs'].append(epoch)
            resultDict['ytest'].append(labels)
            resultDict['yproba'].append(probs)
            resultDict['trainLoss'].append(train_loss.to(float))

            if (train_loss <= bestLoss) and (test_prf1['AUC'] >= bestAuc):
                bestLoss = train_loss
                bestAuc = test_prf1['AUC']
                torch.save(model.state_dict(), 
                        os.path.join(logPath, 'bestModel','{}bestModel_fold{}.pth'.format(hour,fold)))
        dataPackage={'sam':[],'ytest':[],'yprob':[]}
        for sam in [1,3,5,10,20]:
            TRAIN = False
            labels, probs = test(sam)
            s = PRF1(labels,probs)
            print('sam: {}, microAUC: {}, microAPUR: {}'.format(sam,s['AUC'],s['AUPR']))
        dataPackage['sam'].extend([sam]*len(labels))
        dataPackage['ytest'].extend(list(labels))
        dataPackage['yprob'].extend(list(probs))
        pd.DataFrame(dataPackage).to_csv('log/LUPI_RGCN/seed{}_fold{}_sam{}.csv'.format(args.seed,fold,sam),index=False)
        with open(os.path.join(logPath,'{}result_fold{}.pickle'.format(hour,fold)),'wb') as f:
            pickle.dump(resultDict,f)
        fold += 1


elif args.mode == 'findNew':
    print('findNew')

    data.train_pos_edge_index = data.pos_edge_index
    data.train_neg_edge_index = data.neg_edge_index
    data.test_pos_edge_index = torch.tensor((np.load("data/{}_findNew_pos_index.npy".format(args.maskType))))
    data.test_neg_edge_index = torch.tensor((np.load("data/{}_findNew_neg_index.npy".format(args.maskType))))

    print('train_pos_edge_index: ',data.train_pos_edge_index.shape,
          'train_neg_edge_index: ',data.train_neg_edge_index.shape,
          'test_pos_edge_index: ',data.test_pos_edge_index.shape,
          'test_neg_edge_index: ',data.test_neg_edge_index.shape)
    # 构造edge_type, 为dg边0
    data.edge_type = torch.zeros(data.train_pos_edge_index.shape[1]).long()
    
    data_dropout = copy.deepcopy(data)
    
    model, data, data_dropout = Net().to(device), data.to(device), data_dropout.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)


    bestAuc = 0
    bestLoss = 10
    resultDict={'epochs':[],'trainLoss':[],'ytest':[],'yproba':[]}
    for epoch in range(1, 701):
        TRAIN = True
        train_loss, train_prf1 = train()
        TRAIN = False
        labels, probs = test()
        test_prf1= PRF1(labels.flatten(), probs.flatten())
             
        if epoch%50==0:
            print("EPOCH:{} \tLoss:{} \t| TrainROC:{}, TrainAUPR:{} \t| TestROC:{}, TestAUPR:{}".format(
                epoch, train_loss, train_prf1['AUC'], train_prf1['AUPR'], test_prf1['AUC'], test_prf1['AUPR']))

        resultDict['epochs'].append(epoch)
        resultDict['ytest'].append(labels)
        resultDict['yproba'].append(probs)
        resultDict['trainLoss'].append(train_loss.to(float))

        if (train_loss <= bestLoss) and (test_prf1['AUC'] >= bestAuc):
            bestLoss = train_loss
            bestAuc = test_prf1['AUC']
            torch.save(model.state_dict(), 
                    os.path.join(logPath, 'bestModel','{}bestModel_fold{}.pth'.format(hour,'all')))
        if epoch ==700: # 保存最后一次的ytest和yprob
            testProb = pd.DataFrame([labels,probs]).T.rename({0:'ytest',1:'yprob'},axis=1)
            tp=args.maskType
            testProb.to_csv(os.path.join(logPath,'LUPI_RGCN_uk{}_{}.csv'.format(tp[0].upper()+tp[1:],args.seed)))


    with open(os.path.join(logPath,'LUPI_RGCN_uk{}_{}.pickle'.format(tp[0].upper()+tp[1:],args.seed)),'wb') as f:
        pickle.dump(resultDict,f)

