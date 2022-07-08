import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import os
from utils import save_csv

from args import args


class AutoEncoder(nn.Module):
    def __init__(self,inputNum,hiddenList):
        super(AutoEncoder,self).__init__()

        if hiddenList is None:
            hiddenList = [512,128,64]

        # build encoder
        inputNumTemp = inputNum
        modules = []
        for h in hiddenList:
            modules.append(
                nn.Sequential(
                    nn.Linear(inputNumTemp,h),
                    nn.ReLU()
                )
            )
            inputNumTemp = h       
        self.encoder = nn.Sequential(*modules)
        
        # build decoder
        modules = []
        hiddenListReverse = hiddenList[::-1]
        for i in range(len(hiddenListReverse) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hiddenListReverse[i],hiddenListReverse[i+1]),
                    nn.ReLU()
                )
            )
        modules.append(nn.Sequential(nn.Linear(hiddenListReverse[-1],inputNum))) # last layer
        
        self.decoder = nn.Sequential(*modules)
         
    def encode(self,inputTensor):
        return self.encoder(inputTensor)
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# TODO: need to evaluate the decomposition efficiency      
def train(model,ds:Dataset) -> None:
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
#     loss_func = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss()
    # starttime = time.time()
    
    trainInfo={'epoch':[],'loss':[],}
    for epoch in range(args.epoch):
        loss=0
        for x in loader:

            _, decoded = model.forward(x[0])
            lossBatch = loss_func(decoded,x[0])
            loss += lossBatch   
            optimizer.zero_grad()
            lossBatch.backward()
            optimizer.step()
        trainInfo['epoch'].append(epoch)
        trainInfo['loss'].append(loss)
            
        if epoch%50 == 0:
            print('Epoch :', epoch,'|','train_loss:%.4f'%loss)
    print('________________________________________')
    print('finish training')

    return model,trainInfo

def gdFeatureByAE(dis_feat,gene_feat,logInfo = args.logInfo, device=args.device):
    '''
    1. using AutoEncoder to decompose gene features and disease features
    '''
    disFeat = TensorDataset(torch.tensor(dis_feat.todense(),dtype=torch.float32,device=device))
    geneFeat = TensorDataset(torch.tensor(gene_feat.todense(),dtype=torch.float32,device=device))

    # init my AutoEncoder
    disAE = AutoEncoder(dis_feat.shape[1],args.ae_hidden).to(device)
    geneAE = AutoEncoder(gene_feat.shape[1],args.ae_hidden).to(device)
    
    # training disease AE and save model and loss
    disAETrain, disTrainInfo = train(model=disAE,ds=disFeat)
    disTrainInfo['loss'] = list(map(lambda x:float(x.detach()),disTrainInfo['loss']))
    disTrainInfoDF = pd.DataFrame(disTrainInfo)
    torch.save(disAETrain,os.path.join(logInfo['logPath'],logInfo['hour']+'disAE.model'))
    save_csv(disTrainInfoDF,logInfo,'disTrainLoss')
   
    # training gene AE and save model and loss
    geneAETrain,geneTrainInfo = train(model=geneAE,ds=geneFeat)
    geneTrainInfo['loss'] = list(map(lambda x:float(x.detach()),geneTrainInfo['loss']))
    geneTrainInfoDF = pd.DataFrame(geneTrainInfo)
    torch.save(geneAETrain,os.path.join(logInfo['logPath'],logInfo['hour']+'geneAE.model'))
    save_csv(geneTrainInfoDF,logInfo,'geneTrainLoss')

    # using AE to decompose feature
    dis_feat_refine = disAETrain.encode(torch.tensor(dis_feat.todense(),dtype=torch.float32,device=device)).cpu()
    gene_feat_refine = geneAETrain.encode(torch.tensor(gene_feat.todense(),dtype=torch.float32,device=device)).cpu()

    gdFeature = torch.vstack((gene_feat_refine,dis_feat_refine))
    torch.save(gdFeature,os.path.join(logInfo['logPath'],logInfo['hour']+'1_geneDisFeature.tensor'))
    # 生成一份临时文件，用于args.skip_ae=True时跳过AE的训练过程
    torch.save(gdFeature,'log/geneDisFeatureByAE.tensor') 
    
    return gdFeature

def featureByOpenNE(opennePath,logInfo = args.logInfo):
    '''
    2. using openNE to get network embedding features
    '''
    # opennePath = '/~/Documents/Github/OpenNE/results/DeepWalk/DeepWalk_embeddings.txt'
    strucFeat = []
    with open(opennePath,'r') as f:
        for i,line in enumerate(f):
            l = line.split(' ')
            if i==0:
                numFeature=int(l[1].strip())
            else:
                strucFeat.append(l)
    strucFeature = torch.from_numpy(np.array(strucFeat,dtype=float))
    
    # use mask to delete strucFeature first column (node index)
    mask=np.ones_like(strucFeature,dtype=bool)
    mask[:,0]=0 
    strucFeature = strucFeature.sort(dim=0).values[mask].reshape(i,numFeature) #(15546,128)
    embModel = opennePath.split('/')[-2] # 获取DeepWalk
    torch.save(strucFeature,os.path.join(logInfo['logPath'],logInfo['hour']+'2_embeddingFeaure_{}.tensor'.format(embModel)))
    return strucFeature

# 暂时用不到这个函数
def sampling(EdgePair,num):
    '''
    EdgePair : (array([    0,     2,     5, ..., 12128, 12186, 12210]),
                   array([  24,  939, 3093, ...,  687, 2810, 1002]))
    '''
    # must need set random seed before premutation
    np.random.seed(args.seed)
    perm = np.random.permutation(np.arange(EdgePair[0].shape[0]))
    perm = perm[:num]
    return (EdgePair[0][perm],EdgePair[1][perm])

def sampling_from_numpy(EdgePair,num):
    '''
    EdgePair : array([[    0,    24,     1],
                     [    2,   939,     1],
                     [    5,  3093,     1],
    '''
    # must need set random seed before premutation
    np.random.seed(args.seed)
    perm = np.random.permutation(np.arange(EdgePair.shape[0]))
    perm = perm[:num]
    return EdgePair[perm]

# def get_xy(gdTensor,num=3):
#     '''
#     gdTensor: is a matrix, [gene     , gene_dis]
#                            [dis_gene , disease ]
#     num: means that negNum is 3x posNum
#     '''
#     num_gene=12331
#     pos = gdTensor[:num_gene,num_gene:].nonzero() # pos.shape: (3954,) --> edgeIndex
#     neg = np.where(gdTensor[:num_gene,num_gene:]==0) # neg.shape: (39640211,)
#     negSample = sampling(neg,pos[0].shape[0] * num) # negSample.shape: (11862,) negNum is 3x posNum
#     dataPair = list(zip(pos[0],pos[1],[1]*len(pos[0]))) # len(dataPair): 15816
#     dataPair += list(zip(negSample[0],negSample[1],[0]*len(negSample[0]))) 
#     #dataPair: [(0, 24, 1), (2, 939, 1), (5, 3093, 1), (7, 2743, 1), (8, 286, 1)]

#     for i,pair in enumerate(dataPair):
#         if i==0:
#             data = torch.hstack((features[pair[0]],features[pair[1]]))
#             y = torch.tensor(pair[2])
#         else:
#             tempPair = torch.hstack((features[pair[0]],features[pair[1]]))
#             data = torch.vstack((data,tempPair))
#             y = torch.vstack((y,torch.tensor(pair[2]))) 
#     return data,y







    
