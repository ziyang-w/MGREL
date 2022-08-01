from itertools import combinations
from sklearn.model_selection import KFold
from args import args
import pandas as pd
import numpy as np
import torch
import os
import h5py
import load_data
import machineLearning as ml
from utils import set_seed, save_csv,PRF1,save_pickle
from feature_extration import gdFeatureByAE, featureByOpenNE,sampling_from_numpy


def train():
    randomState = set_seed(args.seed)

    '''
    1. load data from mat
    2. make data for OpenNE
    3. OpenNE
    4. using AutoEncoder to decompose gene features and disease features
    5. using openNE to get network embedding features
    6. using nfold ml model to calculate the score
    '''

    gene_feat, dis_feat, comb, label = load_data.load_data_from_mat(skip=args.skip_load_data,save=True)

    # OpenNE TODO: 将OpenNE的超参数放在args中
    # OpenNE TODO: 实现OpenNE切换运行环境自动化运行
    if args.skip_openne:
        print('skip OpenNE!!!')
    else:
        load_data.make_data_for_openne(comb,label)
        openneLog = os.path.join(args.logInfo['logPath'], '{}openne.log'.format(args.logInfo['hour']))
        print('=======running OpenNE========')
        print('this may take a while, and the log will be saved in {}'.format(openneLog))
        os.system('bash OpenNE/src/run_openne.sh > {}'.format(openneLog))
        print('finished OpenNE!')

    if args.skip_ae:
        gdFeature = torch.load('log/geneDisFeatureByAE.tensor')
        print('skip AutoEncoder!!!')
    else:
        print('=======running AutoEncoder========')
        gdFeature = gdFeatureByAE(dis_feat,gene_feat)


    logInfo = args.logInfo
    nfold=args.nfold 

    # openNEModel = ['DeepWalk', 'LINE', 'Node2vec', 'HOPE', 'SDNE','GraRep'] # same to run_model.sh , there are some bugs in 'GraRep'
    openNEModel = args.openne_method#['DeepWalk', 'LINE', 'Node2vec', 'HOPE', 'SDNE','GraRep'] # same to run_model.sh , there are some bugs in 'GraRep'

    # ===应用最优组合===
    meth = openNEModel #['SDNE','HOPE] # 
    ms = '_'.join(meth) # list to string, like 'DeepWalk_LINE', aim to save the model name
    print('==============={}=============='.format(ms))
    embeddingFeature = torch.ones(len(comb)).reshape(-1,1) #用于初始化拼接向量，便于后面的拼接
    for m in meth:
        openNEPath = 'OpenNE/results/{}/{}_embeddings.txt'.format(m,m)
        print('load from OpenNE/results/{}/{}_embeddings.txt'.format(m,m))
        embeddingFeatureTemp = featureByOpenNE(openNEPath)
        embeddingFeature = torch.cat((embeddingFeature,embeddingFeatureTemp),dim=1)
    embeddingFeature = embeddingFeature[:,1:] #去掉第一列，因为第一列是1，即表示节点的编号

    print('embeddingFeature shape: {}'.format(embeddingFeature.shape))

    features = torch.hstack((gdFeature, embeddingFeature)) #shape=12331+3215,64+64

    num_gene=12331
    pos = comb[:num_gene,num_gene:].nonzero() # pos.shape: (3954,) --> edgeIndex TODO: 查看下pos.shape
    neg = np.where(comb[:num_gene,num_gene:]==0) # neg.shape: (39640211,)

    posIndex = np.vstack((pos[0],pos[1]+num_gene,np.ones_like(pos[0]))).T
    negIndex = np.vstack((neg[0],neg[1]+num_gene,np.zeros_like(neg[0]))).T

    kf = KFold(n_splits=nfold, shuffle=True, random_state=randomState)
    fold=1
    prf1List =[] 
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(posIndex),
                                                                            kf.split(negIndex)):
        train_neg_idx_sampling = sampling_from_numpy(train_neg_idx,train_pos_idx.shape[0]*3)
        test_neg_idx_sampling = sampling_from_numpy(test_neg_idx,test_pos_idx.shape[0])
        print('================={}: {}-Cross Validation: Fold {}==================='.format(ms,nfold, fold))
        print(train_pos_idx.shape,train_neg_idx_sampling.shape)
        print(test_pos_idx.shape,test_neg_idx.shape,test_neg_idx_sampling.shape)
        
        xtrain = np.vstack((torch.hstack((features[posIndex[train_pos_idx][:,0]],
                                            features[posIndex[train_pos_idx][:,1]])).detach().numpy(),
                            torch.hstack((features[negIndex[train_neg_idx_sampling][:,0]],
                                            features[negIndex[train_neg_idx_sampling][:,1]])).detach().numpy()))
        ytrain = np.zeros(xtrain.shape[0])
        ytrain[:train_pos_idx.shape[0]] = 1 
        
        xtest = np.vstack((torch.hstack((features[posIndex[test_pos_idx][:,0]],
                                            features[posIndex[test_pos_idx][:,1]])).detach().numpy(),
                        torch.hstack((features[negIndex[test_neg_idx_sampling][:,0]],
                                        features[negIndex[test_neg_idx_sampling][:,1]])).detach().numpy()))
        ytest = np.zeros(xtest.shape[0])
        ytest[:test_pos_idx.shape[0]] = 1
            
        # ===应用最优组合===
        model = ['LGBM', 'RF'] #['LGBM','RF','XGB']
        prf1Dict,dataDict,ensembleModel = ml.model_voting(xtrain,ytrain,xtest,ytest,tag={'fold':fold},random_state=randomState,
                                    logInfo=logInfo,suffix=ms,modelList=list(model))
        prf1List.append(prf1Dict)

        # 测试集以不同比例采样
        dataPackage={'sam':[1]*len(dataDict['ytest']),'ytest':list(dataDict['ytest']),'yprob':list(dataDict['yprob'])} # 以num列表保存每一折的数量，然后以data将所有的结局拼接起来
        for sam in [3,5,10,20]:
            test_neg_idx_sampling = sampling_from_numpy(test_neg_idx,test_pos_idx.shape[0]*sam)
            xtest = np.vstack((torch.hstack((features[posIndex[test_pos_idx][:,0]],
                                            features[posIndex[test_pos_idx][:,1]])).detach().numpy(),
                        torch.hstack((features[negIndex[test_neg_idx_sampling][:,0]],
                                        features[negIndex[test_neg_idx_sampling][:,1]])).detach().numpy()))
            ytest = np.zeros(xtest.shape[0])
            ytest[:test_pos_idx.shape[0]] = 1
            yprob = ensembleModel.predict_proba(xtest)[:, 1]
            
            s = PRF1(ytest,yprob)
            print('sam: {}, microAUC: {}, microAPUR: {}'.format(sam,s['AUC'],s['AUPR']))
            dataPackage['sam'].extend([sam]*len(ytest))
            dataPackage['ytest'].extend(list(ytest))
            dataPackage['yprob'].extend(yprob)
        # save_pickle(dataPackage, logInfo,'0_fold_{}.pkl'.format(fold))
        pd.DataFrame(dataPackage).to_csv('log/ML/seed{}_fold{}_sam{}.csv'.format(args.seed,fold,sam),index=False)
        if fold==1:
            YTEST = dataDict['ytest']
            YPROB = dataDict['yprob']
        else:
            YTEST = np.concatenate((YTEST,dataDict['ytest']))
            YPROB = np.concatenate((YPROB,dataDict['yprob']))

        fold+=1
    macroPRF1 = PRF1(YTEST,YPROB)
    # TODO: 将pickle和csv保存到一个文件夹中，修改args.save_path
    save_pickle({'YTEST':YTEST,'YPROB':YPROB},logInfo,'{}_testProb'.format(ms))
    save_csv(pd.DataFrame(macroPRF1),logInfo,'result_macro_{}'.format(ms))
    save_csv(pd.DataFrame(prf1List),logInfo,'result_'+ms)


def findNew():
    randomState = set_seed(args.seed)

    '''
    1. load data from mat
    2. make data for OpenNE
    3. OpenNE
    4. using AutoEncoder to decompose gene features and disease features
    5. using openNE to get network embedding features
    6. using nfold ml model to calculate the score
    '''
    gene_feat, dis_feat, comb, label = load_data.load_data_from_mat(skip=args.skip_load_data,save=True)
    maskIndex, comb, testSet = load_data.load_findNew(args.maskType)
    print('load data from fineNew')
    # OpenNE TODO: 将OpenNE的超参数放在args中
    # OpenNE TODO: 实现OpenNE切换运行环境自动化运行
    if args.skip_openne:
        print('skip OpenNE!!!')
    else:
        load_data.make_data_for_openne(comb,label)
        openneLog = os.path.join(args.logInfo['logPath'], '{}openne.log'.format(args.logInfo['hour']))
        print('=======running OpenNE========')
        print('this may take a while, and the log will be saved in {}'.format(openneLog))
        os.system('bash OpenNE/src/run_openne.sh > {}'.format(openneLog))
        print('finished OpenNE!')

    if args.skip_ae:
        gdFeature = torch.load('log/geneDisFeatureByAE.tensor')
        print('skip AutoEncoder!!!')
    else:
        print('=======running AutoEncoder========')
        gdFeature = gdFeatureByAE(dis_feat,gene_feat)

    logInfo = args.logInfo

    openNEModel = args.openne_method#['DeepWalk', 'LINE', 'Node2vec', 'HOPE', 'SDNE','GraRep'] # same to run_model.sh , there are some bugs in 'GraRep'

    # ===应用最优组合===
    meth = openNEModel #['SDNE','HOPE] # 
    ms = '_'.join(meth) # list to string, like 'DeepWalk_LINE', aim to save the model name
    print('==============={}=============='.format(ms))
    embeddingFeature = torch.ones(len(comb)).reshape(-1,1) #用于初始化拼接向量，便于后面的拼接
    for m in meth:
        openNEPath = 'OpenNE/results/{}/{}_embeddings.txt'.format(m,m)
        print('load from OpenNE/results/{}/{}_embeddings.txt'.format(m,m))
        embeddingFeatureTemp = featureByOpenNE(openNEPath)
        embeddingFeature = torch.cat((embeddingFeature,embeddingFeatureTemp),dim=1)
    embeddingFeature = embeddingFeature[:,1:] #去掉第一列，因为第一列是1，即表示节点的编号

    print('embeddingFeature shape: {}'.format(embeddingFeature.shape))

    num_gene=12331
    features = torch.hstack((gdFeature, embeddingFeature)) #shape=12331+3215,64+64
    geneF = features[:num_gene,:]
    disF = features[num_gene:,:]

    # train in findNew
    pos = comb[:num_gene,num_gene:].nonzero() # pos.shape: (22054,) --> edgeIndex
    neg = np.where(comb[:num_gene,num_gene:]==0) # neg.shape: (39640211,)
    posIndex = np.vstack((pos[0],pos[1],np.ones_like(pos[0]))).T
    negIndex = np.vstack((neg[0],neg[1],np.zeros_like(neg[0]))).T #因为要从feature中提取节点，所以需要加上num_gene

    # test in findNew 只要找到了边集，传给posIndex_findNew就好了
    pos_findNew = testSet.nonzero() 
    neg_findNew = np.where(testSet==0) 
    posIndex_findNew = np.vstack((pos_findNew[0],pos_findNew[1],np.ones_like(pos_findNew[0]))).T
    negIndex_findNew = np.vstack((neg_findNew[0],neg_findNew[1],np.zeros_like(neg_findNew[0]))).T # 同理，需要加上num_gene


    prf1List =[] 
    macrioPRF1List =[]
    train_pos_idx = posIndex
    train_neg_idx = negIndex
    test_pos_idx = posIndex_findNew
    test_neg_idx = negIndex_findNew

    train_neg_idx_sampling = sampling_from_numpy(train_neg_idx,train_pos_idx.shape[0]*3)
    test_neg_idx_sampling = sampling_from_numpy(test_neg_idx,test_pos_idx.shape[0])
    print('================={}: begin training ==================='.format(ms))
    print(train_pos_idx.shape,train_neg_idx_sampling.shape)
    print(test_pos_idx.shape,test_neg_idx.shape,test_neg_idx_sampling.shape)
    
    xtrain = np.vstack((torch.hstack((geneF[train_pos_idx[:,0]],
                                        disF[train_pos_idx[:,1]])).detach().numpy(),
                        torch.hstack((geneF[train_neg_idx_sampling[:,0]],
                                        disF[train_neg_idx_sampling[:,1]])).detach().numpy()))
    ytrain = np.zeros(xtrain.shape[0])
    ytrain[:train_pos_idx.shape[0]] = 1 
    
    if args.maskType =='gene':
        xtest = np.vstack((torch.hstack((geneF[maskIndex,:][test_pos_idx[:,0]],
                                            disF[test_pos_idx[:,1]])).detach().numpy(),
                        torch.hstack((geneF[maskIndex,:][test_neg_idx_sampling[:,0]],
                                        disF[test_neg_idx_sampling[:,1]])).detach().numpy()))
    elif args.maskType =='dis':
        xtest = np.vstack((torch.hstack((geneF[test_pos_idx[:,0]],
                                            disF[maskIndex,:][test_pos_idx[:,1]])).detach().numpy(),
                        torch.hstack((geneF[test_neg_idx_sampling[:,0]],
                                        disF[maskIndex,:][test_neg_idx_sampling[:,1]])).detach().numpy()))
    ytest = np.zeros(xtest.shape[0])
    ytest[:test_pos_idx.shape[0]] = 1

    print('xtrain shape: {}'.format(xtrain.shape))
    print('ytrain shape: {}'.format(ytrain.shape))
    print('xtest shape: {}'.format(xtest.shape))
    print('ytest shape: {}'.format(ytest.shape))

    # ===应用最优组合===
    model = ['LGBM', 'RF'] #['LGBM','RF','XGB']
    prf1Dict,dataDict,_ = ml.model_voting(xtrain,ytrain,xtest,ytest,random_state=randomState,
                                logInfo=logInfo,suffix=ms,modelList=list(model))
    prf1List.append(prf1Dict)

    YTEST = ytest
    YPROB = dataDict['yprob']

    macroPRF1 = PRF1(YTEST,YPROB)
    macrioPRF1List.append(macroPRF1)

    save_csv(pd.DataFrame({'YTEST':YTEST,'YPROB':YPROB}),logInfo,'ML_prediction_{}_{}'.format(args.mode,args.maskType))
    save_csv(pd.DataFrame(macrioPRF1List),logInfo,'result_macro_'.format(ms))
    save_csv(pd.DataFrame(prf1List),logInfo,'result_'+ms)

def findBest():
    randomState = set_seed(args.seed)
    gene_feat, dis_feat, comb, label = load_data.load_data_from_mat(skip=args.skip_load_data,save=True)

    if args.skip_openne:
        print('skip OpenNE!!!')
    else:
        load_data.make_data_for_openne(comb,label)
        openneLog = os.path.join(args.logInfo['logPath'], '{}openne.log'.format(args.logInfo['hour']))
        print('=======running OpenNE========')
        print('this may take a while, and the log will be saved in {}'.format(openneLog))
        os.system('bash OpenNE/src/run_openne.sh > {}'.format(openneLog))
        print('finished OpenNE!')

    if args.skip_ae:
        gdFeature = torch.load('log/geneDisFeatureByAE.tensor') #if error, try use 
        print('skip AutoEncoder!!!')
    else:
        print('=======running AutoEncoder========')
        gdFeature = gdFeatureByAE(dis_feat,gene_feat)


    logInfo = args.logInfo
    nfold=args.nfold 

    # openNEModel = ['DeepWalk', 'LINE', 'Node2vec', 'HOPE', 'SDNE','GraRep'] # same to run_model.sh , there are some bugs in 'GraRep'
    openNEModel = args.openne_method#['DeepWalk', 'LINE', 'Node2vec', 'HOPE', 'SDNE','GraRep'] # same to run_model.sh , there are some bugs in 'GraRep'
    # compare other openNE model
    # TODO: 在main函数中将test，prob拼接起来，保存并计算结果。

    # ===用于寻找最优组合===
    openNEComb = list(combinations(openNEModel, 1))+list(combinations(openNEModel, 2))+list(combinations(openNEModel, 3))
    for methComb in openNEComb: 
        meth = list(methComb) # tuple to list
        ms = '_'.join(meth) # list to string, like 'DeepWalk_LINE', aim to save the model name
        print('==============={}=============='.format(ms))
        embeddingFeature = torch.ones(len(comb)).reshape(-1,1) #用于初始化拼接向量，便于后面的拼接
        for m in meth:
            openNEPath = 'OpenNE/results/{}/{}_embeddings.txt'.format(m,m)
            print('load from OpenNE/results/{}/{}_embeddings.txt'.format(m,m))
            embeddingFeatureTemp = featureByOpenNE(openNEPath)
            embeddingFeature = torch.cat((embeddingFeature,embeddingFeatureTemp),dim=1)
        embeddingFeature = embeddingFeature[:,1:] #去掉第一列，因为第一列是1，即表示节点的编号

        print('embeddingFeature shape: {}'.format(embeddingFeature.shape))

        features = torch.hstack((gdFeature, embeddingFeature)) #shape=12331+3215,64+64

        num_gene=12331
        pos = comb[:num_gene,num_gene:].nonzero() # pos.shape: (3954,) --> edgeIndex TODO: 查看下pos.shape
        neg = np.where(comb[:num_gene,num_gene:]==0) # neg.shape: (39640211,)

        posIndex = np.vstack((pos[0],pos[1]+num_gene,np.ones_like(pos[0]))).T
        negIndex = np.vstack((neg[0],neg[1]+num_gene,np.zeros_like(neg[0]))).T

        kf = KFold(n_splits=nfold, shuffle=True, random_state=randomState)
        fold=1
        prf1List =[] 
        macrioPRF1List =[]
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(posIndex),
                                                                                kf.split(negIndex)):
            train_neg_idx_sampling = sampling_from_numpy(train_neg_idx,train_pos_idx.shape[0]*3)
            test_neg_idx_sampling = sampling_from_numpy(test_neg_idx,test_pos_idx.shape[0])
            print('================={}: {}-Cross Validation: Fold {}==================='.format(ms,nfold, fold))
            print(train_pos_idx.shape,train_neg_idx_sampling.shape)
            print(test_pos_idx.shape,test_neg_idx.shape,test_neg_idx_sampling.shape)
            
            xtrain = np.vstack((torch.hstack((features[posIndex[train_pos_idx][:,0]],
                                                features[posIndex[train_pos_idx][:,1]])).detach().numpy(),
                                torch.hstack((features[negIndex[train_neg_idx_sampling][:,0]],
                                                features[negIndex[train_neg_idx_sampling][:,1]])).detach().numpy()))
            ytrain = np.zeros(xtrain.shape[0])
            ytrain[:train_pos_idx.shape[0]] = 1 
            
            xtest = np.vstack((torch.hstack((features[posIndex[test_pos_idx][:,0]],
                                                features[posIndex[test_pos_idx][:,1]])).detach().numpy(),
                            torch.hstack((features[negIndex[test_neg_idx_sampling][:,0]],
                                            features[negIndex[test_neg_idx_sampling][:,1]])).detach().numpy()))
            ytest = np.zeros(xtest.shape[0])
            ytest[:test_pos_idx.shape[0]] = 1
                
            # ===用于寻找最优组合===
            modelList = ['LGBM','RF','XGB']
            modelComb = list(combinations(modelList,1))+list(combinations(modelList,2))+list(combinations(modelList,3))
            for model in modelComb:
                prf1Dict,dataDict = ml.model_voting(xtrain,ytrain,xtest,ytest,tag={'fold':fold},random_state=randomState,
                                            logInfo=logInfo,suffix=ms,modelList=list(model))
            prf1List.append(prf1Dict)

            if fold==1:
                YTEST = ytest
                YPROB = dataDict['yprob']
            else:
                YTEST = np.concatenate((YTEST,ytest))
                YPROB = np.concatenate((YPROB,dataDict['yprob']))

            fold+=1
        macroPRF1 = PRF1(YTEST,YPROB)
        macrioPRF1List.append(macroPRF1)
        prf1 = pd.DataFrame(prf1List)
        macroPRF1DF = pd.DataFrame(macrioPRF1List)
        # TODO: 将pickle和csv保存到一个文件夹中，修改args.save_path
        save_pickle({'YTEST':YTEST,'YPROB':YPROB},logInfo,'{}_testProb'.format(ms))
        save_csv(macroPRF1DF,logInfo,'result_macro_'.format(ms))
        save_csv(prf1,logInfo,'result_'+ms)

def test():
    randomState = set_seed(args.seed)

    gene_feat, dis_feat, comb, label = load_data.load_data_from_mat(skip=args.skip_load_data,save=True)

    if args.skip_openne:
        print('skip OpenNE!!!')
    else:
        load_data.make_data_for_openne(comb,label)
        openneLog = os.path.join(args.logInfo['logPath'], '{}openne.log'.format(args.logInfo['hour']))
        print('=======running OpenNE========')
        print('this may take a while, and the log will be saved in {}'.format(openneLog))
        os.system('bash OpenNE/src/run_openne.sh > {}'.format(openneLog))
        print('finished OpenNE!')

    if args.skip_ae:
        gdFeature = torch.load('log/geneDisFeatureByAE.tensor')
        print('skip AutoEncoder!!!')
    else:
        print('=======running AutoEncoder========')
        gdFeature = gdFeatureByAE(dis_feat,gene_feat)


    logInfo = args.logInfo
    nfold=args.nfold 
    openNEModel = args.openne_method#['DeepWalk', 'LINE', 'Node2vec', 'HOPE', 'SDNE','GraRep'] # same to run_model.sh , there are some bugs in 'GraRep'

    # ===应用最优组合===
    meth = openNEModel #['SDNE','HOPE] # 
    ms = '_'.join(meth) # list to string, like 'DeepWalk_LINE', aim to save the model name
    print('==============={}=============='.format(ms))
    embeddingFeature = torch.ones(len(comb)).reshape(-1,1) #用于初始化拼接向量，便于后面的拼接
    for m in meth:
        openNEPath = 'OpenNE/results/{}/{}_embeddings.txt'.format(m,m)
        print('load from OpenNE/results/{}/{}_embeddings.txt'.format(m,m))
        embeddingFeatureTemp = featureByOpenNE(openNEPath)
        embeddingFeature = torch.cat((embeddingFeature,embeddingFeatureTemp),dim=1)
    embeddingFeature = embeddingFeature[:,1:] #去掉第一列，因为第一列是1，即表示节点的编号

    print('embeddingFeature shape: {}'.format(embeddingFeature.shape))

    features = torch.hstack((gdFeature, embeddingFeature)) #shape=12331+3215,64+64

    num_gene=12331
    pos = comb[:num_gene,num_gene:].nonzero() # pos.shape: (3954,) --> edgeIndex TODO: 查看下pos.shape
    neg = np.where(comb[:num_gene,num_gene:]==0) # neg.shape: (39640211,)

    posIndex = np.vstack((pos[0],pos[1]+num_gene,np.ones_like(pos[0]))).T
    negIndex = np.vstack((neg[0],neg[1]+num_gene,np.zeros_like(neg[0]))).T

    kf = KFold(n_splits=nfold, shuffle=True, random_state=randomState)
    fold=1
    prf1List =[] 
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(posIndex),
                                                                            kf.split(negIndex)):
        train_neg_idx_sampling = sampling_from_numpy(train_neg_idx,train_pos_idx.shape[0]*3)
        test_neg_idx_sampling = sampling_from_numpy(test_neg_idx,test_pos_idx.shape[0])
        print('================={}: {}-Cross Validation: Fold {}==================='.format(ms,nfold, fold))
        print(train_pos_idx.shape,train_neg_idx_sampling.shape)
        print(test_pos_idx.shape,test_neg_idx.shape,test_neg_idx_sampling.shape)
        
        xtrain = np.vstack((torch.hstack((features[posIndex[train_pos_idx][:,0]],
                                            features[posIndex[train_pos_idx][:,1]])).detach().numpy(),
                            torch.hstack((features[negIndex[train_neg_idx_sampling][:,0]],
                                            features[negIndex[train_neg_idx_sampling][:,1]])).detach().numpy()))
        ytrain = np.zeros(xtrain.shape[0])
        ytrain[:train_pos_idx.shape[0]] = 1 
        
        xtest = np.vstack((torch.hstack((features[posIndex[test_pos_idx][:,0]],
                                            features[posIndex[test_pos_idx][:,1]])).detach().numpy(),
                        torch.hstack((features[negIndex[test_neg_idx_sampling][:,0]],
                                        features[negIndex[test_neg_idx_sampling][:,1]])).detach().numpy()))
        ytest = np.zeros(xtest.shape[0])
        ytest[:test_pos_idx.shape[0]] = 1
            
        # ===应用最优组合===
        modelList = ['LGBM', 'RF', 'XGB', 'LR', 'NB']
        for model in modelList:
            prf1Dict,dataDict,ensembleModel = ml.model_voting(xtrain,ytrain,xtest,ytest,tag={'fold':fold},random_state=randomState,
                                        logInfo=logInfo,suffix=ms,modelList=[model])
            prf1List.append(prf1Dict)
        fold+=1
    save_csv(pd.DataFrame(prf1List),logInfo,'Find_ML_PRF1_{}'.format(args.seed))

if __name__=='__main__':
    print('========={}========{}========{}========'.format(args.mode,args.mode,args.mode))
    if args.mode=='train':
        train()
    elif args.mode=='findBest':
        findBest()
    elif args.mode=='findNew':
        findNew()
    elif args.mode=='test':
        test()
    else:
        print('mode error, expected train or findNew, but got {}'.format(args.mode))
        exit()
    print('main has been finished')