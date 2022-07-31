from itertools import combinations, islice
from sklearn.model_selection import KFold
from args import args
import pandas as pd
import numpy as np
import torch
import os
import load_data
import machineLearning as ml
from utils import set_seed, save_csv,PRF1,save_pickle
from feature_extration import gdFeatureByAE, featureByOpenNE,sampling_from_numpy


def sen():
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
        # gdFeature = torch.load('log/geneDisFeatureByAE.tensor')
        gdFeature = torch.load('log/geneDisFeatureByAE_{}.tensor'.format(str(args.ae_hidden)))
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
        openNEPath = 'OpenNE/results_{}/{}/{}_embeddings.txt'.format(args.openneDim,m,m)
        print('load from OpenNE/results_{}/{}/{}_embeddings.txt'.format(args.openneDim,m,m))
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
        model = ['LGBM', 'RF']#, 'XGB', 'LR', 'NB']
        prf1Dict,dataDict,_ = ml.model_voting(xtrain,ytrain,xtest,ytest,tag={'fold':fold},random_state=randomState,
                                    logInfo=logInfo,suffix=ms,modelList=model)
        prf1List.append(prf1Dict)

        if fold==1:
            YTEST = dataDict['ytest']
            YPROB = dataDict['yprob']
        else:
            YTEST = np.concatenate((YTEST,dataDict['ytest']))
            YPROB = np.concatenate((YPROB,dataDict['yprob']))

        fold+=1
    save_csv(pd.DataFrame({'YTEST':YTEST,'YPROB':YPROB}),logInfo,'sen_{}_ae_{}_{}'.format(args.openneDim,str(args.ae_hidden),args.seed))
    save_csv(pd.DataFrame(prf1List),logInfo,'senPRF1_{}_{}_{}'.format(args.openneDim,str(args.ae_hidden),args.seed))


def caseStudy():
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
     #TODO: 继续完善修改代码
    num_gene=12331
    features = torch.hstack((gdFeature, embeddingFeature)) #shape=12331+3215,64+64

    num_gene=12331
    pos = comb[:num_gene,num_gene:].nonzero() # pos.shape: (3954,) --> edgeIndex TODO: 查看下pos.shape
    neg = np.where(comb[:num_gene,num_gene:]==0) # neg.shape: (39640211,)

    posIndex = np.vstack((pos[0],pos[1]+num_gene,np.ones_like(pos[0]))).T
    negIndex = np.vstack((neg[0],neg[1]+num_gene,np.zeros_like(neg[0]))).T #因为要从feature中提取节点，所以需要加上num_gene


    train_neg_idx_sampling = sampling_from_numpy(negIndex,posIndex.shape[0]*3)
    # test_neg_idx_sampling = sampling_from_numpy(test_neg_idx,test_pos_idx.shape[0])
    print('================={}: begin training ==================='.format(ms))
    print(posIndex.shape,train_neg_idx_sampling.shape)
    # print(test_pos_idx.shape,test_neg_idx.shape,test_neg_idx_sampling.shape)
    
    xtrain = np.vstack((torch.hstack((features[posIndex[:,0]],
                                        features[posIndex[:,1]])).detach().numpy(),
                        torch.hstack((features[train_neg_idx_sampling[:,0]],
                                        features[train_neg_idx_sampling[:,1]])).detach().numpy()))
    ytrain = np.zeros(xtrain.shape[0])
    ytrain[:posIndex.shape[0]] = 1 
    
    # 测试集为全集
    xtest = np.vstack((torch.hstack((features[posIndex[:,0]],
                                            features[posIndex[:,1]])).detach().numpy(),
                        torch.hstack((features[negIndex[:,0]],
                                        features[negIndex[:,1]])).detach().numpy()))
    ytest = np.zeros(xtest.shape[0])
    ytest[:posIndex.shape[0]] = 1
    print(xtrain.shape)
    print(ytrain.shape)
    print(xtest.shape)
    print(ytest.shape)
    del features,train_neg_idx_sampling,pos,neg

    # ===应用最优组合===
    modelList = ['LGBM', 'RF'] #['LGBM','RF','XGB']

    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    rfVote = RandomForestClassifier(n_estimators=100,
                                class_weight='balanced',
                                random_state=randomState)

    from lightgbm import LGBMClassifier as LGBMC
    lgbmVote = LGBMC(num_leaves=60,
                        learning_rate=0.05,
                        n_estimators=100,
                        class_weight='balanced',
                        random_state=randomState)    

    estList = {'RF':('RF',rfVote),'LGBM':('LGBM',lgbmVote)}
    estList = [estList.get(model) for model in modelList] # 选择模型列表中的模型
    # print(estList)

    ensembleModel = VotingClassifier(estimators=estList,voting='soft').fit(xtrain,ytrain)
    ypre = ensembleModel.predict(xtest)
    yprob = ensembleModel.predict_proba(xtest)[:, 1]
    del xtest
    caseStudy = np.hstack((np.vstack((posIndex, negIndex)), np.vstack((ytest, yprob, ypre)).T))

    save_csv(pd.DataFrame(caseStudy,columns=['gene','disease','label','ytest','yprob','ypred']),
             logInfo,'caseStudy_result_{}'.format(args.seed))

def ablation():
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

    if args.ablationType =='ae':
        features = gdFeature #shape=12331+3215,64+64
    elif args.ablationType =='openne':
        features = embeddingFeature
    print('ablation type: {}'.format(args.ablationType))
    print('features shape: {}'.format(features.shape))

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
        model = ['LGBM', 'RF']
        prf1Dict,dataDict,ensembleModel = ml.model_voting(xtrain,ytrain,xtest,ytest,tag={'fold':fold},random_state=randomState,
                                        logInfo=logInfo,suffix=ms,modelList=model)
        prf1List.append(prf1Dict)
        fold+=1
    save_csv(pd.DataFrame(prf1List),logInfo,'ablation_{}_{}'.format(args.ablationType,args.seed))

# 生成一个迭代器，用于将大的numpy数组分割成小的numpy数组
def split_for_negIndex(negIndex,batch_size):
    negIndex_iter = iter(negIndex)
    while True:
        batch_negIndex = np.fromiter(islice(negIndex_iter,batch_size),dtype=np.int32)
        if batch_negIndex.size == 0:
            break
        yield batch_negIndex

if __name__=='__main__':
    print('========={}========{}========{}========'.format(args.mode,args.mode,args.mode))
    if args.mode=='sen':
        sen()
    elif args.mode=='caseStudy':
        caseStudy()
    elif args.mode=='ablation':
        ablation()
    else:
        print('mode error, expected train or findNew, but got {}'.format(args.mode))
        exit()
    print('main has been finished')