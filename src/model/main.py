from sklearn.model_selection import KFold
from args import args
import pandas as pd
import numpy as np
import torch
import os

import load_data
import machineLearning as ml
from utils import make_logInfo, set_seed, save_csv
from feature_extration import AutoEncoder, gdFeatureByAE, featureByOpenNE,sampling_from_numpy

def main():
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

    load_data.make_data_for_openne(comb,label)

    # OpenNE TODO: 将OpenNE的超参数放在args中
    if args.skip_openne:
        print('skip OpenNE!!!')
    else:
        openneLog = os.path.join(args.logInfo['logPath'], '{}openne.log'.format(args.logInfo['hour']))
        print('=======running OpenNE========')
        print('this may take a while, and the log will be saved in {}'.format(openneLog))
        os.system('sh OpenNE/src/run_openne.sh > {}'.format(openneLog))
        print('finished OpenNE!')

    if args.skip_ae:
        gdFeature = torch.load('log/geneDisFeatureByAE.tensor')
        print('skip AutoEncoder!!!')
    else:
        print('=======running AutoEncoder========')
        gdFeature = gdFeatureByAE(dis_feat,gene_feat)

    openNEModel = ['DeepWalk', 'LINE', 'Node2vec', 'HOPE', 'SDNE'] # same to run_model.sh , there are some bugs in 'GraRep'

    logInfo = args.logInfo
    nfold=args.nfold 

    # compare other openNE model
    for m in openNEModel:
        print('==============={}=============='.format(m))
        openNEPath = 'OpenNE/results/{}/{}_embeddings.txt'.format(m,m)
        embeddingFeature = featureByOpenNE(openNEPath)
        features = torch.hstack((gdFeature, embeddingFeature)) #shape=12331+3215,64+64

        num_gene=12331
        pos = comb[:num_gene,num_gene:].nonzero() # pos.shape: (3954,) --> edgeIndex
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
            print('================={}: {}-Cross Validation: Fold {}==================='.format(m,nfold, fold))
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
                
            prf1Dict = ml.muti_model(xtrain,ytrain,xtest,ytest,tag={'fold':fold},random_state=randomState,logInfo=logInfo,suffix=m)
            prf1List+=prf1Dict

            fold+=1
        prf1 = pd.DataFrame(prf1List)
        save_csv(prf1,logInfo,'result_'+m)
if __name__=='__main__':
    main()
    print('main has been finished')