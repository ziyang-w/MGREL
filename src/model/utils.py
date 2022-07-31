import scipy.sparse as sp
import numpy as np
import os
import pickle
import torch
import random 
import pandas as pd
#==========wzyFunc.dataPrep==========
def set_seed(seed=42):
    '''
    description: Set random seed. eg: random_state = ml.set_seed(42)
    param {int} seed: Random seed to use
    return {int} seed
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed

def make_logInfo(fileName:str, filePath:str,savePath:str='') -> dict:
    '''
    description: 根据数据集fileName, 数据集路径filePath, 构造LogInfo字典
                 主要目的为了后续数据分析文件的保存
    param {str} fileName: 数据集文件名, 包含后缀
    param {str} filePath: 数据集路径
    return  logInfo: {'logPath','plotPath','date','hour','fileName','filePath'}
    '''
    from datetime import datetime
    startDate = datetime.now().strftime('%m-%d')
    hour = datetime.now().strftime('%H_')
    if savePath!='':
        logPath = os.path.join(filePath, 'log', fileName.split('.')[0], savePath)
    else:
        logPath = os.path.join(filePath, 'log', fileName.split('.')[0], startDate)

    if not os.path.exists(logPath):
        os.makedirs(logPath)
        os.makedirs(os.path.join(logPath, 'plot'))
        os.makedirs(os.path.join(logPath, 'pickle'))
    logInfo = {'logPath': logPath,
               'plotPath':os.path.join(logPath, 'plot'),
               'picklePath':os.path.join(logPath, 'pickle'),
               'date':startDate,
               'hour': hour,
               'fileName': fileName,
               'filePath': filePath}
    return logInfo


def save_pickle(variable:any, logInfo:dict, suffix:str, fileName=False):
    '''
    description:  将结果保存到对应的log目录下, 
                  eg: 'filePath\\log\\fileName\\mm-dd\\hh_fileName_suffix.csv'
                  Tips: 在调用时, 一般只加一次后缀, 即在suffix参数中尽量用驼峰法命名, 而不包含'_', 方便后续查找
    param {pd} df: 要保存的DataFrame
    param {dict} logInfo: <- wzyFunc.make_logInfo()
    param {str} suffix: 想要添加的后缀名, 一般应用驼峰法命名, 而不使用'_'来进行分隔
    param { None | True } fileName: 在保存的文件名中是否加入当前分析数据集文件名后缀logInfo['fileName']
    return None
    '''
    suffix += '.pkl'
    if bool(fileName):
        tPath = os.path.join(logInfo['logPath'],
                             str(logInfo['hour'])+logInfo['fileName'].split('.')[0]+'_'+suffix)
    else:
        tPath = os.path.join(logInfo['picklePath'], str(logInfo['hour'])+suffix)
    pickle.dump(variable, open(tPath, 'wb'))
    print('file has been saved in : %s' % tPath)

def save_csv(df:pd.DataFrame, logInfo:dict, suffix:str, fileName=False):
    '''
    description:  将结果保存到对应的log目录下, 
                  eg: 'filePath\\log\\fileName\\mm-dd\\hh_fileName_suffix.csv'
                  Tips: 在调用时, 一般只加一次后缀, 即在suffix参数中尽量用驼峰法命名, 而不包含'_', 方便后续查找
    param {pd} df: 要保存的DataFrame
    param {dict} logInfo: <- wzyFunc.make_logInfo()
    param {str} suffix: 想要添加的后缀名, 一般应用驼峰法命名, 而不使用'_'来进行分隔
    param { None | True } fileName: 在保存的文件名中是否加入当前分析数据集文件名后缀logInfo['fileName']
    return None
    '''
    suffix += '.csv'
    if bool(fileName):
        tPath = os.path.join(logInfo['logPath'],
                             str(logInfo['hour'])+logInfo['fileName'].split('.')[0]+'_'+suffix)
    else:
        tPath = os.path.join(logInfo['logPath'], str(logInfo['hour'])+suffix)
    df.to_csv(tPath,
              index=False,
              encoding='utf-8-sig')
    print('file has been saved in : %s' % tPath)


#==========wzyFunc.modelEvaluation==========
def PRF1(real_score:np.array, predict_score:np.array)->dict:
    """Calculate the performance metrics.
    Resource code is acquired from:
    Yu Z, Huang F, Zhao X et al.
     Predicting drug-disease associations through layer attention graph convolutional network,
     Brief Bioinform 2021;22.

    Parameters
    ----------
    real_score: true labels, ytest
    predict_score: model predictions, yprob

    Return
    ---------
    AUC, AUPR, Accuracy, F1-Score, Precision, Recall, Specificity
    """
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    cm = {
    'AUC':auc[0, 0],
    'AUPR':aupr[0, 0],
    'F1':f1_score,
    'A':accuracy,
    'R(Sen)(TPR)':recall,
    'Spec(TNR)':specificity,
    'P(PPV)':precision,
    'YI':recall + specificity - 1,
    'threshold':thresholds.T[max_index][0][0].item()}
    return cm






#==========LUPI_RGCN==========
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
    
def network_edge_threshold(network_adj, threshold):
    edge_tmp, edge_value, shape_tmp = sparse_to_tuple(network_adj)
    preserved_edge_index = np.where(edge_value>threshold)[0]
    preserved_network = sp.csr_matrix(
        (edge_value[preserved_edge_index], 
        (edge_tmp[preserved_edge_index,0], edge_tmp[preserved_edge_index, 1])),
        shape=shape_tmp)
    return preserved_network