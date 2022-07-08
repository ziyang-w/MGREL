import numpy as np

from utils import save_pickle, PRF1


def rf(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, suffix='',tag=False,logInfo=False)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用
    '''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve
    d = {'model':'RF','suffix':'resultRF','l':'Random Forest'}
    model = RandomForestClassifier(n_estimators=100,
                                   class_weight='balanced',
                                   random_state=random_state).fit(xtrain, ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    # prf1Dict = PRF1(np.array(ytest), ypre, yprob)
    prf1Dict = PRF1(np.array(ytest), yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['model'],prf1Dict['AUC'],prf1Dict['AUPR']))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
        if 'fold' in tag.keys() and bool(logInfo): # 如果传入fold字段, 则将保存模型的预测数据
            dataDict = {'ytest':ytest, 'ypre':ypre, 'yprob':yprob}
            save_pickle(variable=dataDict, suffix=suffix+'_{}_fold{}'.format(d['model'],tag['fold']),logInfo=logInfo)
    
    return prf1Dict

def lgbm(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, suffix='',tag=False, logInfo=False)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用
    '''
    from lightgbm import LGBMClassifier as LGBMC
    from sklearn.metrics import roc_curve
    d = {'model':'LGBM','suffix':'resultLGBM','l':'LightGBM'}
    model = LGBMC(num_leaves=60,
                      learning_rate=0.05,
                      n_estimators=100,
                      class_weight='balanced',
                      random_state=random_state).fit(xtrain, ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    # prf1Dict = PRF1(np.array(ytest), ypre, yprob)
    prf1Dict = PRF1(np.array(ytest), yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['model'],prf1Dict['AUC'],prf1Dict['AUPR']))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
        if 'fold' in tag.keys() and bool(logInfo): # 如果传入fold字段, 则将保存模型的预测数据
            dataDict = {'ytest':ytest, 'ypre':ypre, 'yprob':yprob}
            save_pickle(variable=dataDict, suffix=suffix+'_{}_fold{}'.format(d['model'],tag['fold']),logInfo=logInfo)
    return prf1Dict

def xgb(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, suffix='', tag=False,logInfo=False)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用
    只有在tag字段中传入包含fold键值对和logInfo, 才会保存模型的预测数据, 即ytest, ypre, yprob
    '''
    from xgboost import XGBRFClassifier as XGBC
    from sklearn.metrics import roc_curve
    d = {'model':'XGB','suffix':'resultXGB','l':'XGBoost'}
    model = XGBC(n_estimators =100,
                         random_state=random_state,
                         learning_rate=0.1,
                         booster='gbtree',
                         objective='reg:logistic'
                        #  is_unbalance=True,
                        #  scale_pos_weight=len(ytest)/sum(ytrain)
                         #silent=False
                    ).fit(xtrain,ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    # prf1Dict = PRF1(np.array(ytest), ypre, yprob)
    prf1Dict = PRF1(np.array(ytest), yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['model'],prf1Dict['AUC'],prf1Dict['AUPR']))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
        if 'fold' in tag.keys() and bool(logInfo): # 如果传入fold字段, 则将保存模型的预测数据
            dataDict = {'ytest':ytest, 'ypre':ypre, 'yprob':yprob}
            save_pickle(variable=dataDict, suffix=suffix+'_{}_fold{}'.format(d['model'],tag['fold']),logInfo=logInfo)
    return prf1Dict

def lr(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, suffix='',tag=False,logInfo=False)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用
    '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve
    d = {'model':'LR','suffix':'resultLR','l':'Logistic Regression'}
    model = LogisticRegression(max_iter=1000, class_weight='auto').fit(xtrain,ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    # prf1Dict = PRF1(np.array(ytest), ypre, yprob)
    prf1Dict = PRF1(np.array(ytest), yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['model'],prf1Dict['AUC'],prf1Dict['AUPR']))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
        if 'fold' in tag.keys() and bool(logInfo): # 如果传入fold字段, 则将保存模型的预测数据
            dataDict = {'ytest':ytest, 'ypre':ypre, 'yprob':yprob}
            save_pickle(variable=dataDict, suffix=suffix+'_{}_fold{}'.format(d['model'],tag['fold']),logInfo=logInfo)
            
    
    return prf1Dict


def muti_model(xtrain:np.array,
                ytrain:np.array,
                xtest:np.array,
                ytest:np.array,
                tag:dict,
                random_state:int,suffix='',logInfo=False) -> dict:
    '''
    description: 通过传入划分好的测试集和训练集, 对数据进行多模型建模和验证
    param {np.array} xtrain, ytrain, xtest, ytest <- skearn.model_selection.KFold().split(X,Y) !! 注意顺序不同 !!
    param {dict | None} tag: 自定义传入结果字典的标签
                             若要在交叉验证中调用并且保存ytest, ypre, yprob, 则需要传入tag={'fold':k}
    param {dict | None} logInfo: <- wzyFunc.dataPrep.make_logInfo()
    param {int} random_state: <- wzyFunc.machineLearning.set_seed()
    return {dict} prf1List

    --------example:-----------
    random_state = ml.set_seed(42)
    nfold=5
    # skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)
    fold=1
    prf1List=[]
    for xtrain,xtest,ytrain,ytest in zip(kf.split(X),kf.split(Y)):
        print('=========={}-Cross Validation: Fold {}==========='.format(nfold,fold))
        # OTHER CODE

        prf1Dict = muti_model(xtrain,ytrain,xtest,ytest,tag={'fold':fold},random_state=random_state,logInfo=logInfo)

        print('AUC: {:.4}, AUPR: {:.4f}'.format(prf1Dict['AUC'],prf1Dict['AUPR']))
        prf1List.append(prf1Dict)
    prf1DF = pd.DataFrame(prf1Dict)
    group = prf1DF.groupby('model')
    kmodel = {
        'TPR_MEAN': group.mean()['R(Sen)(TPR)'],
        'TPR_STD': group.std()['R(Sen)(TPR)'],
        'L': list(group.mean().index)
    }
    plot_ROC_kmodel(kmodel,logInfo)
    '''
    prf1List=[]
    prf1List.append(rf(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix))
    prf1List.append(lgbm(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix))
    prf1List.append(xgb(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix))
    return prf1List