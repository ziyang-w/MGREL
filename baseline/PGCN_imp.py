import os
import dgl
import h5py
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from model_zoo import PGCN
from utils import remove_graph, get_metrics_auc, set_seed


DEVICE = '1'
NUM_FOLD = 5
BATCH_SIZE = 512
LR = 0.001
EPOCH = 300

if 'PGCN_result' not in os.listdir(path='..'):
    os.mkdir('../PGCN_result')

if DEVICE != 'cpu':
    print('Training on GPU')
    DEVICE = torch.device('cuda:{}'.format(DEVICE))
else:
    print('Training on CPU')
    DEVICE = torch.device('cpu')

for SEED in range(10):
    # Load edge data
    g_g = sp.load_npz('../data/gg.npz')  # gene-gene
    g_d = sp.load_npz('../data/gd.npz')  # gene-disease
    d_d = sp.load_npz('../data/dd.npz')  # disease-disease
    g_g = g_g.toarray()
    g_d = g_d.toarray()
    d_d = d_d.toarray()
    # Load initial node features
    num_gene = 12331
    num_disease = 3215
    gene_phenes_path = '../data/genes_phenes.mat'
    f = h5py.File(gene_phenes_path, 'r')

    # gene features
    gene_feature_path = '../data/GeneFeatures.mat'
    f_gene_feature = h5py.File(gene_feature_path, 'r')
    gene_feature_exp = np.array(f_gene_feature['GeneFeatures'])
    gene_feature_exp = np.transpose(gene_feature_exp)
    # PCA is used to reduce the dimensionality.
    pca = PCA(n_components=100)
    gene_feature_exp = pca.fit_transform(gene_feature_exp)

    row_list = [3215, 1137, 744, 2503, 1143, 324, 1188, 4662, 1243]
    gene_feature_list_other_spe = list()
    for i in range(1, 9):
        dg_ref = f['GenePhene'][i][0]
        disease_gene_adj_tmp = sp.csc_matrix((np.array(f[dg_ref]['data']),
                                              np.array(f[dg_ref]['ir']),
                                              np.array(f[dg_ref]['jc'])),
                                             shape=(12331, row_list[i]))
        gene_feature_list_other_spe.append(disease_gene_adj_tmp)

    gene_feat = sp.hstack(gene_feature_list_other_spe +
                          [gene_feature_exp]).tocsr().toarray()

    # disease features
    disease_network_adj = sp.csc_matrix((np.array(f['PhenotypeSimilarities']['data']),
                                         np.array(f['PhenotypeSimilarities']['ir']),
                                         np.array(f['PhenotypeSimilarities']['jc'])),
                                        shape=(3215, 3215))
    disease_network_adj = disease_network_adj.tocsr().toarray()

    disease_tfidf_path = '../data/clinicalfeatures_tfidf.mat'
    f_disease_tfidf = h5py.File(disease_tfidf_path, "r")
    disease_tfidf = np.array(f_disease_tfidf['F'])
    disease_tfidf = np.transpose(disease_tfidf)
    disease_tfidf = sp.csc_matrix(disease_tfidf).tocsr().toarray()
    dis_feat = np.hstack((disease_network_adj, disease_tfidf))

    # Construct graph
    g_g_asso = np.array(np.where(g_g == 1)).T
    g_d_asso = np.array(np.where(g_d == 1)).T
    d_d_asso = np.array(np.where(d_d == 1)).T
    non_link_gene = set(list(range(12331))) - (
            set(np.unique(g_g_asso)) | set(np.unique(g_d_asso[:, 0])))
    self_g_g_loop = np.array([[idx, idx]
                              for idx in non_link_gene])
    g_g_asso = np.vstack((g_g_asso, self_g_g_loop))
    graph_data = {
        ('gene', 'gene_gene', 'gene'): (torch.tensor(g_g_asso[:, 0]),
                                        torch.tensor(g_g_asso[:, 1])),
        ('gene', 'gene_disease', 'disease'): (torch.tensor(g_d_asso[:, 0]),
                                              torch.tensor(g_d_asso[:, 1])),
        ('disease', 'disease_gene', 'gene'): (torch.tensor(g_d_asso[:, 1]),
                                              torch.tensor(g_d_asso[:, 0])),
        ('disease', 'disease_disease', 'disease'): (torch.tensor(d_d_asso[:, 0]),
                                                    torch.tensor(d_d_asso[:, 1]))
    }
    g = dgl.heterograph(graph_data)

    set_seed(SEED)
    data = torch.tensor(g_d_asso).to(DEVICE)
    label = torch.tensor(np.ones(data.shape[0])).to(DEVICE)

    kf = StratifiedKFold(NUM_FOLD, shuffle=True, random_state=SEED)
    fold = 1
    # pred_result, pred_label = [], []
    pred_result = {1: [], 3: [], 5: [], 10: [], 20: []}
    pred_label = {1: [], 3: [], 5: [], 10: [], 20: []}
    for (train_idx, val_idx) in kf.split(data.cpu().numpy(), label.cpu().numpy()):
        print('{}-Fold Cross Validation: Fold {}'.format(NUM_FOLD, fold))
        train_data = data[train_idx]
        train_label = label[train_idx]
        val_data = data[val_idx]
        # print(val_data)
        val_label = label[val_idx]
        val_gene_id = [datapoint[0].item() for datapoint in val_data]
        val_disease_id = [datapoint[-1].item() for datapoint in val_data]
        dda_idx = torch.where(val_label == 1)[0].cpu().numpy()
        val_dda_drugid = np.array(val_gene_id)[dda_idx]
        val_dda_disid = np.array(val_disease_id)[dda_idx]
        g_train = g
        g_train = remove_graph(g_train, val_dda_drugid.tolist(), val_dda_disid.tolist()).to(DEVICE)
        feature = {'gene': torch.from_numpy(gene_feat).to(torch.float32).to(DEVICE),
                   'disease': torch.from_numpy(dis_feat).to(torch.float32).to(DEVICE)}
        train_loader = DataLoader(TensorDataset(train_data, train_label), BATCH_SIZE,
                                  shuffle=True, drop_last=False)
        model = PGCN(in_feats=[feature['gene'].shape[1],
                               feature['disease'].shape[1]],
                     rel_names=g.etypes)
        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.BCEWithLogitsLoss()
        train_neg = []
        for epoch in range(1, EPOCH + 1):
            total_loss = 0
            pred_train, label_train = torch.zeros(train_label.shape).to(DEVICE), \
                                      torch.zeros(train_label.shape).to(DEVICE)
            trainPred, trainLabel = [], []

            for i, data_ in enumerate(train_loader):
                x_train, y_train = data_[0].to(DEVICE), data_[1].to(DEVICE)
                neg_gene = torch.randint(0, g_train.num_nodes('gene'), (len(x_train),)).to(DEVICE)
                # neg_dis = torch.randint(0, g_train.num_nodes('disease'), (len(x_train),)
                x_train_neg = torch.stack((neg_gene, x_train[:, 1])).T
                # train_neg.extend(x_train_neg.detach().cpu().numpy().tolist())
                neg_label = torch.zeros(y_train.shape).to(DEVICE)
                X_train = torch.concat([x_train, x_train_neg])
                X_train = torch.tensor([(i.item() - 1) * g_d.shape[1] + j
                                        for (i, j) in X_train])
                Y_train = torch.concat([y_train, neg_label])

                model.train()
                pred = model(g_train, feature).flatten()[X_train]
                pred_score = torch.sigmoid(pred)

                optimizer.zero_grad()
                loss = criterion(pred, Y_train)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() / len(train_loader)
                trainPred.extend(pred.detach().cpu().numpy())
                trainLabel.extend(Y_train.detach().cpu().numpy())
            # AUC_train, AUPR_train = get_metrics_auc(label_train.cpu().detach().numpy(),
            #                                         pred_train.cpu().detach().numpy())
            AUC, AUPR = get_metrics_auc(trainLabel, trainPred)
            print('Epoch {} Loss: {:.5f}; Train: AUC {:.3f}, AUPR {:.3f}'.format(epoch, total_loss,
                                                                                 AUC, AUPR))
            del pred, pred_score, trainPred, trainLabel, pred_train, label_train
        model.eval()
    #     g_d_neg_asso = np.array(np.where(g_d == 0)).T
    #     g_d_neg_asso = np.random.permutation(g_d_neg_asso)
    #     x_val_neg = g_d_neg_asso[len(val_label) * (fold - 1): len(val_label) * fold]
    #     neg_label = torch.zeros(val_label.shape).to(DEVICE)
    #     val_data = torch.concat([val_data, torch.tensor(x_val_neg).to(DEVICE)])
    #     val_data = torch.tensor([(i.item() - 1) * g_d.shape[1] + j
    #                              for (i, j) in val_data])
    #     val_label = torch.concat([val_label, neg_label]).detach().cpu().numpy()
    #     val_pred = torch.sigmoid(model(g_train, feature).flatten()[val_data]).detach().cpu().numpy()
    #     print('-' * 20 + '+' * 20 + '-' * 20)
    #     AUC, AUPR = get_metrics_auc(val_label, val_pred)
    #     print('Overall Val: AUC {:.3f}, AUPR {:.3f}'.format(AUC, AUPR))
    #     print('-' * 20 + '+' * 20 + '-' * 20)
    #     pred_result.extend(val_pred)
    #     pred_label.extend(val_label)
    #     fold += 1
    #     del g_train
    # pd.DataFrame(np.array([pred_result, pred_label]).T,
    #              columns=['Predict', 'Label']).to_csv('RGCN_predictions_{}.csv'.format(SEED),
    #                                                   index=False, header=False)

        g_d_neg_asso = np.array(np.where(g_d == 0)).T
        g_d_neg_asso = np.random.permutation(g_d_neg_asso)
        for ratio in [1, 3, 5, 10, 20]:
            x_val_neg = g_d_neg_asso[len(val_label) * ratio * (fold - 1): len(val_label) * ratio * fold]
            neg_label = torch.zeros(len(val_label) * ratio).to(DEVICE)
            val_data_r = torch.concat([val_data, torch.tensor(x_val_neg).to(DEVICE)])
            val_data_r = torch.tensor([(i.item() - 1) * g_d.shape[1] + j
                                       for (i, j) in val_data_r])
            val_label_r = torch.concat([val_label, neg_label]).detach().cpu().numpy()
            val_pred = torch.sigmoid(model(g_train, feature).flatten()[val_data_r]).detach().cpu().numpy()
            print('-' * 20 + '+' * 20 + '-' * 20)
            AUC, AUPR = get_metrics_auc(val_label_r, val_pred)
            print('Ratio {}:1 Overall Val: AUC {:.3f}, AUPR {:.3f}'.format(ratio, AUC, AUPR))
            print('-' * 20 + '+' * 20 + '-' * 20)
            pred_result[ratio].extend(val_pred)
            pred_label[ratio].extend(val_label_r)
        fold += 1
        del g_train
    for ratio in [1, 3, 5, 10, 20]:
        pd.DataFrame(np.array([pred_result[ratio], pred_label[ratio]]).T,
                     columns=['Predict', 'Label']).to_csv('../PGCN_result/PGCN_predictions_ratio{}_{}.csv'.format(ratio,
                                                                                                                  SEED),
                                                          index=False)

