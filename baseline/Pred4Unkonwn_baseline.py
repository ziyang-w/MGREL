import os
import random
import dgl
import h5py
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from model_zoo import PGCN, HAN, HGT, HeteroRGCN
from utils import remove_graph, get_metrics_auc, set_seed


def Model(model_name, in_feats=None, rel_names=None,
          meta_paths=None, node_dict=None, edge_dict=None):
    if model_name == 'RGCN':
        return HeteroRGCN(in_feats=in_feats, rel_names=rel_names)
    elif model_name == 'PGCN':
        return PGCN(in_feats=in_feats, rel_names=rel_names)
    elif model_name == 'HAN':
        return HAN(in_feats=in_feats, meta_paths=meta_paths)
    elif model_name == 'HGT':
        return HGT(in_feats=in_feats, node_dict=node_dict, edge_dict=edge_dict)


DEVICE = '2'

Model_name = ['PGCN', 'RGCN', 'HGT', 'HAN']
SEED = 0
NUM_FOLD = 5
BATCH_SIZE = 512
LR = 0.001
EPOCH = 300
if DEVICE != 'cpu':
    print('Training on GPU')
    DEVICE = torch.device('cuda:{}'.format(DEVICE))
else:
    print('Training on CPU')
    DEVICE = torch.device('cpu')
for model_name in Model_name:
    for SEED in range(1, 10):
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

        val_g_id = np.load('../data/gd_findNew_mask.npy')
        data_split = np.array(['train' if i[0] not in val_g_id else 'val'
                               for i in g_d_asso])
        train_pos = g_d_asso[np.where(data_split == 'train')]
        val_pos = g_d_asso[np.where(data_split == 'val')]
        train_g_id = [i for i in range(g_d.shape[0]) if i not in val_g_id]
        g_d_train, g_d_val = g_d[train_g_id], g_d[val_g_id]
        train_neg = np.random.permutation(np.array(np.where(g_d_train == 0)).T)[: len(train_pos)]
        val_neg = np.random.permutation(np.array(np.where(g_d_val == 0)).T)[: len(val_pos)]

        # val_d_id = np.load('../data/dis_gd_findNew_mask.npy')
        # data_split = np.array(['train' if i[-1] not in val_d_id else 'val'
        #                        for i in g_d_asso])
        # train_pos = g_d_asso[np.where(data_split == 'train')]
        # val_pos = g_d_asso[np.where(data_split == 'val')]
        # train_d_id = [i for i in range(g_d.shape[1]) if i not in val_d_id]
        # g_d_train, g_d_val = g_d.T[train_d_id].T, g_d.T[val_d_id].T
        # train_neg = np.random.permutation(np.array(np.where(g_d_train == 0)).T)[: len(train_pos)]
        # val_neg = np.random.permutation(np.array(np.where(g_d_val == 0)).T)[: len(val_pos)]

        val_dda_drugid = val_pos[:, 0]
        val_dda_disid = val_pos[:, 1]
        g_train = g
        g_train = remove_graph(g_train, val_dda_drugid.tolist(), val_dda_disid.tolist()).to(DEVICE)
        feature = {'gene': torch.from_numpy(gene_feat).to(torch.float32).to(DEVICE),
                   'disease': torch.from_numpy(dis_feat).to(torch.float32).to(DEVICE)}
        train_data, val_data = torch.tensor(np.vstack((train_pos, train_neg))).to(DEVICE), \
                               torch.tensor(np.vstack((val_pos, val_neg))).to(DEVICE)
        train_label = torch.concat([torch.ones(len(train_pos)), torch.zeros(len(train_neg))])
        val_label = torch.concat([torch.ones(len(val_pos)), torch.zeros(len(val_neg))])
        train_loader = DataLoader(TensorDataset(train_data, train_label), BATCH_SIZE,
                                  shuffle=True, drop_last=False)
        node_dict = {}
        edge_dict = {}
        for ntype in g_train.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in g_train.etypes:
            edge_dict[etype] = len(edge_dict)
        model = Model(model_name,
                      in_feats=[feature['gene'].shape[1],
                                feature['disease'].shape[1]],
                      rel_names=g.etypes,
                      meta_paths=[['gene_disease', 'disease_gene'],
                                  ['disease_gene', 'gene_disease']],
                      node_dict=node_dict,
                      edge_dict=edge_dict)
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
                x_train = torch.tensor([(i.item() - 1) * g_d.shape[1] + j
                                        for (i, j) in x_train])
                model.train()
                pred = model(g_train, feature).flatten()[x_train]
                pred_score = torch.sigmoid(pred)

                optimizer.zero_grad()
                loss = criterion(pred, y_train)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() / len(train_loader)
                trainPred.extend(pred.detach().cpu().numpy())
                trainLabel.extend(y_train.detach().cpu().numpy())
            # AUC_train, AUPR_train = get_metrics_auc(label_train.cpu().detach().numpy(),
            #                                         pred_train.cpu().detach().numpy())
            AUC, AUPR = get_metrics_auc(trainLabel, trainPred)
            print('Epoch {} Loss: {:.5f}; Train: AUC {:.3f}, AUPR {:.3f}'.format(epoch, total_loss,
                                                                                 AUC, AUPR))
            del pred, pred_score, trainPred, trainLabel, pred_train, label_train
        model.eval()
        val_data = torch.tensor([(i.item() - 1) * g_d.shape[1] + j
                                 for (i, j) in val_data])
        val_pred = torch.sigmoid(model(g_train, feature).flatten()[val_data]).detach().cpu().numpy()
        print('-' * 20 + '+' * 20 + '-' * 20)
        AUC, AUPR = get_metrics_auc(val_label, val_pred)
        print('Overall Val: AUC {:.3f}, AUPR {:.3f}'.format(AUC, AUPR))
        print('-' * 20 + '+' * 20 + '-' * 20)
        pd.DataFrame(np.array([val_pred, val_label.detach().cpu().numpy()]).T,
                     columns=['Predict', 'Label']).to_csv('{}_predictions_ukGene_{}.csv'.format(model_name,
                                                                                                SEED),
                                                          index=False, header=False)

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

        # val_g_id = np.load('../data/gd_findNew_mask.npy')
        # data_split = np.array(['train' if i[0] not in val_g_id else 'val'
        #                        for i in g_d_asso])
        # train_pos = g_d_asso[np.where(data_split == 'train')]
        # val_pos = g_d_asso[np.where(data_split == 'val')]
        # train_g_id = [i for i in range(g_d.shape[0]) if i not in val_g_id]
        # g_d_train, g_d_val = g_d[train_g_id], g_d[val_g_id]
        # train_neg = np.random.permutation(np.array(np.where(g_d_train == 0)).T)[: len(train_pos)]
        # val_neg = np.random.permutation(np.array(np.where(g_d_val == 0)).T)[: len(val_pos)]

        val_d_id = np.load('../data/dis_gd_findNew_mask.npy')
        data_split = np.array(['train' if i[-1] not in val_d_id else 'val'
                               for i in g_d_asso])
        train_pos = g_d_asso[np.where(data_split == 'train')]
        val_pos = g_d_asso[np.where(data_split == 'val')]
        train_d_id = [i for i in range(g_d.shape[1]) if i not in val_d_id]
        g_d_train, g_d_val = g_d.T[train_d_id].T, g_d.T[val_d_id].T
        train_neg = np.random.permutation(np.array(np.where(g_d_train == 0)).T)[: len(train_pos)]
        val_neg = np.random.permutation(np.array(np.where(g_d_val == 0)).T)[: len(val_pos)]

        val_dda_drugid = val_pos[:, 0]
        val_dda_disid = val_pos[:, 1]
        g_train = g
        g_train = remove_graph(g_train, val_dda_drugid.tolist(), val_dda_disid.tolist()).to(DEVICE)
        feature = {'gene': torch.from_numpy(gene_feat).to(torch.float32).to(DEVICE),
                   'disease': torch.from_numpy(dis_feat).to(torch.float32).to(DEVICE)}
        train_data, val_data = torch.tensor(np.vstack((train_pos, train_neg))).to(DEVICE), \
                               torch.tensor(np.vstack((val_pos, val_neg))).to(DEVICE)
        train_label = torch.concat([torch.ones(len(train_pos)), torch.zeros(len(train_neg))])
        val_label = torch.concat([torch.ones(len(val_pos)), torch.zeros(len(val_neg))])
        train_loader = DataLoader(TensorDataset(train_data, train_label), BATCH_SIZE,
                                  shuffle=True, drop_last=False)
        node_dict = {}
        edge_dict = {}
        for ntype in g_train.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in g_train.etypes:
            edge_dict[etype] = len(edge_dict)
        model = Model(model_name,
                      in_feats=[feature['gene'].shape[1],
                                feature['disease'].shape[1]],
                      rel_names=g.etypes,
                      meta_paths=[['gene_disease', 'disease_gene'],
                                  ['disease_gene', 'gene_disease']],
                      node_dict=node_dict,
                      edge_dict=edge_dict)
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
                x_train = torch.tensor([(i.item() - 1) * g_d.shape[1] + j
                                        for (i, j) in x_train])
                model.train()
                pred = model(g_train, feature).flatten()[x_train]
                pred_score = torch.sigmoid(pred)

                optimizer.zero_grad()
                loss = criterion(pred, y_train)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() / len(train_loader)
                trainPred.extend(pred.detach().cpu().numpy())
                trainLabel.extend(y_train.detach().cpu().numpy())
            # AUC_train, AUPR_train = get_metrics_auc(label_train.cpu().detach().numpy(),
            #                                         pred_train.cpu().detach().numpy())
            AUC, AUPR = get_metrics_auc(trainLabel, trainPred)
            print('Epoch {} Loss: {:.5f}; Train: AUC {:.3f}, AUPR {:.3f}'.format(epoch, total_loss,
                                                                                 AUC, AUPR))
            del pred, pred_score, trainPred, trainLabel, pred_train, label_train
        model.eval()
        val_data = torch.tensor([(i.item() - 1) * g_d.shape[1] + j
                                 for (i, j) in val_data])
        val_pred = torch.sigmoid(model(g_train, feature).flatten()[val_data]).detach().cpu().numpy()
        print('-' * 20 + '+' * 20 + '-' * 20)
        AUC, AUPR = get_metrics_auc(val_label, val_pred)
        print('Overall Val: AUC {:.3f}, AUPR {:.3f}'.format(AUC, AUPR))
        print('-' * 20 + '+' * 20 + '-' * 20)
        pd.DataFrame(np.array([val_pred, val_label.detach().cpu().numpy()]).T,
                     columns=['Predict', 'Label']).to_csv('{}_predictions_ukDis_{}.csv'.format(model_name,
                                                                                               SEED),
                                                          index=False, header=False)
