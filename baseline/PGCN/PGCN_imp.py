# This is a sample Python script.
import os
import random
import dgl
import h5py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import dgl.nn as dglnn
import scipy.sparse as sp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from tqdm import tqdm

DEVICE = 'cpu'
SEED = 0
NUM_FOLD = 5
BATCH_SIZE = 512
LR = 0.001
EPOCH = 300

if DEVICE != 'cpu':
    print('Training on GPU')
    device = torch.device('cuda:{}'.format(DEVICE))
else:
    print('Training on CPU')
    device = torch.device('cpu')

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
g.nodes['gene'].data['h'] = torch.from_numpy(gene_feat).to(torch.float32)
g.nodes['disease'].data['h'] = torch.from_numpy(dis_feat).to(torch.float32)


# g = dgl.to_homogeneous(g)


class PGCN(nn.Module):

    def __init__(self, in_feats, rel_names):
        super().__init__()
        self.gene_emb = nn.Linear(in_feats[0], 64, bias=False)
        self.dis_emb = nn.Linear(in_feats[1], 64, bias=False)
        HeteroGraphdict_1 = {}
        for rel in rel_names:
            graphconv = dglnn.GraphConv(64, 64)
            nn.init.xavier_normal_(graphconv.weight)
            HeteroGraphdict_1[rel] = graphconv
        self.embedding_1 = dglnn.HeteroGraphConv(HeteroGraphdict_1, aggregate='sum')
        HeteroGraphdict_2 = {}
        for rel in rel_names:
            graphconv = dglnn.GraphConv(64, 32)
            nn.init.xavier_normal_(graphconv.weight)
            HeteroGraphdict_2[rel] = graphconv
        self.embedding_2 = dglnn.HeteroGraphConv(HeteroGraphdict_2, aggregate='sum')
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.weights = nn.Linear(32, 32, bias=False)
        nn.init.xavier_uniform_(self.weights.weight)

    def forward(self, g, feature):
        h = {'gene': self.gene_emb(feature['gene']),
             'disease': self.dis_emb(feature['disease'])}

        h = self.embedding_1(g, h)
        h = {k: self.relu(self.dropout(v)) for k, v in h.items()}

        h = self.embedding_2(g, h)
        h = {k: self.relu(self.dropout(v)) for k, v in h.items()}
        outputs = torch.matmul(self.weights(h['gene']), h['disease'].T)
        return outputs


def remove_graph(g, test_gene_id, test_disease_id):
    etype = ('gene', 'gene_disease', 'disease')
    edges_id = g.edge_ids(torch.tensor(test_gene_id),
                          torch.tensor(test_disease_id),
                          etype=etype)
    g = dgl.remove_edges(g, edges_id, etype=etype)
    etype = ('disease', 'disease_gene', 'gene')
    edges_id = g.edge_ids(torch.tensor(test_disease_id),
                          torch.tensor(test_gene_id),
                          etype=etype)
    g = dgl.remove_edges(g, edges_id, etype=etype)
    return g


def set_seed(seed=0):
    print('seed = {}'.format(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    # dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_metrics_auc(real_score, predict_score):
    AUC = roc_auc_score(real_score, predict_score)
    AUPR = average_precision_score(real_score, predict_score)
    return AUC, AUPR


# def construct_negative_graph(train_edge, graph, k):
#     new_g = graph
#     new_g = remove_graph(new_g, train_edge[:, 0], train_edge[:, 1])
#     neg_gene = torch.randint(0, graph.num_nodes('gene'), len(train_edge) * k)
#     neg_dis = torch.randint(0, graph.num_nodes('disease'), len(train_edge) * k)
#     new_g = dgl.add_edges(new_g, neg_gene, neg_dis, ('gene', 'gene_disease', 'disease'))
#     new_g = dgl.add_edges(new_g, neg_dis, neg_gene, ('disease', 'disease_gene', 'gene'))
#     return new_g


# data = np.vstack((np.hstack((gene_feat, np.zeros(dis_feat.shape[1]))),
#                   np.hstack((np.zeros(gene_feat.shape[1]), dis_feat))))
# data = np.array([[i, j] for i in range(g_d.shape[0])
#                  for j in range(g_d.shape[1])])
set_seed(SEED)
data = torch.tensor(g_d_asso).to(DEVICE)
label = torch.tensor(np.ones(data.shape[0])).to(DEVICE)

kf = StratifiedKFold(NUM_FOLD, shuffle=True, random_state=SEED)
fold = 1
pred_result, pred_label = [], []
for (train_idx, val_idx) in kf.split(data, label):
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
    feature = {'gene': g_train.nodes['gene'].data['h'],
               'disease': g_train.nodes['disease'].data['h']}
    train_loader = DataLoader(TensorDataset(train_data, train_label), BATCH_SIZE,
                              shuffle=True, drop_last=False)
    model = PGCN(in_feats=[feature['gene'].shape[1],
                           feature['disease'].shape[1]],
                 rel_names=g.etypes)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(1, EPOCH + 1):
        total_loss = 0
        pred_train, label_train = torch.zeros(train_label.shape).to(device), \
                                  torch.zeros(train_label.shape).to(device)
        trainPred, trainLabel = [], []
        for i, data_ in enumerate(train_loader):
            x_train, y_train = data_[0].to(device), data_[1].to(device)
            neg_gene = torch.randint(0, g_train.num_nodes('gene'), (len(x_train),))
            # neg_dis = torch.randint(0, g_train.num_nodes('disease'), (len(x_train),)
            x_train_neg = torch.stack((neg_gene, x_train[:, 1])).T
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
    model.eval()
    g_d_neg_asso = np.array(np.where(g_d == 0)).T
    g_d_neg_asso = np.random.permutation(g_d_neg_asso)
    x_val_neg = g_d_neg_asso[len(val_label) * (fold - 1): len(val_label) * fold]
    neg_label = torch.zeros(val_label.shape).to(DEVICE)
    val_data = torch.concat([val_data, x_val_neg])
    val_data = torch.tensor([(i.item() - 1) * g_d.shape[1] + j
                             for (i, j) in val_data])
    val_label = torch.concat([val_label, neg_label]).detach().cpu().numpy()
    val_pred = torch.sigmoid(model(g_train, feature)[val_data]).detach().cpu().numpy()
    print('-' * 20 + '+' * 20 + '-' * 20)
    AUC, AUPR = get_metrics_auc(val_label, val_pred)
    print('Overall Val: AUC {:.3f}, AUPR {:.3f}'.format(AUC, AUPR))
    print('-' * 20 + '+' * 20 + '-' * 20)
    pred_result.extend(val_pred)
    pred_label.extend(val_label)
    fold += 1
pd.DataFrame(np.array([pred_result, pred_label]),
             columns=['Predict', 'Label']).to_csv('PGCN_predictions_{}.csv'.format(SEED),
                                                  index=False, header=False)
