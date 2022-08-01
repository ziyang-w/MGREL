import os
import random
import torch
import dgl
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


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


def get_metrics_auc(real_score, predict_score):
    AUC = roc_auc_score(real_score, predict_score)
    AUPR = average_precision_score(real_score, predict_score)
    return AUC, AUPR


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
