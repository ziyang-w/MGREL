from unittest import skip
import h5py
import torch
import numpy as np
import os
import scipy.sparse as sp
import pickle
from utils import network_edge_threshold
import networkx as nx

def load_data_from_mat(skip:bool=False,save:bool=True) -> torch.Tensor:
    '''
    read data from .mat and get:
    * .npz: dis_feat, gene_feat, comb
    * .npz: dd, gd, gg
    * .npy: feature_matrix, edge_index, labels

    REFERENCE:
    https://github.com/juanshu30/Disease-Gene-Prioritization-with-Privileged-Information-and-Heteroscedastic-Dropout/blob/main/LUPI_RGCN/Processing_data.py
    '''
    num_nodes = 15546
    num_features = 37287
    num_gene = 12331
    num_disease = 3215

    if skip:
        gene_feat = sp.load("data/gene_feat.npz")
        dis_feat = sp.load("data/dis_feat.npz")
        comb  = sp.load("data/comb.npz")
    
    else:
        # gene interaction network
        gene_phenes_path = 'data/genes_phenes.mat'
        f = h5py.File(gene_phenes_path, 'r')

        print('loading data from {}'.format(gene_phenes_path))

        gene_network_adj = sp.csc_matrix((np.array(f['GeneGene_Hs']['data']),
                                        np.array(f['GeneGene_Hs']['ir']), np.array(f['GeneGene_Hs']['jc'])),
                                        shape=(12331, 12331))
        gene_network_adj = gene_network_adj.tocsr()
        gene_network_adj_row = (gene_network_adj.tocoo()).row
        gene_network_adj_col = (gene_network_adj.tocoo()).col
        gene_network_adj_value = np.array(f['GeneGene_Hs']['data'])

        disease_network_adj = sp.csc_matrix((np.array(f['PhenotypeSimilarities']['data']),
                                            np.array(f['PhenotypeSimilarities']['ir']),
                                            np.array(f['PhenotypeSimilarities']['jc'])),
                                            shape=(3215, 3215))
        disease_network_adj = disease_network_adj.tocsr()

        disease_network_adj_row = (disease_network_adj.tocoo()).row
        disease_network_adj_col = (disease_network_adj.tocoo()).col
        disease_network_adj_value = np.array(f['PhenotypeSimilarities']['data'])

        # gene disease network
        # dg_ref = f['GenePhene'][0][0]
        # gene_disease_adj = sp.csc_matrix((np.array(f[dg_ref]['data']),
        #                                 np.array(f[dg_ref]['ir']), np.array(f[dg_ref]['jc'])),
        #                                 shape=(12331, 3215))
        # gene_disease_adj = gene_disease_adj.tocsr()

        # using our data to replace the gene_disease_adj
        gene_disease_adj = pickle.load(open('./data/geneDisAdj_sum.pkl','rb'))
        gene_disease_adj = gene_disease_adj.T.tocsr() # 重复的边会被去相加
        print(gene_disease_adj.sum())
        print(gene_disease_adj.shape)

        # disease features
        disease_tfidf_path = 'data/clinicalfeatures_tfidf.mat'
        f_disease_tfidf = h5py.File(disease_tfidf_path, "r")
        disease_tfidf = np.array(f_disease_tfidf['F'])
        disease_tfidf = np.transpose(disease_tfidf)
        disease_tfidf = sp.csc_matrix(disease_tfidf)

        # Gene feature1:microarray features
        gene_feature_path = 'data/GeneFeatures.mat'
        f_gene_feature = h5py.File(gene_feature_path, 'r')
        gene_feature_exp = np.array(f_gene_feature['GeneFeatures'])
        gene_feature_exp = np.transpose(gene_feature_exp)
        gene_network_exp = sp.csc_matrix(gene_feature_exp)

        # Gene feature2:other species features
        row_list = [3215, 1137, 744, 2503, 1143, 324, 1188, 4662, 1243]
        gene_feature_list_other_spe = list()
        for i in range(1, 9):
            dg_ref = f['GenePhene'][i][0]
            disease_gene_adj_tmp = sp.csc_matrix((np.array(f[dg_ref]['data']),
                                                np.array(f[dg_ref]['ir']), np.array(f[dg_ref]['jc'])),
                                                shape=(12331, row_list[i]))
            gene_feature_list_other_spe.append(disease_gene_adj_tmp)
            
        # Combine the gene features: 4536 micro-array features and features from other species
        gene_feat = sp.hstack(gene_feature_list_other_spe + [gene_feature_exp])
        dis_feat = disease_tfidf

        
        ### Create the gene disease network from gene network and disease network
        # gene Network
        gene_network_adj_row = (gene_network_adj.tocoo()).row
        gene_network_adj_col = (gene_network_adj.tocoo()).col
        gene_network_adj_value = np.array(f['GeneGene_Hs']['data'])
        gene_network_matrix = np.zeros((np.shape(gene_network_adj)))
        for i, j in zip(gene_network_adj_row, gene_network_adj_col):
            gene_network_matrix[i, j] = 1

        # disease Network
        disease_network_adj = network_edge_threshold(disease_network_adj, 0.2)
        disease_network_adj_row = (disease_network_adj.tocoo()).row
        disease_network_adj_col = (disease_network_adj.tocoo()).col
        disease_network_adj_value = np.array(f['PhenotypeSimilarities']['data'])
        disease_network_matrix = np.zeros((np.shape(disease_network_adj)))
        for i, j in zip(disease_network_adj_row, disease_network_adj_col):
            disease_network_matrix[i, j] = 1
        disease_network_matrix = disease_network_matrix - np.diag(np.ones(len(disease_network_matrix)))

        print(gene_disease_adj.sum())
        #disease:
        gene_disease_adj_row = (gene_disease_adj.tocoo()).row
        gene_disease_adj_col = (gene_disease_adj.tocoo()).col
        disease_gene_matrix = np.zeros((np.shape(gene_disease_adj)))
        for i, j in zip(gene_disease_adj_row, gene_disease_adj_col):
            disease_gene_matrix[i, j] = 1

        print(disease_gene_matrix.sum())
        
        # parprep data for make graph
        # comb is adj matrix, need this to prepare data for OpenNE
        comb1 = np.hstack((gene_network_matrix, disease_gene_matrix))
        comb2 = np.hstack((disease_gene_matrix.T, disease_network_matrix))
        comb = np.vstack((comb1, comb2))

        ### Combine the fetaure matrix
        disease_network_adj_row = (disease_network_adj.tocoo()).row
        disease_network_adj_col = (disease_network_adj.tocoo()).col
        disease_network_adj_value = np.array(f['PhenotypeSimilarities']['data'])
        disease_network_matrix = np.zeros((np.shape(disease_network_adj)))
        for i, j in zip(disease_network_adj_row, disease_network_adj_col):
            disease_network_matrix[i, j] = 1
        disease_network_matrix = disease_network_matrix - np.diag(np.ones(len(disease_network_matrix)))

        # edge_index, label and the feature matrix that will be used
        edge_index = torch.tensor(comb.nonzero()) # 已经返回的是连接的位置了, shape=(2,1376654)
        label = torch.zeros(num_disease + num_gene)
        label[0:num_gene] = 1
        disease_feat_matrix = dis_feat.tocsc().toarray()
        disease_feat_none = np.zeros((num_gene, np.shape(disease_feat_matrix)[1]))
        gene_feat_matrix = gene_feat.tocsc().toarray()
        gene_feat_none = np.zeros((num_disease,np.shape(gene_feat_matrix)[1]))

        # feature matrix,shape=(15546, 34072)
        feature_matrix = np.vstack((np.hstack((gene_feat_matrix, disease_feat_none)), 
                                    np.hstack((gene_feat_none, disease_feat_matrix))))

        feature_matrix = torch.from_numpy(feature_matrix).double()  

        if save:
            # save ndarray to .npy
            sp.save_npz("data/gg",sp.csr_matrix(gene_network_matrix))
            sp.save_npz("data/dd",sp.csr_matrix(disease_network_matrix))
            sp.save_npz("data/gd",sp.csr_matrix(disease_gene_matrix))
            
            # this data are prepaerd for LUPI_RGCN
            np.save("data/feature_matrix",feature_matrix)
            np.save("data/label",label) # noteType: 1 for gene, 0 for disease
            np.save("data/edge_index",edge_index)
            # this data are prepared for our model
            sp.save_npz("data/gene_feat",gene_feat)
            sp.save_npz("data/dis_feat",dis_feat)
            sp.save_npz("data/comb",sp.csr_matrix(comb))
            print('feature_matrix.npy, label.npy, edge_index.npy')
            print('gg.npz, gd.npz, dd.npz')
            print('gene_feat.npz, dis_feat.npz, comb.npz')
            print('has been saved in {}/data'.format(os.getcwd()))

    return gene_feat, dis_feat, comb, label

def make_data_for_openne(comb:np.array,label:np.array)->None:
    '''
    comb  <- load_data_from_mat()
    label <- load_data_from_mat()
    '''
    # using networkx to make adjlist files -> OpenNE
    G = nx.from_numpy_matrix(comb)
    nx.write_adjlist(G, 'OpenNE/GeneDis/adj.adjlist')
    print('adjlist file has been made into: OpenNE/GeneDis/adj.adjlist')
            
    # network -> labels.txt -> OpenNE
    with open('OpenNE/GeneDis/labels.txt','w') as f:
        for i,r in enumerate(label):
            f.write('{} {}\n'.format(i,r.item()))
    print('labels.txt file has been made into: OpenNE/GeneDis/labels.txt')
