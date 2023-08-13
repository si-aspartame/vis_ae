import itertools
import sys
import time

import numpy as np
import pandas as pd
import torch
#import torchsort
from scipy.sparse import csgraph, csr_matrix
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn import neighbors
from torch import distributions as D


def save_scores(crm_scores, elapsed_time, args):
    f = open(f'csv/{args.model}_{args.data}.csv', 'a')
    f.write(f'{crm_scores[0]},{crm_scores[1]},{crm_scores[2]},{crm_scores[3]},{args.z_dim},{elapsed_time}\n')
    f.close()
    return

def pairwise_norm_func(input_tensor, p, batch_size):
    distances = torch.stack([torch.linalg.norm(input_tensor[n]-input_tensor[m], ord=p) for n, m in itertools.combinations(range(batch_size), 2)], dim=0).cuda()
    return distances

def pairwise_norm_indivisual_func(input_tensor, p, batch_size):
    n_dim = input_tensor.shape[1]
    distances = torch.stack([torch.linalg.norm(input_tensor[n]-input_tensor[m], ord=p) for n, m in itertools.combinations(range(batch_size), 2)], dim=0).cuda()
    return distances

def pairwise_wasserstein_func(input_tensor, batch_size):
    distances = torch.stack([torch.Tensor([wasserstein_distance(input_tensor[n], input_tensor[m])]) for n, m in itertools.combinations(range(batch_size), 2)], dim=0).cuda()
    return distances

def get_mnkld(m1, v1, m2, v2):
    output = D.kl.kl_divergence(D.MultivariateNormal(m1, torch.diag_embed(v1)), D.MultivariateNormal(m2, torch.diag_embed(v2)))
    return output

def pairwise_kld_func(mu, var, batch_size):
    kld = lambda m1, v1, m2, v2 : D.kl.kl_divergence(D.MultivariateNormal(m1, torch.diag_embed(v1)), D.MultivariateNormal(m2, torch.diag_embed(v2)))
    #+kld(mu[n], var[n], mu[m], var[m]))*0.5
    pairwise_kld = torch.stack([kld(mu[m], var[m], mu[n], var[n]) for n, m in itertools.combinations(range(batch_size), 2)], dim=0).cuda()
    return pairwise_kld

def rank_func(distance):
    return torch.stack([torch.sum(torch.relu((torch.relu(distance - d)*10000)+1)) for d in distance]).reshape(1,-1)

def get_upper_triangular(distance_matrix):
    d_m_t = distance_matrix.triu().requires_grad_().cuda()#upper triangular of d_m
    mask = torch.ones(d_m_t.shape).bool().triu(diagonal=1)
    unduplicated_distances = d_m_t[mask]#values only in upper triangular
    return unduplicated_distances

def local_distance_matrix(x_data, n_neighbors=5):
    graph = neighbors.kneighbors_graph(x_data, n_neighbors)
    graph = csr_matrix(graph)
    d_m = np.array([csgraph.shortest_path(csgraph=graph, directed=False, indices=n, return_predecessors=False) for n, _ in enumerate(x_data)])
    if float('inf') in d_m:
        print('[error]n_neighbors is too small')
        sys.exit()
    return d_m

def direct_load_distance_matrix(file_name):
    d_m = pd.read_csv(f'data/csv/{file_name}.csv').to_numpy()
    return d_m

def label_distance_matrix(label, p):
    # d_m = np.zeros((len(label), len(label)))
    # for n, m in itertools.combinations(range(len(label)), 2):
    #     d_m[n, m] = np.abs(label[n]-label[m])
    d_m = distance.squareform(distance.pdist(label.astype(float).reshape(-1 ,1)%3))#+0.001
    return d_m

def bray_curtis_distance(tensor1, tensor2):
    # 同じ列数であることを確認
    if tensor1.size(1) != tensor2.size(1):
        raise ValueError("The number of columns for both tensors should be the same.")
    
    # 短いテンソルの行数を等しくする
    while tensor1.shape[0] > tensor2.shape[0]:
        mean_value = torch.mean(tensor2, dim=0, keepdim=True)
        tensor2 = torch.cat([tensor2, mean_value], dim=0)
    
    while tensor2.shape[0] > tensor1.shape[0]:
        mean_value = torch.mean(tensor1, dim=0, keepdim=True)
        tensor1 = torch.cat([tensor1, mean_value], dim=0)

    # Bray-Curtisの計算
    numerator = torch.sum(torch.abs(tensor1 - tensor2), dim=1)
    denominator = torch.sum(tensor1 + tensor2, dim=1)

    # 0での除算を避けるための小さな値
    epsilon = 1e-10
    distances = numerator / (denominator + epsilon)
    
    return distances