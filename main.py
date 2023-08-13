#%%
import argparse
import math
import random
import time

import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.nn as nn
from plotly.offline import init_notebook_mode
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset

from lib import distance_functions as di
from lib import functions as fu
from linear_AE import *

import importlib
importlib.reload(fu)
importlib.reload(di)

parser = argparse.ArgumentParser(description='rank_AE')
parser.add_argument('--LATENT_DIMENSION', type=int, default=2, metavar='N',help='LATENT_DIMENSION')

#%%
__epoch = 1
__es = 10
__batch_size = 8

#--pytorch tensor type
init_notebook_mode(connected = True)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
#--set seed
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#%%
in_data, label = fu.load_data("qm9_spectrum", force_new=True)
in_data = in_data[:5000]
label = label[:5000]
wave = in_data[:,:1000]
in_data = in_data[:, 1000:]

in_data = np.column_stack([in_data, np.array(list(range(len(in_data))))])
in_dim = in_data.shape[1] - 1#without row index

#%%
length = len(in_data)
category = [int(n/(len(in_data)/5)) for n in range(len(in_data))]#疑似カテゴリ
unique_categories = list(set(category))
in_dim = in_data.shape[1] - 1

#%%
mse = nn.MSELoss().cuda()#reconstruction_error
cos = nn.CosineSimilarity(dim=1, eps=1e-6)#regularization_error

def custom_loss(original, latent, output, ws):
    rec_error = mse(original, output)#全体の入力と出力の差
    x_rank, z_rank = di.rank_func(ws), di.rank_func(di.pairwise_norm_func(latent, 2, __batch_size))#可視化用埋め込み領域から作られた距離行列とと外部距離行列の差
    x2z = -1 * (cos(x_rank - x_rank.mean(), z_rank - z_rank.mean()) - 1)
    reg_error = 0.01 * x2z#可視化領域の損失とコードブックの損失で正則化
    return rec_error, reg_error

def get_loss(data, model, batch_idx):
    ws = di.pairwise_wasserstein_func(wave[batch_idx], __batch_size)#external_dm[batch_idx][:, batch_idx]#
    data = data[:, :in_dim].requires_grad_().cuda()#without row index column
    x, z, y, = model(data)#quantized_z, second_z, y_for_umap
    return custom_loss(x, z, y, ws)#comparing dm of latent representation and external_dm

model = AutoEncoders(in_data.shape[1]-1, __batch_size, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

#%%
all_loss = np.array([])
best_loss = 99999
es_count = 0
start_time = time.time()
print(f"final_length:{len(in_data)}")

#%%
print('Train...')
for epoch in range(1, __epoch+1):
    temp_loss = np.array([])#loss, rec_error, reg_error, x2y, x2z
    model.train()
    for n, data in enumerate(DataLoader(torch.from_numpy(in_data).type(torch.cuda.FloatTensor), batch_size = __batch_size, shuffle = True, generator=torch.Generator(device='cuda'))):
        batch_idx = list(map(int, data[:, in_dim].tolist()))
        data = data[:, :in_dim]#without row index column
        rec_error, reg_error = get_loss(data, model, batch_idx)
        optimizer.zero_grad()
        loss = rec_error + reg_error
        loss.backward()
        optimizer.step()
        temp_loss = np.append(temp_loss, np.array(list(map(lambda tensor:tensor.data.sum().item() / (len(in_data) / __batch_size), [loss, rec_error, reg_error])))).reshape(-1, 3)
    temp_loss = np.sum(temp_loss, axis=0)
    loss_dict = {'loss':temp_loss[0], 'rec_error':temp_loss[1], 'reg_error':temp_loss[2]}
    if loss_dict['loss'] < best_loss:
        print(f'[BEST] ', end='')
        torch.save(model.state_dict(), f'best.pth')
        best_loss = loss_dict['loss']
        es_count = 0
    es_count += 1
    print(f"epoch [{epoch}/{__epoch}], loss:{loss_dict['loss']}, {int(time.time()-start_time)}s \n rec_error = {loss_dict['rec_error']}, reg_error:{loss_dict['reg_error']}")
    all_loss = np.append(all_loss, np.array([epoch, loss_dict['loss'], loss_dict['rec_error'], loss_dict['reg_error']])).reshape(-1, 4)
    if es_count == __es:
        print('early stopping!')
        break#early_stopping

#%%
print('CreatePlots...')
model.load_state_dict(torch.load(f'best.pth'))
model.eval()
print('Eval')

lat_result = np.empty((0, 2))#潜在空間
for n, data in enumerate(DataLoader(torch.from_numpy(in_data).type(torch.cuda.FloatTensor), batch_size =__batch_size, shuffle = False)):#シャッフルしない
    batch = data[:, :in_dim].view(__batch_size, 1, in_dim)#.cuda()
    temp = model(batch)
    lat_result = np.vstack((lat_result, temp[1].view(__batch_size, 2).data.cpu().numpy()))#numpy().reshape(args.batch_size, args.z_dim)

#%%
elapsed_time = time.time() - start_time
print(elapsed_time)

#%%
print('SaveLatent...')
pd.DataFrame(lat_result).to_csv('out.csv', header=False, index=False)

#%%
print(len(lat_result), len(label))
fig = fu.plot_latent(lat_result, label[:len(lat_result)], mode='scatter', title='scatter')
fig.write_image(f"/home/si/Develop/notebook_files/Github/ref_ae/comparison/scatter.png")

