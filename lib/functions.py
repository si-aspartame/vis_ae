import inspect as ins
import os
import pickle
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from natsort import natsorted
from sklearn import datasets, preprocessing
from sklearn.datasets import make_s_curve, make_swiss_roll
from torch import Size


def func_source(functions):
    return [[f.__name__, ins.getsource(f)] for f in functions]

import numpy as np


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def make_directories():
    today = time.strftime('%Y_%m_%d')
    now = time.strftime('%H_%M_%S')
    if not os.path.isdir(os.getcwd()+'/results'):
        print('results is not found, make')
        os.makedirs('results')
    os.chdir('results')
    if not os.path.isdir(os.getcwd()+'/'+today):
        print(f"{today} is not found, make")
        os.makedirs(today)
    os.chdir(today)
    os.makedirs(now)
    os.chdir(now)
    maked_dir = os.getcwd()
    os.makedirs('saved_model')
    os.makedirs('rec')
    os.makedirs('lat')
    os.makedirs('comparison')
    os.makedirs('csv')
    return maked_dir

def save_dict(dictionary, filename, rename=False):
    pf = pd.DataFrame.from_dict(dictionary, orient='index')
    pf.to_csv(f"{filename}.txt", sep=",", encoding="utf-8", header=False)
    if rename == True:
        cwd = str(os.getcwd())
        os.chdir('../')
        os.rename(cwd, cwd+'_scored')
        os.chdir(cwd+'_scored')
    return
    
def save_globals_and_functions(p, functions):
    save_dict(p, 'parameters')
    fs = func_source(functions)
    for name, source in fs:
        f = open(f'{name}.txt', 'w')
        f.write(source)
        f.close()
    print('completed')
    os.chdir('../../../')
    return

def split_points(in_data, label):
    in_data = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(in_data)
    bool_a = np.array([(0.5 < in_data[n, 0]) for n, _ in enumerate(in_data)]) * np.array([(in_data[n, 0] < 0.55) for n, _ in enumerate(in_data)])
    bool_b = np.array([(0.5 < in_data[n, 1]) for n, _ in enumerate(in_data)]) * np.array([(in_data[n, 1] < 0.55) for n, _ in enumerate(in_data)])
    bool_c = np.array([(0.5 < in_data[n, 2]) for n, _ in enumerate(in_data)]) * np.array([(in_data[n, 2] < 0.55) for n, _ in enumerate(in_data)])
    bool_not_abc = np.logical_not(bool_a+bool_b)
    return in_data[bool_not_abc], label[bool_not_abc]

def load_data(mode, batch_size=1, noise=0, force_new=False, scaling=True):
    path = os.getcwd()
    print(path)
    if os.path.exists(f'data/PICKLE_DATA/{mode}.pickle') and not force_new:
        print('[load]')
        with open(f'data/PICKLE_DATA/{mode}.pickle', 'rb') as directory:
            in_data, label = pickle.load(directory)
    else:
        print('[new]')
        if mode == 'curve':
            in_data, label = make_s_curve(n_samples=10000, noise=noise)
        elif mode == 'roll':
            in_data, label = make_swiss_roll(n_samples=5000, noise=noise)
        elif mode == 'split_roll':
            in_data, label = make_swiss_roll(n_samples=30000, noise=noise)
            in_data, label = split_points(in_data, label)
        elif mode == 'split_curve':
            in_data, label = make_s_curve(n_samples=10000, noise=noise)
            in_data, label = split_points(in_data, label)
        elif mode == 'mnist':
            in_data, label = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, data_home=None)
        elif mode == 'fashion_mnist':
            in_data, label = datasets.fetch_openml('Fashion-MNIST', return_X_y=True, data_home=None)
        elif mode == 'california':
            in_data, label = datasets.fetch_california_housing(return_X_y=True, data_home=None)
        elif mode == 'boston':
            in_data, label = datasets.fetch_california_housing(return_X_y=True, data_home=None)
        elif mode =='cancer':
            in_data, label = datasets.load_breast_cancer(return_X_y=True)
        elif mode == 'MSD':
            in_data, label = datasets.fetch_openml(data_id=31, return_X_y=True)
        elif mode == 'csv':
            df = pd.read_csv('data/material/dl_struct_gap.csv').dropna(how='any', axis=0)
            label = df[df.columns[1]].values
            in_data = df.drop(df.columns[[0,1,2,3,4]], axis=1).values
        elif mode == 'qm7':
            df = pd.read_csv('data/material/qm7_ofm.csv')
            print(df.shape)
            df = df.dropna(how='any', axis=0)
            label = df[df.columns[1]].values
            in_data = df.drop(df.columns[[0,1,2,3,4]], axis=1).values
        elif mode == 'qm9':
            df = pd.read_csv('data/material/qm9_ofm.csv').dropna(how='any', axis=0)
            label = df[df.columns[1]].values
            in_data = df.drop(df.columns[[0,1,2,3,4]], axis=1).values
        elif mode == 'qm9_spectrum':
            print('qm9_spectrum')
            df = pd.read_csv('data/qm9_spectrum_ofm.csv').dropna(how='any', axis=0)
            label = df[df.columns[0]].values
            in_data = df.drop(df.columns[[0,1]], axis=1).values
        elif mode == 'microscope':
            print('microscope')
            in_data = []
            for n in range(1,6):
                file_list = natsorted(os.listdir(f'data/microscope/{n}'))
                for l in file_list:
                    img = cv2.imread(str(f'data/microscope/{n}'+'/'+l), -1)
                    im_gray_calc = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
                    #print(im_gray_calc)
                    #img[index] = 255
                    in_data.append(im_gray_calc.reshape(im_gray_calc.shape[1]**2))
            in_data = np.array(in_data)
            color = np.zeros(2020)
            color[0:901] = 0#901
            color[901:1061] = 1#160*6
            color[1061:1458] = 2#397*
            color[1458:1668] = 3#210
            color[1668:2020] = 4#352
            label = color#np.array(list(range(len(in_data))))
        if scaling == True:
            print("[scaling]")
            print(np.max(in_data), np.min(in_data))
            in_data = min_max(in_data)#preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(in_data)
            print(np.max(in_data), np.min(in_data))
            print(in_data)
        print(f'data_length:{len(in_data)}')
        with open(f'data/PICKLE_DATA/{mode}.pickle', 'wb') as directory:
            pickle.dump([in_data, label], directory)
    fixed_size = len(in_data)-(len(in_data)%batch_size)
    in_data = np.array(in_data)[:fixed_size, :].astype(np.float32)
    label = label[:fixed_size]
    print(f'DATA_LENGTH:{len(in_data)}')
    os.chdir(path)
    return in_data, label

def plot_latent(x_data, label, size=5, mode='scatter', title='no title'):
    print(len(x_data[:, 0]), len(x_data[:, 1]), len(label))
    df = pd.DataFrame({'X':x_data[:, 0], 'Y':x_data[:, 1], 'Labels':label}).sort_values('Labels')
    if mode == 'density':
        fig = px.density_contour(df, x='X', y='Y', title=title, width=1000, height=1000)
        fig.update_traces(contours_coloring="fill", contours_showlabels = True, colorscale=px.colors.sequential.Blues)
    elif mode == 'scatter':
        if x_data.shape[1] == 2:
            print(len(np.repeat(5, len(label))))
            fig = px.scatter(df, x='X', y='Y', color='Labels', size=np.repeat(5, len(label)), size_max=5, opacity=0.5, color_continuous_scale=px.colors.sequential.Agsunset, title=title, width=1000, height=1000)
            fig.update_traces(marker=dict(line=dict(width=0)), selector=dict(mode='markers'))
            fig.update_layout(yaxis=dict(scaleanchor='x'), showlegend=True)#縦横比を1:1に
        elif x_data.shape[1] == 3:
            df = pd.DataFrame({'X':x_data[:, 0], 'Y':x_data[:, 1], 'Z':x_data[:, 2], 'Labels':label}).sort_values('Labels')
            fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Labels', size=size.tolist(), size_max=5, opacity=0.5, color_continuous_scale=px.colors.sequential.Agsunset, title=title, width=1000, height=1000,)
            fig.update_traces(marker=dict(line=dict(width=0)), selector=dict(mode='markers'))
            fig.update_layout(showlegend=True)#縦横比を1:1に
        else:
            fig=[[0,0]]
    elif mode == 'stacking2d':
        fig = ff.create_2d_density(df['X'], df['Y'], colorscale=px.colors.sequential.ice, hist_color='rgb(70, 70, 200)', point_color='rgb(200, 200, 250)', point_size=5, title=title, width=1000, height=1000, ncontours=50)
    return fig




