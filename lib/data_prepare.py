import datetime
import pandas as pd
import torch
import numpy as np
import os
from fastdtw import fastdtw
from tqdm import tqdm

import controldiffeq
from .utils import print_log, StandardScaler 

def make_dir(name):
    if (os.path.exists(name)):
        print('has  save path')
    else:
        os.makedirs(name)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_data(data, times):
    augmented_data = []
    times_repeated = times.unsqueeze(0).unsqueeze(0).repeat(data.shape[0], data.shape[2], 1).unsqueeze(-1).transpose(1, 2)
    augmented_data.append(times_repeated)
    augmented_data.append(torch.tensor(data[..., :]))
    return torch.cat(augmented_data, dim=3)

def calculate_adjacency_matrix(adj_mx,weight_adj_epsilon=0.1):
    print("Start Calculate the weight by Gauss kernel!")
    distances = adj_mx[~np.isinf(adj_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(adj_mx / std))
    adj_mx[adj_mx < weight_adj_epsilon] = 0
    return adj_mx

def add_external_information(df, timesolts,add_time_in_day, add_day_in_week,idx_of_ext_timesolts=None,ext_data=None):
    num_samples, num_nodes, feature_dim = df.shape
    is_time_nan = np.isnan(timesolts).any()
    data_list = [df]
    print(df.shape)

    if add_time_in_day and not is_time_nan:
        time_ind = (timesolts - timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week and not is_time_nan:
        dayofweek = []
        for day in timesolts.astype("datetime64[D]"):
            dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, dayofweek] = 1

        day_in_week2 = np.argmax(day_in_week, axis=2).astype(np.float32)[...,np.newaxis]
        data_list.append(day_in_week2)
    if ext_data is not None:
        if not is_time_nan:
            indexs = []
            for ts in timesolts:
                ts_index = idx_of_ext_timesolts[ts]
                indexs.append(ts_index)
            select_data = ext_data[indexs]
            for i in range(select_data.shape[1]):
                data_ind = select_data[:, i]
                data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                data_list.append(data_ind)
        else:
            if ext_data.shape[0] == df.shape[0]:
                select_data = ext_data
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                    data_list.append(data_ind)
    data = np.concatenate(data_list, axis=-1)
    return data

def load_rel(data_path , filename,set_weight_link_or_dist='dist',init_weight_inf_or_zero='zero',bidir=True,calculate_weight_adj=False):
 
    relfile = pd.read_csv(data_path+filename + '.rel')
    geofile = pd.read_csv(data_path + filename + '.geo')
    geo_ids = list(geofile['geo_id'])
    num_nodes = len(geo_ids)
    geo_to_ind = {}
    for index, idx in enumerate(geo_ids):
        geo_to_ind[idx] = index

    print('set_weight_link_or_dist: {}'.format(set_weight_link_or_dist))
    print('init_weight_inf_or_zero: {}'.format(init_weight_inf_or_zero))
    weight_col = ''
    if weight_col != '':
        if isinstance(weight_col, list):
            if len(weight_col) != 1:
                raise ValueError('`weight_col` parameter must be only one column!')
            weight_col = weight_col[0]
        distance_df = relfile[~relfile[weight_col].isna()][[
            'origin_id', 'destination_id', weight_col]]
    else:
        if len(relfile.columns) != 5:
            raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
        else:
            weight_col = relfile.columns[-1]
            distance_df = relfile[~relfile[weight_col].isna()][[
                'origin_id', 'destination_id', weight_col]]
    adj_mx = np.zeros((len(geo_ids), len(geo_ids)), dtype=np.float32)
    if init_weight_inf_or_zero.lower() == 'inf' and set_weight_link_or_dist.lower() != 'link':
        adj_mx[:] = np.inf
    for row in distance_df.values:
        if row[0] not in geo_to_ind or row[1] not in geo_to_ind:
            continue
        if set_weight_link_or_dist.lower() == 'dist':
            adj_mx[geo_to_ind[row[0]], geo_to_ind[row[1]]] = row[2]
            if bidir:
                adj_mx[geo_to_ind[row[1]], geo_to_ind[row[0]]] = row[2]
        else:
            adj_mx[geo_to_ind[row[0]], geo_to_ind[row[1]]] = 1
            if bidir:
                adj_mx[geo_to_ind[row[1]], geo_to_ind[row[0]]] = 1
    print("Loaded file " + filename + '.rel &.geo,adj_mx shape=' + str(adj_mx.shape))
    if calculate_weight_adj:
        adj_mx = calculate_adjacency_matrix(adj_mx)
    return adj_mx, num_nodes

def load_dyna(data_path, filename, geo_ids, data_col,load_external=True):
    print_log("Loading file " + filename + '.dyna')
    dynafile = pd.read_csv(data_path + filename + '.dyna')
    if data_col != '':
        if isinstance(data_col, list):
            data_col = data_col.copy()
        else:
            data_col = [data_col].copy()
        data_col.insert(0, 'time')
        data_col.insert(1, 'entity_id')
        dynafile = dynafile[data_col]
    else:
        dynafile = dynafile[dynafile.columns[2:]]
    timesolts = list(dynafile['time'][:int(dynafile.shape[0] / len(geo_ids))])
    idx_of_timesolts = dict()
    if not dynafile['time'].isna().any():
        timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), timesolts))
        timesolts = np.array(timesolts, dtype='datetime64[ns]')
        for idx, _ts in enumerate(timesolts):
            idx_of_timesolts[_ts] = idx
    feature_dim = len(dynafile.columns) - 2
    df = dynafile[dynafile.columns[-feature_dim:]]
    len_time = len(timesolts)
    data = []
    for i in range(0, df.shape[0], len_time):
        data.append(df[i:i+len_time].values)
    data = np.array(data, dtype=np.float)
    data = data.swapaxes(0, 1)
    print_log("Loaded file " + filename + '.dyna' + ', shape=' + str(data.shape))
    if load_external:
        data = add_external_information(data,timesolts = timesolts, idx_of_ext_timesolts=None, add_time_in_day=True, add_day_in_week=True, ext_data = None)
    return data.astype(np.float32)


def add_window_horizon(df, input_window = 12, output_window = 12):
    num_samples = df.shape[0]
    x_offsets = np.sort(np.concatenate((np.arange(-input_window + 1, 1, 1),)))
    y_offsets = np.sort(np.arange(1, output_window + 1, 1))

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = df[t + x_offsets, ...]
        y_t = df[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y
 
def split_train_val_test( x, y, train_rate = 0.6, eval_rate = 0.2, test_rate = 0.2):
    test_rate = 1 - train_rate - eval_rate
    num_samples = x.shape[0]
    num_test = round(num_samples * test_rate)
    num_train = round(num_samples * train_rate)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]
    print_log("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
    print_log("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
    print_log("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

 
 


def shift_traffic_data(data, num, fill_value=0):
    """
    Shifts array along the first axis by the number of steps given by num.
    Positive num shifts down, negative num shifts up.
    fill_value is used to replace the missing values after the shift.
    """
    arr = data[:,:,:,:1]
    result = np.full_like(arr, fill_value=fill_value, dtype=np.float64)
    if num > 0:  # Shift down
        result[:num] = arr[:num]
        result[num:] = arr[:-num]
    elif num < 0:  # Shift up
        result[num:] = arr[num:]
        result[:num] = arr[-num:]
    else:  # No shift
        result[:] = arr

    return result

def shift_day_week_to_x(x, num,  weight_cache='', diff_rate = 0.0):
    day_shift = shift_traffic_data(x[...,:1], num =288+num)
    two_shift = shift_traffic_data(x[...,:1], num =576+num)
    threeday_shift = shift_traffic_data(x[...,:1], num =864+num)
    four_shift = shift_traffic_data(x[...,:1], num =1152+num)
    five_shift = shift_traffic_data(x[...,:1], num =1440+num)
    sixday_shift = shift_traffic_data(x[...,:1], num =1728+num)
    week_shift = shift_traffic_data(x[...,:1], num =2016+num)
    avg_shift = np.zeros_like(day_shift)
    
    shifts = np.concatenate([day_shift, two_shift, threeday_shift, four_shift, five_shift, sixday_shift, week_shift], axis= -1)
    shift_weight = np.load(weight_cache).astype(np.float32)
    avg_shift[:2016] = 0.8*day_shift[:2016] + 0.2*week_shift[:2016]
    avg_shift[2016:] = (shifts[2016:] * shift_weight).sum(axis=-1)[...,np.newaxis]
    if diff_rate > 0.0:
        node_x_diff = np.diff(x[...,:1], axis=0)  # 默认计算沿着第一个轴（时间轴）的差分
        node_x_diff_padded = np.zeros_like(x[...,:1])
        node_x_diff_padded[1:] = node_x_diff
        avg_shift = avg_shift + diff_rate * node_x_diff_padded
    shift_x =  np.concatenate([x, avg_shift], axis= -1)
    return shift_x
  

 
def generate_data_point(
    data_dir, dataset, data_col, output_dim , tod=False, dow=False, dom=False, batch_size=64, log=None, load_dtw = False, load_external=True, train_rate=0.6, eval_rate=0.2, test_rate=0.2, full_rate=1.0, input_dim  = 3, log_dir = None
):  
    x_list, y_list = [], []
    geofile = pd.read_csv(data_dir + str(dataset) + '.geo')
    adj_semx = np.load(data_dir +'dtw_'+ str(dataset) + '.npy')

    geo_ids = list(geofile['geo_id'])
    df = load_dyna(data_path=data_dir, filename=str(dataset), geo_ids=geo_ids,  load_external=load_external, data_col=data_col)

    df = df[: round(df.shape[0] * full_rate)].astype(np.float32)
    adj_mx, num_nodes = load_rel(data_path=data_dir, filename=str(dataset))
    x, y = add_window_horizon(df,input_window= 12, output_window= 12)
    weight_cache =  '../finetune_his/'+dataset +'_final_best_hisweight.npy'
    
    if dataset == "PEMS04" or dataset == "PEMS03":
        diff_rate = 0.5
    else :
        diff_rate = 0.0
        
    print(diff_rate)
    x = shift_day_week_to_x(x, -12,  weight_cache=weight_cache, diff_rate= diff_rate).astype(np.float32)
    x_list.append(x.astype(np.float32))
    y_list.append(y.astype(np.float32))
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    y = y[..., :output_dim]
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y,train_rate=train_rate, eval_rate= eval_rate, test_rate= test_rate) #划分数据集
    
 
    scaler = StandardScaler(mean=x_train[..., :output_dim].mean(), std=x_train[..., :output_dim].std())
    print_log(f"StandardScaler mean: {x_train[..., :output_dim].mean()}, std: {x_train[..., :output_dim].std()}", log=log) #BTND
    x_train[..., :output_dim] = scaler.transform(x_train[..., :output_dim])
    x_val[..., :output_dim] = scaler.transform(x_val[..., :output_dim])
    x_test[..., :output_dim] = scaler.transform(x_test[..., :output_dim])
    
    x_train[..., -1:] = scaler.transform(x_train[..., -1:])
    x_val[...,-1:] = scaler.transform(x_val[..., -1:])
    x_test[..., -1:] = scaler.transform(x_test[..., -1:])
    #######################shift#####################
    # train_shift = scaler.transform(train_shift)
    # val_shift = scaler.transform(val_shift)
    # test_shift = scaler.transform(test_shift)


    print_log(f"Mean:\tx-{scaler.mean}\tStd-{scaler.std}", log=log)
    #---------------------------------cde---------------------------------
    times = torch.linspace(0, 11, 12)
    coeff_tra = process_data(x_train[...,:-1], times)
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, coeff_tra.transpose(1,2))
    coeff_val = process_data(x_val[...,:-1], times)
    valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, coeff_val.transpose(1,2))
    coeff_test = process_data(x_test[...,:-1], times)
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, coeff_test.transpose(1,2))
    print_log(f"Traincoeffs:\tx-{train_coeffs[0].shape}\tlen-{len(train_coeffs)}", log=log)
    print_log(f"Valsetcoeffs:  \tx-{valid_coeffs[0].shape}\tlen-{len(valid_coeffs)}", log=log)
    print_log(f"Testsetcoeffs:\tx-{test_coeffs[0].shape}\tlen-{len(test_coeffs)}", log=log)
    #-------------------------loader---------------------------------------------
    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train), *train_coeffs
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val),*valid_coeffs
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test), *test_coeffs
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,pin_memory=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False,pin_memory=True
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,pin_memory=True
    )
    
    return trainset_loader, valset_loader, testset_loader, scaler, adj_mx, adj_semx



