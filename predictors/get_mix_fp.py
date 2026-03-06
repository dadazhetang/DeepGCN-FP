import argparse
import copy
import time
import joblib
import dgl
import numpy as np
import random
import os
import pandas as pd
from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from joblib import dump
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dgl.dataloading import GraphDataLoader
from pathlib import Path
import gc
from dataset import  load_smrt_data_one_hot_ecfpmix, get_node_dim, get_edge_dim
from dataset import get_node_dim, get_edge_dim
from models import  GCNModelWithEdgeAFPreadout
from utils import count_parameters, count_no_trainable_parameters, count_trainable_parameters
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
current_dir = Path(os.getcwd())
PROJECT_ROOT = current_dir.parent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    dgl.seed(seed)
    dgl.random.seed(seed)

def getdata(fp,batch_size):
        all_fps = []
        ecfps = list(fp)
        ecfps = [list(map(int, s)) for s in ecfps]
        num_dataloader = int(np.ceil(len(ecfps) / batch_size))
        for i in range(num_dataloader):
            ecfp = ecfps[i * batch_size:(i + 1) * batch_size]
            all_fps.append(ecfp)
        return all_fps
def train_train(model,train_dataset,all_fps,save_path):
    mix = []
    temp_all_lable = []
    all_readout = []
    t = 0
    for graph, ecfp_temp in zip(train_dataset, all_fps):
        t += 1
        ecfp_temp = torch.tensor(ecfp_temp).to(device)
        g = graph[0]
        labels = graph[1]
        g = g.to(device)
        labels = labels.reshape(-1, 1)
        labels = labels.to(device)
        _, readout, ecfps = model(g.to('cpu'), ecfp_temp.to('cpu'))
        mix.append(torch.cat([ecfps, readout], axis=1))
        temp_all_lable.append(labels)
        all_readout.append(readout)

        print(t)
        if (len(train_dataset)-t)>100 and  t % 100 == 0:
            mix_one_val = torch.cat(mix, dim=0)
            mix_one_val = mix_one_val.detach().numpy()
            readout_one_val = torch.cat(all_readout, dim=0)
            readout_one_val = readout_one_val.detach().numpy()
            temp_all_lable_val = torch.cat(temp_all_lable, dim=0)
            temp_all_lable_val =  temp_all_lable_val.detach().cpu().numpy()
            dump(mix_one_val, os.path.join(save_path,"data"+str(int(t/100))+".joblib"))
            dump(temp_all_lable_val, os.path.join(save_path, "label"+str(int(t/100))+".joblib"))
            dump(readout_one_val, os.path.join(save_path, "readout"+str(int(t/100))+".joblib"))
            del mix
            del temp_all_lable
            del mix_one_val
            del temp_all_lable_val
            del all_readout
            mix = []
            temp_all_lable = []
            all_readout = []
        if (len(train_dataset)-t) == 0:
            mix_one_val = torch.cat(mix, dim=0)
            mix_one_val = mix_one_val.detach().numpy()
            readout_one_val = torch.cat(all_readout, dim=0)
            readout_one_val = readout_one_val.detach().numpy()
            temp_all_lable_val = torch.cat(temp_all_lable, dim=0)
            temp_all_lable_val =  temp_all_lable_val.detach().cpu().numpy()
            dump(mix_one_val, os.path.join(save_path,"data"+"0"+".joblib"))
            dump(temp_all_lable_val, os.path.join(save_path, "label" + "0" + ".joblib"))
            dump(readout_one_val, os.path.join(save_path, "readout" + "0" + ".joblib"))

            del mix
            del temp_all_lable
            del all_fps
            del mix_one_val
            del temp_all_lable_val
            del all_readout
def main():
    seed_torch(seed=args.seed)
    args.name = f"{args.model_name}_norm_{args.norm}_layer_{args.num_layers}_k_{args.gru_out_layer}_lr_{args.lr}_dropout_{args.dropout}_seed_{args.seed}_exclude_node_{str(args.exclude_node)}_exclude_{str(args.exclude_edge)}"
    batch_size = args.batch_size
    model = GCNModelWithEdgeAFPreadout(node_in_dim=get_node_dim(), edge_in_dim=get_edge_dim(), hidden_feats=[200] * 16,
                                       dropout=0.1)

    model.load_state_dict(torch.load(args.best_model_file))
    model.eval()
    all_dataset = load_smrt_data_one_hot_ecfpmix(random_state=args.seed,dataset_name=args.dataset,raw_dir=PROJECT_ROOT/"transfer_learning/dataset",exclude_node=args.exclude_node,exclude_edge=args.exclude_edge)
    #val_dataset

    all_dataloader = GraphDataLoader(all_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    temp_fps_train = all_dataloader.dataset.all_fp
    ecfps_1024_train = temp_fps_train[0]

    # 1 ecfps
    fp_train = ecfps_1024_train
    train_data =getdata(fp_train, batch_size)
    result_path = args.all_fp_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    train_train(model, all_dataloader, train_data,result_path)


if __name__ == '__main__':
    """
    Model and training parameters
    """
    parser = argparse.ArgumentParser(description='')
    #wandb name, dataset name, model name
    parser.add_argument('--name', type=str, default="test", help='wandb_running_name')
    parser.add_argument('--dataset', type=str, default='SMRT', help='Name of dataset.')
    parser.add_argument('--model_name', type=str, default='GCN_edge_attention_GRU', help='Name of model, choose from: GAT, GCN, GIN, AFP, DEEPGNN')
    parser.add_argument('--best_model_file', type=str, default='none', help='path of your best model.')
    parser.add_argument('--all_fp_path', type=str, default='none', help='path of fp result.')
    # GNN model args
    parser.add_argument('--num_layers', type=int, default=16, help='Number of GNN layers.')
    parser.add_argument('--hid_dim', type=int, default=200, help='hidden dim.')
    parser.add_argument('--gru_out_layer', type=int, default=2, help='readout layer')
    parser.add_argument('--norm', type=str, default='none', help='choose from: batch_norm, layer_norm, none')
    parser.add_argument('--update_func', type=str, default='no_relu', help='choose from: batch_norm, layer_norm, none')

    # training args
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--early_stop', type=int, default=30, help='Early stop epoch.')
    parser.add_argument('--seed', type=int, default=42, help='set seed')

    parser.add_argument('--exclude_node', default=None, type=str, help='exclude node')
    parser.add_argument('--exclude_edge', default=None, type=str, help='exclude edge')
    args = parser.parse_args()
    print(args)

    main()