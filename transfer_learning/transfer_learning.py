
"""
DeepGCN Architecture Implementation

Original Source: https://github.com/kangqiyue/DeepGCN-RT/blob/master/transfer_learning.py
Original Author: Kang et al. (2023)
Original License: Apache License 2.0

Modified by: Zehang Peng (2026)
Modifications:
  - Replaced regression loss with BCEWithLogitsLoss for classification task
  - Adjusted output layer dimension for adduct type prediction
  - Added compatibility with GCN fingerprint extraction workflow

Copyright 2023 Kang et al.

"""
from pathlib import Path
import argparse
import copy
import time
import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, matthews_corrcoef, auc, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import dgl
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score
from dataset import  load_smrt_data_one_hot, get_node_dim, get_edge_dim
from dataset import get_node_dim, get_edge_dim
from models import GCNModelWithEdgeAFPreadout
from utils import count_parameters, count_no_trainable_parameters, count_trainable_parameters
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
# -*- coding: UTF-8 -*-
def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    dgl.seed(seed)
    dgl.random.seed(seed)


# def train function
def train(model, device, dataloader, optimizer, loss_fn, target,type_data):
    num_batches = len(dataloader)
    train_loss = 0
    test = []
    model.train()
    for step, (bg, labels) in enumerate(dataloader):

        bg = bg.to(device)
        labels = labels.reshape(-1, 1)
        labels = labels.to(device)
        # batched_graph = batched_graph.to(device)
        pred,_,_ = model(bg,1)
        loss = loss_fn(pred, labels)
        nan_mask = torch.isnan(pred)
        nan_positions = torch.nonzero(nan_mask, as_tuple=False).flatten()
        nan_labels = labels[nan_positions]
        unique_tensor = torch.unique(nan_labels, dim=0)
        unique_elements = [item[0] for item in unique_tensor.tolist()]
        # runing loss in each batch
        train_loss += loss.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        test.append(unique_elements)
    print()
    return train_loss / num_batches


def test(model,device, dataloader, loss_fn, target,type_data):
    num_batches = len(dataloader)
    test_loss = 0
    test = []
    if type_data == "regression":
        test_loss_MAE = 0
    labels_all = torch.tensor([]).to(device)
    pred_all = torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for step, (bg, labels) in enumerate(dataloader):
            # node_feats = [n.to(device) for n in node_feats]
            # edge_feats = [e.to(device) for e in edge_feats]
            bg = bg.to(device)
            labels= labels.reshape(-1,1)
            labels = labels.to(device)
            # batched_graph = batched_graph.to(device)
            pred,_,_ = model(bg,1)
            test_loss += loss_fn(pred, labels).item()
            nan_mask = torch.isnan(pred)
            nan_positions = torch.nonzero(nan_mask, as_tuple=False).flatten()
            nan_labels = labels[nan_positions]
            unique_tensor = torch.unique(nan_labels, dim=0)
            unique_elements = [item[0] for item in unique_tensor.tolist()]
            test.append(unique_elements)
            if type_data == "regression":
                test_loss_MAE+=target(pred,labels).item()
            labels_all = torch.cat([labels_all, labels])
            pred_all = torch.cat([pred_all,pred])

    labels_all_cpu = labels_all.cpu().detach().numpy()
    pred_all_cpu = pred_all.cpu().detach().numpy()
    if type_data == "regression":
        test_loss_MAE /= num_batches
        result = test_loss_MAE
    if type_data =="binary classification":
        temp1 = pred_all_cpu
        y_scores_sigmoid = 1 / (1 + np.exp(-np.array(temp1)))
        #result = target(labels_all_cpu, pred_all_cpu)
        AUC = target(labels_all_cpu, y_scores_sigmoid)
        #threshold = 0.5
        #y_pred = (y_scores_sigmoid >= threshold).astype(int)
        #accuracy = accuracy_score(labels_all_cpu, y_pred)
        #mcc = matthews_corrcoef(labels_all_cpu, y_pred)
        #precision, recall, _ = precision_recall_curve(labels_all_cpu, y_scores_sigmoid)
        #auprc = auc(recall, precision)
        #f1 = f1_score(labels_all_cpu, y_pred)

        result = AUC

    test_loss /= num_batches
    return  (test_loss, result)


def main():
    seed_torch(args.seed)
    #train args
    epochs = args.epochs
    type_data = args.type_dataset
    early_stop = args.early_stop
    batch_size = args.batch_size
    lr = args.lr
    dropout = args.dropout
    num_layers = args.num_layers
    if type_data == "binary classification":
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        target = roc_auc_score

    if type_data == "regression":
        loss_fn = nn.SmoothL1Loss(reduction="mean")
        target = nn.L1Loss(reduction="mean")

    dataset_name = args.dataset
    # check cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #save path
    output_dir = PROJECT_ROOT / "output"
    file_savepath = output_dir / dataset_name
    result_save_path = output_dir

    file_savepath.mkdir(parents=True, exist_ok=True)
    result_save_path.mkdir(parents=True, exist_ok=True)
    print(file_savepath)

    '''dataset'''
    from dataset import TLDataset
    dataset = TLDataset(name=dataset_name, raw_dir=str(PROJECT_ROOT / "dataset"))


    '''k fold validation'''
    result = []
    # Define the K-fold Cross Validator
    kfold =  KFold(n_splits=5, shuffle=True, random_state=args.seed)
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'\nFOLD {fold}')
        print('--------------------------------')
        '''init model'''
        # model = GCNModelWithEdgeAFPreadout(node_in_dim=get_node_dim(), edge_in_dim=get_edge_dim(),hidden_feats=[200] * num_layers)
        model = GCNModelWithEdgeAFPreadout(node_in_dim=get_node_dim(), edge_in_dim=get_edge_dim(), hidden_feats=[200] * args.num_layers,
                                           output_norm="none", gru_out_layer=2, dropout=args.dropout)

        '''load best model params'''
        if args.best_model_file not in ["no"]:
            best_model_path = args.best_model_file
            checkpoint = torch.load(best_model_path, map_location=device)  # 加载断点
            model.load_state_dict(checkpoint)
            print(f"model loaded from: {best_model_path}")
        model.to(device)

        print('----args----')
        print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))
        print('----model----')
        # print(model)
        print(f"---------params-------------")
        print(f"all params: {count_parameters(model)}\n"
              f"trainable params: {count_trainable_parameters(model)}\n"
              f"freeze params: {count_no_trainable_parameters(model)}\n")

        '''data_loader'''
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, sampler=train_subsampler)
        test_dataloader = GraphDataLoader(dataset, batch_size=len(test_subsampler), sampler = test_subsampler, shuffle=False)

        # log_file
        best_loss = float("inf")
        best_model_stat = copy.deepcopy(model.state_dict())
        times, log_file = [], []

        print('---------- Training ----------')
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, verbose=True)

        # training & validation & testing
        for i in range(epochs):
            t1 = time.time()
            train_loss = train(model, device, train_dataloader, optimizer, loss_fn,target,type_data)
            test_loss, test_MAE = test(model, device, test_dataloader, loss_fn, target,type_data)
            t2 = time.time()
            times.append(t2 - t1)

            print(
                f'Epoch {i} |lr: {optimizer.param_groups[0]["lr"]:.6f} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | '
                f' time/epoch: {sum(times) / len(times):.1f}')

            # local file log
            log_file_loop = [i, optimizer.param_groups[0]["lr"], train_loss, test_loss, test_MAE]
            log_file.append(log_file_loop)
            scheduler.step()

            if test_loss < best_loss:
                es = 0
                best_loss = test_loss
                best_model_stat = copy.deepcopy(model.state_dict())
            else:
                es += 1
                print("Counter {} of {}".format(es, early_stop))
                # early stopping
                if es > early_stop:
                    print("Early stop, best_loss: ", best_loss)
                    break

        save_log_file(best_model_stat, log_file, file_savepath, fold)




def save_log_file(best_model_stat, log_file, file_savepath, fold):
    result = pd.DataFrame(log_file)
    result.columns = ["epoch", "lr", "train_loss",  "test_loss", "test_MAE"]
    result.to_csv(os.path.join(file_savepath, f"fold_{fold}_log_file.csv"))

    # print min
    index = result.iloc[:,3].idxmin(axis =0)
    print("the index of min loss is shown as follows:",result.iloc[index, :])
    torch.save(best_model_stat, os.path.join(file_savepath, f"fold_{fold}_best_model_weight.pth"))

    # return result["test_loss"].idxmin(axis =0)




if __name__ == '__main__':
    """
    Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description='GNN_RT_MODEL')
    #wandb name, dataset name, model name
    parser.add_argument('--name', type=str, default="test", help='wandb_running_name')
    parser.add_argument('--dataset', type=str, default='transfer learning', help='Name of dataset.')
    parser.add_argument('--model_name', type=str, default='GCN_edge_attention_GRU', help='Name of model, choose from: GAT, GCN, GIN, AFP, DEEPGNN')

    # GNN model args
    parser.add_argument('--num_layers', type=int, default=16, help='Number of GNN layers.')
    parser.add_argument('--hid_dim', type=int, default=200, help='Hidden channel size.')

    # training args
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--early_stop', type=int, default=50, help='Early stop epoch.')

    parser.add_argument('--seed', type=int, default=0, help='set seed')
    parser.add_argument('--best_model_file', type=str, default='no', help='best model')
    # kind of dataset
    parser.add_argument('--type_dataset', type=str, default='regression', help='regression or binary classification')
    args = parser.parse_args()
    print(args)

    main()

