from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from transformers import activations
import torch_geometric as tg
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    SuperGATConv,
    ResGatedGraphConv,
    GCN2Conv,
    GatedGraphConv,
    SAGEConv,
)
from torch_geometric.data import Data
import numpy as np
import torch
import torch.nn.functional as F

# from utils import load_data
import pandas as pd
import argparse
import joblib
import random
from graphadapter import LinearHead


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_mask(n, i, j, seed):
    np.random.seed(seed)
    randint = np.random.randint(0, 100, (n,))
    train_mask = torch.tensor((randint < i)).bool()
    val_mask = torch.tensor(((randint >= i) & (randint < j))).bool()
    test_mask = torch.tensor(((randint >= j) & (randint < 100))).bool()
    return train_mask, val_mask, test_mask


def normal(x):
    x = (x - x.mean(dim=0).view(1, -1)) / x.std(dim=0).view(1, -1)
    return x


def load_data_with_prompt_embedding(dataname, train_ratio, val_ratio, split):  ##
    if (train_ratio >= 100) or (val_ratio + train_ratio >= 100):
        raise "train or validation ratio out of 100"
    x = np.load(f"./token_embedding/{dataname}/sentence_embeddings.npy")
    edge_index = np.load("./datasets/" + dataname + "/edge_index.npy")
    y = np.load("./datasets/" + dataname + "/y.npy")
    x = torch.tensor(x).float()
    y = torch.tensor(y).long()
    edge_index = torch.tensor(edge_index).T
    edge_index = tg.utils.to_undirected(edge_index)
    edge_index = tg.utils.add_self_loops(edge_index)[0]
    edge_index = tg.utils.sort_edge_index(edge_index)
    data = Data()
    data.x = x.float()
    data.y = y
    if dataname != "arxiv":
        train_mask, val_mask, test_mask = get_mask(x.shape[0], train_ratio, train_ratio + val_ratio, split)
    else:
        train_mask = np.load("./datasets/" + dataname + "/train.npy")
        val_mask = np.load("./datasets/" + dataname + "/vaild.npy")
        test_mask = np.load("./datasets/" + dataname + "/test.npy")
    data.edge_index = edge_index
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def evaluate(out, label, mask, metric="acc"):
    if metric == "roc":
        py = out[:, 1][mask].cpu().numpy()
        # val = (out[data.val_mask]==data.y[data.val_mask]).sum()
        #  print(data.y)
        gy = label[mask].cpu().long().numpy()
        val = roc_auc_score(gy, py)
        return val
    elif metric == "acc":
        py = out.max(dim=1)[1][mask].cpu().numpy()
        # val = (out[data.val_mask]==data.y[data.val_mask]).sum()
        #  print(data.y)
        gy = label[mask].cpu().long().numpy()
        val = accuracy_score(gy, py)
        return val
    elif metric == "ap":
        py = out[:, 1][mask].cpu().numpy()
        # val = (out[data.val_mask]==data.y[data.val_mask]).sum()
        #  print(data.y)
        gy = label[mask].cpu().long().numpy()
        val = average_precision_score(gy, py)
        return val


def finetune(data, args):
    model = None
    device = args.device
    model = LinearHead(data.x.shape[1], int(data.y.max()) + 1, args)

    prompt_x = np.load("./prompt_embedding/" + args.dataset_name + "/prompt_embedding.npy")
    prompt_x = torch.tensor(prompt_x).float().to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.lin.parameters(),
                "lr": args.learning_rate,
                "weight_decay": 1e-3,
            },
            {
                "params": model.ga.parameters(),
                "lr": args.learning_rate,
                "weight_decay": 1e-3,
            },
        ],
    )

    data = data.to(device)
    model = model.to(device)

    loss = None
    val_acc = 0
    test = 0
    for i in range(35000):
        model.train()
        model.ga.train()
        optimizer.zero_grad()
        out, gate = model(data.x, data.edge_index, prompt_x)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            model.ga.eval()
            out, eval_gate = model(data.x, data.edge_index, prompt_x)
            val = evaluate(out, data.y, data.val_mask)
            if val >= val_acc:
                test = evaluate(out, data.y, data.test_mask)
                tr = evaluate(out, data.y, data.train_mask)
                print(f"best in epoch {i}: train:{tr:.4f},valid:{val:.4f},test:{test:.4f}")
                val_acc = val
                duration = 0
    print("final_loss", loss.item())
    model.eval()
    return test
