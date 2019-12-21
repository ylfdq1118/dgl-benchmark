import torch
import numpy as np
import itertools
import time
import pandas as pd
from collections import defaultdict

class Timer(object):
    def __init__(self):
        self.start = 0
        self.end = 0

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.time()

    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.end = time.time()
        self.dur = self.end - self.start

B = [20, 80, 400, 2000]
N = [10, 50, 250, 1500]
F = [50, 100, 200]
T = 100
records = []

for n_graphs, graph_size, feature_dims in itertools.product(B, N, F):
    print(n_graphs, graph_size, feature_dims)
    offsets = (graph_size,) * n_graphs
    offsets_arr = np.array(offsets)
    offsets_tensor = torch.from_numpy(offsets_arr).cuda()
    idx = torch.from_numpy(np.arange(n_graphs).repeat(offsets)).cuda()
    n_nodes = sum(offsets)
    x = torch.randn(n_nodes, feature_dims).cuda().requires_grad_()
    w = torch.sparse_coo_tensor(torch.stack([idx, torch.arange(n_nodes).cuda()], 0), torch.ones(n_nodes).cuda())
    w_mean = torch.from_numpy((1. / offsets_arr).repeat(offsets)).cuda().float()
    w_mean = torch.sparse_coo_tensor(torch.stack([idx, torch.arange(n_nodes).cuda()], 0), w_mean)
    record = defaultdict(float)
    torch.cuda.synchronize()
    t = Timer()

    # warm up
    for i in range(20):
        y1 = torch.zeros(n_graphs, feature_dims).cuda().scatter_add_(0, idx[:, None].expand(n_nodes, feature_dims), x)
        w = torch.sparse_coo_tensor(torch.stack([idx, torch.arange(n_nodes).cuda()], 0), torch.ones(n_nodes).cuda())
        y2 = w @ x

    for i in range(T):
        # scatter_add
        with t:
            y1 = torch.zeros(n_graphs, feature_dims).cuda().scatter_add_(0, idx[:, None].expand(n_nodes, feature_dims), x)
        record['scatter_add_fw'] += t.dur
        loss = y1.sum()
        with t:
            loss.backward()
        record['scatter_add_bw'] += t.dur
        x.grad.zero_()

        # spmv
        with t:
            y2 = torch.spmm(w, x)
        record['spmv_add_fw'] += t.dur
        loss = y2.sum()
        with t:
            loss.backward()
        record['spmv_add_bw'] += t.dur
        x.grad.zero_()

        # split
        with t:
            xs = x.split(offsets)
            y3 = torch.stack([_x.sum(0) for _x in xs], 0)
        record['split_fw'] += t.dur
        loss = y3.sum()
        with t:
            loss.backward()
        record['split_bw'] += t.dur
        x.grad.zero_()

        # scatter_mean
        with t:
            y1 = torch.zeros(n_graphs, feature_dims).cuda().scatter_add_(0, idx[:, None].expand(n_nodes, feature_dims), x)
            y1 /= offsets_tensor.float()[:, None]
        record['scatter_mean_fw'] += t.dur
        loss = y1.sum()
        with t:
            loss.backward()
        record['scatter_mean_bw'] += t.dur
        x.grad.zero_()

        # spmv
        with t:
            y2 = torch.spmm(w_mean, x)
        record['spmv_mean_fw'] += t.dur
        loss = y2.sum()
        with t:
            loss.backward()
        record['spmv_mean_bw'] += t.dur
        x.grad.zero_()

        # split
        with t:
            xs = x.split(offsets)
            y3 = torch.stack([_x.mean(0) for _x in xs], 0)
        record['split_mean_fw'] += t.dur
        loss = y3.sum()
        with t:
            loss.backward()
        record['split_mean_bw'] += t.dur
        x.grad.zero_()

    for k in record:
        record[k] /= T
    record['B'] = n_graphs
    record['N'] = graph_size
    record['F'] = feature_dims
    records.append(record)

df = pd.DataFrame(records)
df.to_csv('readout.csv', float_format='%.7f')
