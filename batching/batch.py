import dgl
import torch
import time
import numpy as np
import pandas as pd

def create_graph(n):
    m = 5 * n
    g = dgl.DGLGraph()
    g.add_nodes(n)
    g.add_edges(np.random.randint(0, n, m), np.random.randint(0, n, m))
    return g

def prof_batching():
    perf = []
    for n in [50, 100]:
        for f in [2, 4, 8, 16]:
            for b in [16, 64, 256, 1024]:
                gs = []
                for i in range(b):
                    g = create_graph(n)
                    g.set_n_repr({'h%d' % i: torch.randn(g.number_of_nodes(), 25) for i in range(f)})
                    g.set_e_repr({'a%d' % i: torch.randn(g.number_of_edges(), 5) for i in range(f)})
                    gs.append(g)
                print('Profiling b={b}, n={n}, f={f}'.format(b=b, n=n, f=f))

                ts_batch = 0
                ts_unbatch = 0
                ts_batch_with_repr = 0
                ts_unbatch_with_repr = 0
                ts_readout = 0

                for i in range(10):
                    t0 = time.time()
                    gb = dgl.batch(gs)
                    ts_batch += time.time() - t0

                    t0 = time.time()
                    gs = dgl.unbatch(gb)
                    ts_unbatch += time.time() - t0

                    t0 = time.time()
                    gb = dgl.batch(gs)
                    ts_batch_with_repr += time.time() - t0

                    # TODO: use batched readout function
                    t0 = time.time()
                    hs = gb.get_n_repr()['h0'].split(gb.batch_num_nodes)
                    m = [h.mean(0) for h in hs]
                    ts_readout += time.time() - t0

                    t0 = time.time()
                    gs = dgl.unbatch(gb)
                    m = [g.get_n_repr()['h0'].mean(0) for g in gs]
                    ts_unbatch_with_repr += time.time() - t0

                perf.append({
                    '# graphs': b,
                    '# nodes': n,
                    '# features': f,
                    'batch time': ts_batch / 10,
                    'unbatch time': ts_unbatch / 10,
                    'batch time w/ repr': ts_batch_with_repr / 10,
                    'unbatch time w/ repr': ts_unbatch_with_repr / 10,
                    'readout': ts_readout / 10,
                    })
    return pd.DataFrame(perf)

perf = prof_batching()
print(perf)
perf.to_csv('batch.csv')
