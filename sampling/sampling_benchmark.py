import mxnet as mx
import numpy as np
import time
import argparse
import dgl
from dgl.graph import create_graph_index
from dgl import utils
from dgl.data import register_data_args, load_data

class MXNetGraph(object):
    """A simple graph object that uses scipy matrix."""
    def __init__(self, mat):
        self._mat = mat

    def get_graph(self):
        return self._mat

    def number_of_nodes(self):
        return self._mat.shape[0]

    def number_of_edges(self):
        return mx.nd.contrib.getnnz(self._mat).asnumpy()[0]

class GraphData:
    def __init__(self, csr):
        num_edges = mx.nd.contrib.getnnz(csr).asnumpy()[0]
        edge_ids = mx.nd.arange(0, num_edges, step=1, repeat=1, dtype=np.int64)
        csr = mx.nd.sparse.csr_matrix((edge_ids, csr.indices, csr.indptr), shape=csr.shape, dtype=np.int64)
        self.graph = MXNetGraph(csr)
        self.features = mx.nd.random.normal(shape=(csr.shape[0]))
        self.labels = mx.nd.floor(mx.nd.random.normal(loc=0, scale=10, shape=(csr.shape[0])))
        self.num_labels = 10

def test_subgraph_gen(args):
    # load and preprocess dataset
    t0 = time.time()
    if args.graph_file != '':
        csr = mx.nd.load(args.graph_file)[0]
        data = GraphData(csr)
        csr = None
    else:
        data = load_data(args)
    graph = data.graph
    try:
        graph_data = graph.get_graph()
        print("#nodes: " + str(graph.number_of_nodes())
                + ", #edges: " + str(graph.number_of_edges()) + " edges")
    except:
        graph_data = graph
    print("load data: " + str(time.time() - t0))

    t0 = time.time()
    ig = dgl.DGLGraph(graph_data, readonly=True)
    print("create immutable graph index: " + str(time.time() - t0))

    if args.neigh_expand <= 0:
        neigh_expand = ig.number_of_nodes()
    else:
        neigh_expand = args.neigh_expand
    it = dgl.contrib.sampling.NeighborSampler(ig, args.subgraph_size, neigh_expand,
                                              neighbor_type='in', num_workers=args.n_parallel,
                                              shuffle=True)
    try:
        for _ in range(2500):
            t0 = time.time()
            num_nodes = 0
            num_edges = 0
            for _ in range(args.n_parallel):
                subg, seeds = next(it)
                num_nodes += subg.number_of_nodes()
                num_edges += subg.number_of_edges()
                mx.nd.waitall()
            t1 = time.time()
            print("subgraph on immutable graphs {:4f}, #nodes {:d} | #edges {:d}".format(
                t1 - t0, num_nodes, num_edges))
    except:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='subgraph')
    register_data_args(parser)
    parser.add_argument("--subgraph-size", type=int, default=1000,
            help="The number of seed vertices in a subgraph.")
    parser.add_argument("--n-parallel", type=int, default=1,
            help="the number of subgraph construction in parallel.")
    parser.add_argument("--graph-file", type=str, default="",
            help="graph file")
    parser.add_argument("--neigh-expand", type=int, default=16,
            help="the number of neighbors to sample.")
    args = parser.parse_args()

    test_subgraph_gen(args)

