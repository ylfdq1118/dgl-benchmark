import dgl
import unittest
import os
import numpy as np
import time
from dgl.distributed import DistGraph, DistDataLoader
import argparse


class NeighborSampler(object):
    def __init__(self, g, fanouts, sample_neighbors):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors

    def sample_blocks(self, seeds):
        import torch as th
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks

def start_client(args):
    import dgl
    import torch as th
    os.environ['DGL_DIST_MODE'] = 'distributed'
    dgl.distributed.init_rpc(args.ip_config, num_workers=args.num_workers)
    gpb = None
    _, graph_name = dgl.distributed.load_partition_book(args.conf_file, 0)
    g = DistGraph(args.ip_config, graph_name)

    # Create sampler
    fanouts = [int(fanout) for fanout in args.fan_out.split(',')]
    sampler = NeighborSampler(g, fanouts, dgl.distributed.sample_neighbors)

    # Create PyTorch DataLoader for constructing blocks
    dataloader = DistDataLoader(
        dataset=np.arange(g.number_of_nodes()),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=False,
        drop_last=False)
    
    print('before sampling')
    start = time.time()
    num_batches = 0
    for idx, block in enumerate(dataloader):
        num_batches += 1
        if num_batches % 1000 == 0:
            print('sample {} batches takes {} seconds, rate={}'.format(num_batches,
                                                                       time.time() - start,
                                                                       num_batches / (time.time() - start)))
    dgl.distributed.shutdown_servers()
    dgl.distributed.finalize_client()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("sampling benchmark")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--num-workers', type=int, default=1,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    argparser.add_argument('--conf_file', type=str, help='The path to the partition config file')
    args = argparser.parse_args()

    sampled_graph = start_client(args)
