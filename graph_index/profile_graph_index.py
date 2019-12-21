import dgl
from dgl.graph_index import *
from dgl.utils import toindex
import dgl.ndarray as nd
import dgl.backend as F

import time
import numpy as np

def gen_table(name, rst):
    print()
    print('=== %s ===' % name)
    maxlen = 10
    celltop = '-' * maxlen
    hline = celltop.join(['+', '+', '+'])
    for k, t in rst:
        print(hline)
        print('| {:8s} | {:.4f}   |'.format(str(k), t))
    print(hline)
    print()

def prof_add_nodes(g):
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        print('Profiling add nodes:', n)
        t0 = time.time()
        for i in range(10):
            g.add_nodes(n)
        rst.append(('%.1E' % n, (time.time() - t0)/10))
        g.clear()
    g.clear()
    return rst

def prof_clear_nodes(g):
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        print('Profiling clear nodes:', n)
        dur = []
        for i in range(10):
            g.add_nodes(n)
            t0 = time.time()
            g.clear()
            dur.append(time.time() - t0)
        rst.append(('%.1E' % n, np.average(dur)))
    g.clear()
    return rst

def prof_add_edges(g):
    # add E = 5 * V
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        print('Profiling add edges:', m)
        u = np.random.randint(0, n, m)
        v = np.random.randint(0, n, m)
        u = toindex(u)
        v = toindex(v)
        u.todgltensor()
        v.todgltensor()
        dur = []
        for i in range(5):
            g.add_nodes(n)
            t0 = time.time()
            g.add_edges(u, v)
            dur.append(time.time() - t0)
            g.clear()
        rst.append(('%.1E' % m, np.average(dur)))
    g.clear()
    return rst

def prof_extend(g):
    old_n = 1000000
    old_m = 5 * old_n
    old_u = np.random.randint(0, old_n, old_m)
    old_v = np.random.randint(0, old_n, old_m)
    old_u = toindex(old_u)
    old_v = toindex(old_v)
    old_u.todgltensor()
    old_v.todgltensor()
    rst = []

    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        print('Profiling add edges:', m)
        u = np.random.randint(0, old_n + n, m)
        v = np.random.randint(0, old_n + n, m)
        u = toindex(u)
        v = toindex(v)
        u.todgltensor()
        v.todgltensor()
        dur = []

        for i in range(5):
            g.add_nodes(old_n)
            g.add_edges(old_u, old_v)
            t0 = time.time()
            g.add_nodes(n)
            g.add_edges(u, v)
            dur.append(time.time() - t0)
            g.clear()
        rst.append(('%.1E' % m, np.average(dur)))
    g.clear()
    return rst

def prof_clear_edges(g):
    # clear E = 5 * V
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        print('Profiling clear edges:', m)
        u = np.random.randint(0, n, m)
        v = np.random.randint(0, n, m)
        u = toindex(u)
        v = toindex(v)
        u.todgltensor()
        v.todgltensor()
        dur = []
        for i in range(5):
            g.add_nodes(n)
            g.add_edges(u, v)
            t0 = time.time()
            g.clear()
            dur.append(time.time() - t0)
        rst.append(('%.1E' % m, np.average(dur)))
    g.clear()
    return rst

def prof_has_nodes(g):
    # look for 5% of the nodes
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        nn = int(n * .05)
        print('Profiling has nodes:', nn)
        g.add_nodes(n)
        vids = np.random.randint(0, n, nn)
        vids = toindex(vids)
        vids.todgltensor()
        t0 = time.time()
        for i in range(10):
            g.has_nodes(vids)
        rst.append(('%.1E' % nn, (time.time() - t0)/10))
        g.clear()
    return rst

def prof_has_edges(g):
    # E = 5 * V
    # look for 5% of the edges
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        mm = int(m * .05)
        print('Profiling has edges:', mm)
        u = np.random.randint(0, n, m)
        v = np.random.randint(0, n, m)
        u = toindex(u)
        v = toindex(v)
        u.todgltensor()
        v.todgltensor()
        g.add_nodes(n)
        g.add_edges(u, v)
        uu = np.random.randint(0, n, mm)
        vv = np.random.randint(0, n, mm)
        uu = toindex(uu)
        vv = toindex(vv)
        uu.todgltensor()
        vv.todgltensor()
        t0 = time.time()
        for i in range(10):
            g.has_edges_between(uu, vv)
        rst.append(('%.1E' % mm, (time.time() - t0)/10))
        g.clear()
    return rst


def prof_edge_ids(g):
    # E = 5 * V
    # look for all the edges
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        print('Profiling edge ids:', m)
        u = np.random.randint(0, n, m)
        v = np.random.randint(0, n, m)
        u = toindex(u)
        v = toindex(v)
        u.todgltensor()
        v.todgltensor()
        g.add_nodes(n)
        g.add_edges(u, v)
        t0 = time.time()
        for i in range(5):
            g.edge_ids(u, v)
        rst.append(('%.1E' % m, (time.time() - t0)/5))
        g.clear()
    return rst


def prof_in_edges(g):
    # E = 5 * V
    # look for all the nodes
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        print('Profiling in edges:', n)
        u = toindex(np.random.randint(0, n, m))
        v = toindex(np.random.randint(0, n, m))
        u.todgltensor()
        v.todgltensor()
        g.add_nodes(n)
        g.add_edges(u, v)
        vids = toindex(np.arange(n))
        vids.todgltensor()
        t0 = time.time()
        for i in range(10):
            g.in_edges(vids)
        rst.append(('%.1E' % n, (time.time() - t0)/10))
        g.clear()
    return rst


def prof_out_edges(g):
    # E = 5 * V
    # look for all the nodes
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        print('Profiling out edges:', n)
        u = toindex(np.random.randint(0, n, m))
        v = toindex(np.random.randint(0, n, m))
        u.todgltensor()
        v.todgltensor()
        g.add_nodes(n)
        g.add_edges(u, v)
        vids = toindex(np.arange(n))
        vids.todgltensor()
        t0 = time.time()
        for i in range(10):
            g.out_edges(vids)
        rst.append(('%.1E' % n, (time.time() - t0)/10))
        g.clear()
    return rst


def prof_edges(g, sorted):
    # E = 5 * V
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        print('Profiling edges:', n)
        u = toindex(np.random.randint(0, n, m))
        v = toindex(np.random.randint(0, n, m))
        u.todgltensor()
        v.todgltensor()
        g.add_nodes(n)
        g.add_edges(u, v)
        t0 = time.time()
        for i in range(10):
            g.edges(sorted)
        rst.append(('%.1E' % n, (time.time() - t0)/10))
        g.clear()
    return rst


def prof_in_degrees(g):
    # E = 5 * V
    # look for all the nodes
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        print('Profiling in degrees:', n)
        u = toindex(np.random.randint(0, n, m))
        v = toindex(np.random.randint(0, n, m))
        u.todgltensor()
        v.todgltensor()
        g.add_nodes(n)
        g.add_edges(u, v)
        vids = toindex(np.arange(n))
        vids.todgltensor()
        t0 = time.time()
        for i in range(10):
            g.in_degrees(vids)
        rst.append(('%.1E' % n, (time.time() - t0)/10))
        g.clear()
    return rst

def prof_out_degrees(g):
    # E = 5 * V
    # look for all the nodes
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        print('Profiling out degrees:', n)
        u = toindex(np.random.randint(0, n, m))
        v = toindex(np.random.randint(0, n, m))
        u.todgltensor()
        v.todgltensor()
        g.add_nodes(n)
        g.add_edges(u, v)
        vids = toindex(np.arange(n))
        vids.todgltensor()
        t0 = time.time()
        for i in range(10):
            g.out_degrees(vids)
        rst.append(('%.1E' % n, (time.time() - t0)/10))
        g.clear()
    return rst


def prof_node_subgraph(g, pert):
    # E = 5 * V
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        nn = int(n * pert)
        u = toindex(np.random.randint(0, n, m))
        v = toindex(np.random.randint(0, n, m))
        u.todgltensor()
        v.todgltensor()
        g.add_nodes(n)
        g.add_edges(u, v)
        vids = toindex(np.unique(np.random.randint(0, n, nn)))
        vids.todgltensor()
        print('Profiling node subgraph:', len(vids))
        t0 = time.time()
        for i in range(5):
            g.node_subgraph(vids)
        rst.append((n, (time.time() - t0)/5))
        g.clear()
    return rst


def prof_adjmat(g):
    # E = 5 * V
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        u = toindex(np.random.randint(0, n, m))
        v = toindex(np.random.randint(0, n, m))
        u.todgltensor()
        v.todgltensor()
        g.add_nodes(n)
        g.add_edges(u, v)
        print('Profiling adjacency matrix:', n)
        dur = []
        for i in range(10):
            t0 = time.time()
            g.adjacency_matrix(False, F.cpu()).get(nd.cpu())
            dur.append(time.time() - t0)
            g._cache.clear()
        rst.append((n, np.average(dur)))
        g.clear()
    return rst


def prof_incmat(g):
    # E = 5 * V
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 5 * n
        u = toindex(np.random.randint(0, n, m))
        v = toindex(np.random.randint(0, n, m))
        u.todgltensor()
        v.todgltensor()
        g.add_nodes(n)
        g.add_edges(u, v)
        print('Profiling incidence matrix:', n)
        dur = []
        for i in range(10):
            t0 = time.time()
            g.incidence_matrix(False, F.cpu()).get(nd.cpu())
            dur.append(time.time() - t0)
            g._cache.clear()
        rst.append((n, np.average(dur)))
        g.clear()
    return rst


def prof_line_graph(g):
    # E = 1 * V
    rst = []
    for k in [.1, 1, 10, 100, 1000]:
        n = int(1000 * k)
        m = 1 * n
        u = toindex(np.random.randint(0, n, m))
        v = toindex(np.random.randint(0, n, m))
        u.todgltensor()
        v.todgltensor()
        g.add_nodes(n)
        g.add_edges(u, v)
        print('Profiling line graph:', n)
        t0 = time.time()
        for i in range(5):
            g.line_graph()
        rst.append(('%.1E' % n, (time.time() - t0)/5))
        g.clear()
    return rst


def prof_union():
    # E = 5 * V
    # Merge different number of graphs.
    # Each size is 50, 100, 500, 1K
    def _create(n):
        m = 5 * n
        g = create_graph_index()
        g.add_nodes(n)
        u = toindex(np.random.randint(0, n, m))
        v = toindex(np.random.randint(0, n, m))
        g.add_edges(u, v)
        return g
    rst = []
    for n in [50, 100, 500, 1000]:
        for k in [5, 10, 50, 100, 500]:
            graphs = [_create(n) for i in range(k)]
            print('Profiling union %d graph(n=%d):' % (k, n))
            t0 = time.time()
            for i in range(10):
                disjoint_union(graphs)
            rst.append(('%dx%d' % (n, k), (time.time() - t0)/10))
    return rst


def prof_partition():
    # E = 5 * V
    # Merge different number of graphs.
    # Each size is 50, 100, 500, 1K
    def _create(n):
        m = 5 * n
        g = create_graph_index()
        g.add_nodes(n)
        u = toindex(np.random.randint(0, n, m))
        v = toindex(np.random.randint(0, n, m))
        g.add_edges(u, v)
        return g
    rst = []
    for n in [50, 100, 500, 1000]:
        for k in [5, 10, 50, 100, 500]:
            graphs = [_create(n) for i in range(k)]
            print('Profiling partition %d graph(n=%d):' % (k, n))
            bg = disjoint_union(graphs)
            t0 = time.time()
            for i in range(10):
                disjoint_partition(bg, k)
            rst.append(('%dx%d' % (n, k), (time.time() - t0)/10))
    return rst


g = create_graph_index(multigraph=True)
rst = prof_add_nodes(g)
gen_table('add nodes', rst)

rst = prof_clear_nodes(g)
gen_table('clear nodes', rst)

rst = prof_add_edges(g)
gen_table('add edges', rst)

rst = prof_extend(g)
gen_table('extend', rst)

rst = prof_clear_edges(g)
gen_table('clear edges', rst)

rst = prof_has_nodes(g)
gen_table('has nodes (5%)', rst)

rst = prof_has_edges(g)
gen_table('has edges (5%)', rst)

rst = prof_edge_ids(g)
gen_table('edge ids', rst)

rst = prof_in_edges(g)
gen_table('in edges', rst)

rst = prof_out_edges(g)
gen_table('out edges', rst)

rst = prof_edges(g, sorted=False)
gen_table('edges', rst)

rst = prof_edges(g, sorted=True)
gen_table('edges sorted', rst)

rst = prof_in_degrees(g)
gen_table('in degrees', rst)

rst = prof_out_degrees(g)
gen_table('out degrees', rst)

rst = prof_node_subgraph(g, .01)
gen_table('node subgraph (1%)', rst)

rst = prof_node_subgraph(g, .05)
gen_table('node subgraph (5%)', rst)

rst = prof_node_subgraph(g, .1)
gen_table('node subgraph (10%)', rst)

rst = prof_node_subgraph(g, .2)
gen_table('node subgraph (20%)', rst)

rst = prof_adjmat(g)
gen_table('adjmat', rst)

rst = prof_incmat(g)
gen_table('incmat', rst)

rst = prof_line_graph(g)
gen_table('line graph (density 1/n)', rst)

rst = prof_union()
gen_table('disjoint union graph', rst)

rst = prof_partition()
gen_table('disjoint partition graph', rst)
