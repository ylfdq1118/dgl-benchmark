"""GCN using builtin functions that enables SPMV optimization.

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import argparse
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
import math

from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import dgl.function as fn

class GCNLayer(gluon.Block):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.g = g
        with self.name_scope():
            stdv = 1. / math.sqrt(out_feats)
            self.weight = self.params.get('weight', shape=(in_feats, out_feats),
                    init=mx.init.Uniform(stdv))
            if bias:
                self.bias = self.params.get('bias', shape=(out_feats,),
                    init=mx.init.Uniform(stdv))
            else:
                self.bias = None
        self.activation = activation
        self.dropout = dropout

    def forward(self, h):
        if self.dropout:
            h = mx.nd.Dropout(h, p=self.dropout)
        h = mx.nd.dot(h, self.weight.data(h.context))
        # normalization by square root of src degree
        h = h * self.g.ndata['norm']
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h = self.g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * self.g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias.data(h.context)
        if self.activation:
            h = self.activation(h)
        return h

class GCN(gluon.Block):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = gluon.nn.Sequential()
        # input layer
        self.layers.add(GCNLayer(g, in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(GCNLayer(g, n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.add(GCNLayer(g, n_hidden, n_classes, None, dropout))


    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h
def evaluate(model, features, labels, mask):
    pred = model(features).argmax(axis=1)
    accuracy = ((pred == labels) * mask).sum() / mask.sum().asscalar()
    return accuracy.asscalar()

def main(args):
    # load and preprocess dataset
    data = load_data(args)
    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    train_mask = mx.nd.array(data.train_mask.astype(np.uint8))
    val_mask = mx.nd.array(data.val_mask.astype(np.uint8))
    test_mask = mx.nd.array(data.test_mask.astype(np.uint8))
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.sum().asscalar(),
              val_mask.sum().asscalar(),
              test_mask.sum().asscalar()))

    if args.gpu < 0:
        cuda = False
        ctx = mx.cpu(0)
    else:
        cuda = True
        ctx = mx.gpu(args.gpu)

    features = features.as_in_context(ctx)
    labels = labels.as_in_context(ctx)
    train_mask = train_mask.as_in_context(ctx)
    val_mask = val_mask.as_in_context(ctx)
    test_mask = test_mask.as_in_context(ctx)

    # create GCN model
    g = data.graph
    g = DGLGraph(g)
    if not args.dataset.startswith("reddit"):
        g.add_edges(g.nodes(), g.nodes())
    # normalization
    degs = g.in_degrees().astype('float32')
    norm = mx.nd.power(degs, -0.5)
    if cuda:
        norm = norm.as_in_context(ctx)
    g.ndata['norm'] = mx.nd.expand_dims(norm, 1)

    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                mx.nd.relu,
                args.dropout)
    model.initialize(ctx=ctx)
    n_train_samples = train_mask.sum().asscalar()
    loss_fcn = gluon.loss.SoftmaxCELoss()

    # use optimizer
    print(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), 'adam',
            {'learning_rate': args.lr, 'wd': args.weight_decay})

    # initialize graph
    for epoch in range(args.n_epochs + 1):
        # forward
        with mx.autograd.record():
            pred = model(features)
            loss = loss_fcn(pred, labels, mx.nd.expand_dims(train_mask, 1))
            loss = loss.sum() / n_train_samples

        loss.backward()
        trainer.step(batch_size=1)

        loss = loss.asscalar()
        if epoch == 0:
            # skip first epoch for warm up
            mx.nd.waitall()
            start = time.time()
        mx.nd.waitall()
        end = time.time()
    # test set accuracy
    print(evaluate(model, features, labels, test_mask))
    print("{:.4f}".format(end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()
    if args.dataset.startswith("reddit"):
        # use reddit dataset that already added self-loop
        args.dataset = "reddit-self-loop"

    print(args)

    main(args)
