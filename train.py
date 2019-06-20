import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from gcn_ns_sc import gcn_ns_train
from gcn_cv_sc import gcn_cv_train
from graphsage_cv import graphsage_cv_train

import numpy.random as npr
import scipy.sparse as sps
import sbm

def main(args):
    # load and preprocess dataset
#   data = load_data(args)

    src = np.load('src.npy')
    dst = np.load('dst.npy')
    x = np.load('x.npy')
    y = np.load('y.npy')

    '''
    _n = 15000
    unique_y = np.unique(y)
    x = np.vstack([x[np.nonzero(y == i)[0][:_n]] for i in unique_y])
    y = np.repeat(unique_y, _n)
    k = len(unique_y)
    adj, _ = sbm.generate(_n * k, [_n] * k, np.eye(k) * 0.001)
    '''

    dat = np.ones_like(src)
    n = len(x)
    adj = sps.coo_matrix((dat, (src, dst)), shape=[n, n]).maximum(sps.eye(n))
#   adj = sps.eye(n, n)

    data = type('', (), {})
    data.graph = dgl.graph_index.create_graph_index(adj, readonly=True, multigraph=False)
    data.features = x
    data.labels = y
    data.num_labels = len(np.unique(y))
    data.train_mask = np.zeros(n)
    data.val_mask = np.zeros(n)
    data.test_mask = np.zeros(n)
    p = npr.permutation(n)
    data.train_mask[p[:args.n_train]] = 1
    data.val_mask[p[args.n_train : args.n_train + args.n_val]] = 1
    data.test_mask[p[args.n_train + args.n_val:]] = 1

    if args.gpu >= 0:
        ctx = mx.gpu(args.gpu)
    else:
        ctx = mx.cpu()

    if args.self_loop and not args.dataset.startswith('reddit'):
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    train_nid = mx.nd.array(np.nonzero(data.train_mask)[0]).astype(np.int64)
    test_nid = mx.nd.array(np.nonzero(data.test_mask)[0]).astype(np.int64)

    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    train_mask = mx.nd.array(data.train_mask)
    val_mask = mx.nd.array(data.val_mask)
    test_mask = mx.nd.array(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    n_train_samples = train_mask.sum().asscalar()
    n_val_samples = val_mask.sum().asscalar()
    n_test_samples = test_mask.sum().asscalar()

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              n_train_samples,
              n_val_samples,
              n_test_samples))

    # create GCN model
    g = DGLGraph(data.graph, readonly=True)
    '''
    g = DGLGraph()
    g.add_nodes(n)
    g.add_edges(src, dst)
    g.readonly()
    '''
    g.ndata['features'] = features
    g.ndata['labels'] = labels

    if args.model == "mlp":
        mlp_train(ctx, args, n_classes, features, labels, train_mask, val_mask, test_mask)
    elif args.model == "gcn_ns":
        gcn_ns_train(g, ctx, args, n_classes, train_nid, test_nid, n_test_samples)
    elif args.model == "gcn_cv":
        gcn_cv_train(g, ctx, args, n_classes, train_nid, test_nid, n_test_samples, False)
    elif args.model == "graphsage_cv":
        graphsage_cv_train(g, ctx, args, n_classes, train_nid, test_nid, n_test_samples, False)
    else:
        print("unknown model. Please choose from gcn_ns, gcn_cv, graphsage_cv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--model", type=str,
                        help="select a model. Valid models: gcn_ns, gcn_cv, graphsage_cv")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=3,
            help="number of neighbors to be sampled")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--nworkers", type=int, default=1,
            help="number of workers")
    parser.add_argument('--n-train', type=int)
    parser.add_argument('--n-val', type=int)
    args = parser.parse_args()

    print(args)

    main(args)
