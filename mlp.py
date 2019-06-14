from mxnet import gluon
import mxnet.ndarray as nd

class MLP(gluon.Block):
    def __init__(self, n_feats, activation, dropout):
        super(MLP, self).__init__()
        self.layers = 
        self.activation = activation

def mlp_train(ctx, args, n_classes, features, labels, train_mask, val_mask, test_mask):
    pass
