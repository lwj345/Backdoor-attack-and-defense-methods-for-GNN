import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
from torch.nn import init
import torch_sparse
import numpy as np
import random

class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        if isinstance(adj, torch_sparse.SparseTensor):
            output = torch_sparse.matmul(adj, support)
        else:
            output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, with_bn=False, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers-2):
                self.layers.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nhid, nclass, with_bias=with_bias))

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def forward(self, x, adj):
        # import pdb;pdb.set_trace()
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_rand_smooth(self, x, adj, rate_prune_edge=0.2):

        for ix, layer in enumerate(self.layers):
            if utils.is_sparse_tensor(adj):
                values_ = adj.coalesce().values()
                indices_ = adj.coalesce().indices()
                device = adj.device
                size = adj.size()
                n_values = values_.shape[0]
                total_idx = np.arange(n_values).tolist()
                selected_idx = random.sample(total_idx,int(n_values*(1-rate_prune_edge)))
                values_selected = values_[selected_idx]
                indices_selected = indices_[:,selected_idx]
                # import pdb;pdb.set_trace()
                adj_selected = torch.sparse_coo_tensor(indices=indices_selected, values=values_selected, size=[size[0], size[1]]).to(device)

                adj_norm = utils.normalize_adj_tensor(adj_selected, sparse=True)
            else:
                rand_num = torch.rand(adj.shape[0],adj.shape[1])
                prune_mask = rand_num.ge(rate_prune_edge).to(adj.device)
                adj = adj*prune_mask
                adj_norm = utils.normalize_adj_tensor(adj)
            # import pdb;pdb.set_trace()
            x = layer(x, adj_norm)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_x(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward_sampler(self, x, adjs):
        # for ix, layer in enumerate(self.layers):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler_syn(self, x, adjs):
        for ix, (adj) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def fit_with_val_poison(self, features, adj, labels, data, idx_train, train_iters=200, initialize=True, verbose=False, normalize=True, patience=None, noval=False, **kwargs):
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:
            self._train_with_val_poison(labels, data, idx_train, train_iters, verbose, adj_val=True)
        else:
            self._train_with_val_poison(labels, data, idx_train, train_iters, verbose)

    def _train_with_val_poison(self, labels, data, idx_train, train_iters, verbose, adj_val=False):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = self.loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)

                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def fit_with_val(self, features, adj, labels, data, train_iters=200, initialize=True, verbose=False, normalize=True, patience=None, noval=False, defense_type=None,prune_rate=0.2, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)
        # defense_type = 'prune'
        # if defense_type == 'prune':
        #     normalize = True
        #     feat_cosine = torch.mm(features,features.T)
        #     feat_cosine = feat_cosine.flatten()#.values = 
        #     sorted_cos = torch.sort(feat_cosine,descending=False)
        #     sorted_cos_values = sorted_cos.values
        #     split_value_index = int(sorted_cos.values.shape[0]*prune_rate)
        #     split_value = sorted_cos_values[split_value_index]
        #     mask = adj.ge(split_value)
        #     adj = adj * mask
        #     # mask = 
        if defense_type == 'prune':

            adj = adj.to_sparse()
            indices = adj.coalesce().indices()
            device = indices.device
            # feat_cosine = torch.mm(features,features.T)
            self_ = [i for i in range(indices.shape[1]) if indices[0][i]==indices[1][i]]
            filtered_ = [i for i in range(indices.shape[1]) if indices[0][i]!=indices[1][i]]
            self_indices = indices[:,self_]########
            filtered_indices = indices[:,filtered_]########
            filtered_features0 = features[filtered_indices[0]]
            filtered_features1 = features[filtered_indices[1]]
            multiply = filtered_features0*filtered_features1
            similarities = multiply.sum(axis=1)########
            sorted_sim = torch.sort(similarities,descending=False)
            sorted_sim_values = sorted_sim.values
            split_value_index = int(sorted_sim.values.shape[0]*prune_rate)
            split_value = sorted_sim_values[split_value_index]########
            mask = similarities.ge(split_value)
            filtered_indices = filtered_indices[:,mask]
            pruned_indices = torch.concat((self_indices,filtered_indices),dim=1)
            #接下来就是如何通过mask获得对应的index
            size_adj = adj.shape
            values = torch.ones_like(pruned_indices[1])
            adj = torch.sparse_coo_tensor(pruned_indices,values,size=size_adj,device=device)#.to(device)
            adj = utils.normalize_adj_tensor(adj, sparse=True)
            torch.cuda.empty_cache()

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels
        # defense_type = 'rand_smooth'
        if noval:
            self._train_with_val(labels, data, train_iters, verbose, defense_type, prune_rate, adj_val=True)
        else:
            self._train_with_val(labels, data, train_iters, verbose, defense_type, prune_rate)

    def _train_with_val(self, labels, data, train_iters, verbose, defense_type, prune_rate, adj_val=False):

        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            
            if defense_type == 'rand_smooth':
                # import pdb;pdb.set_trace()
                output = self.forward_rand_smooth(self.features, self.adj_norm, prune_rate)
            else:
                output = self.forward(self.features, self.adj_norm)
            
            loss_train = self.loss(output, labels)
            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)#################这里也得改改 用 forward_sampler

                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    @torch.no_grad()
    def predict(self, features=None, adj=None, defense_type=None, prune_rate=0.0):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            

            if defense_type == 'rand_smooth':
                # self.adj_norm
                output = self.forward_rand_smooth(self.features, self.adj_norm, prune_rate)
            else:
                output = self.forward(self.features, self.adj_norm)
            return output#self.forward(self.features, self.adj_norm)

    @torch.no_grad()
    def predict_unnorm(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            self.adj_norm = adj
            return self.forward(self.features, self.adj_norm)

    def _train_with_val2(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
