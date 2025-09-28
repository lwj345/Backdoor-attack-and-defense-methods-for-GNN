# follow https://github.com/ventr1c/UGBA/blob/main/models/backdoor.py 写一个backdoor的类
# 然后他serve as https://github.com/liuyugeng/baadd/blob/master/DC/utils.py里的 def update_trigger的功能对应着backdoor的fit function
# 看一下UGBA的是怎么找到attach的nodes的。
# 你得确定一下之前的paper都是怎么attack graph的-》UBGA： 先得到poisoned graph然后再train GNNs

#你得确定一下NDSS的 update_trigger都是加在哪些图片上的，一如如何更新的（按照哪个label更新的）
# NDSS是把被poisoned的images的label改成target label
#*******************
#得看一下那些个edge_weights是干什么用的
from models.gcn import GCN#改成gcond的
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_sparse import SparseTensor
import deeprobust.graph.utils as utils
from tqdm import tqdm, trange
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1

class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat,nout*nfeat)
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2))
        self.device = device

    def forward(self, input, thrd):
        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """
        #this should also generate the edges between the triggers and their attached_nodes.
        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)
        feat = self.feat(h)
        edge_weight = self.edge(h)
        edge_weight = GW(edge_weight, thrd, self.device)
        return feat, edge_weight

class BackdoorGC:
    
    def __init__(self,args,device,idx_attach,features,adj,labels):
        #threshold, trigger_size, hidden, lr_model, lr_trigger, weight_decay, target_class, outer_epochs_backdoor, inner_epochs, seed, target_loss_weight, device
        self.args = args
        self.device = device
        self.seed=args.seed

        self.threshold=args.threshold
        self.trigger_size=args.trigger_size
        self.lr_trigger=args.lr_trigger
        self.target_class=args.target_class
        self.target_loss_weight=args.target_loss_weight
        self.trigger_index = self.get_trigger_index()
        self.original_adj = adj.clone()
        self.original_features = features.clone()
        self.labels = labels.clone()

        self.idx_attach = idx_attach
        self.hidden=args.hidden
        self.lr_model=args.lr_model
        self.weight_decay=args.weight_decay
        # self.inner_epochs=args.inner_epochs
        self.outer_epochs=args.outer_epochs_backdoor

        self.trojan = GraphTrojanNet(self.device, features.shape[1],self.trigger_size,layernum=2).to(self.device)
        self.optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.lr_trigger, weight_decay=self.weight_decay)
        
        self.weights = None

    def to_device(self, device):
        self.trigger_index = self.trigger_index.to(device)
        self.original_adj = self.original_adj.to(device)
        self.device = device
        self.original_features=self.original_features.to(device)
        self.trojan = self.trojan.to(device)

    def get_trigger_index(self):#devised for implementing the get_trojan_edge()
        trigger_size = self.trigger_size
        edge_list = []
        edge_list.append([0,0])# this is for the index between the node0 in the trigger and the poisoned node
        for j in range(trigger_size):
            for k in range(j+1):
                edge_list.append([j,k])
        edge_index = torch.tensor(edge_list,device=self.device).long().T
        return edge_index

    def get_trojan_edge(self,start, idx_attach=None):#generate the index for the edges within the triggers and the edges between node0 in trigger and poisoned node.1
        edge_list = []
        if idx_attach is None:
            idx_attach = self.idx_attach
        # start0 = start
        # start0 - idx_attach_num + idx
        trigger_size = self.trigger_size
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0,0] = idx
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            start += trigger_size

        edge_index = torch.cat(edge_list,dim=1)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])
        return edge_index

    def get_trojan_edge_induct(self,start,idx_attach):#generate the index for the edges within the triggers and the edges between node0 in trigger and poisoned node.
        edge_list = []
        trigger_size = self.trigger_size
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0,0] = idx
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list,dim=1)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])
        return edge_index

    def get_poisoned(self,idx_attach):#,edge_index,edge_weight):
        '''
        return normalized dense adj matrix and the features, which are all poisoned
        '''
        adj = self.original_adj.clone()#adj_input.clone()
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)##############Here we can convert sparse adj to normalized_adj
        else:
            adj_norm = utils.normalize_adj_tensor(adj)
        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()

        # idx_attach = idx_outter#self.idx_attach
        features = self.original_features
        self.trojan.eval()

        trojan_edge = self.get_trojan_edge(len(features)).to(self.device)#those edges are among the trigger nodes, the edges here could be converted into adj, (node_h, node_t)
        # try:

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.threshold)
        # except RuntimeError as e:
        #     import pdb;pdb.set_trace()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])
        updated_feat = torch.cat([features,trojan_feat])
        trojan_labels = torch.ones(trojan_feat.shape[0], dtype=torch.long)*self.target_class

        updated_labels = torch.cat([self.labels.clone(),trojan_labels.to(self.labels.device)])

        indices_trojan_edges = trojan_weights>=0
        # indices_trojan_edges[0][0] = True
        edges_attack_trigger = [[],[]]
        if (indices_trojan_edges).nonzero().shape[0] != 0:
            indices = (indices_trojan_edges).nonzero()
            
            for i in range(idx_attach.shape[0]):
                for j in range(self.trigger_size):
                    if j !=0:
                        edges_attack_trigger[0].append(idx_attach[i])
                        edges_attack_trigger[1].append(len(features)+i*self.trigger_size+j)
            
            row = torch.tensor(edges_attack_trigger[0], dtype=torch.long).to(trojan_edge.device)#torch.cat([edges_attack_trigger[0], edges_attack_trigger[1]])
            col = torch.tensor(edges_attack_trigger[1], dtype=torch.long).to(trojan_edge.device)#torch.cat([edges_attack_trigger[1],edges_attack_trigger[0]])
            edge_index_inject = torch.stack([row,col])
            added_edges = torch.cat([trojan_edge,edge_index_inject],dim=1)#torch.cat([trojan_edge,edge_index_inject],dim=1) #这个的效果是最好的
            # added_edges = added_edges[:,0:trojan_edge.shape[1]+2]
        else:
            added_edges = trojan_edge#torch.cat([trojan_edge,trojan_edge],dim=1)[:,0:3]#
        row_adj = adj.storage.row()
        col_adj = adj.storage.col()
        value_adj = adj.storage.value()
        updated_adj_value_one = torch.ones(value_adj.shape[0]+added_edges.shape[1]).to(self.device)#######
        edges_adj = torch.stack([row_adj,col_adj])
        updated_adj_edges_indices = torch.cat([edges_adj,added_edges],dim=1).to(self.device)#########
        '''#####################
        不确定这么写对不对,需要测试:如上
        '''#####################

        new_size_adj = adj.sizes()[0]+self.trigger_size*idx_attach.shape[0]

        if self.args.dataset in ['flickr', 'reddit', 'ogbn-arxiv']:
            dense_adj = torch.sparse.FloatTensor(updated_adj_edges_indices,updated_adj_value_one,torch.Size((new_size_adj,new_size_adj)))#.to_dense()
            dense_adj_norm = utils.normalize_adj_tensor(dense_adj, sparse=True)##############Here we can convert sparse adj to normalized_adj
        else:
            dense_adj = torch.sparse.FloatTensor(updated_adj_edges_indices,updated_adj_value_one,torch.Size((new_size_adj,new_size_adj))).to_dense()
            if utils.is_sparse_tensor(dense_adj):
                dense_adj_norm = utils.normalize_adj_tensor(dense_adj, sparse=True)##############Here we can convert sparse adj to normalized_adj
            else:
                dense_adj_norm = utils.normalize_adj_tensor(dense_adj)

        updated_adj = dense_adj_norm

        return updated_feat, updated_adj, updated_labels

    def get_poisoned_induct_test(self, feat_test, adj_test, idx_attach):#,edge_index,edge_weight):
        '''
        return normalized dense adj matrix and the features, which are all poisoned
        '''
        features, adj = utils.to_tensor(feat_test, adj_test, device=self.device)
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)##############Here we can convert sparse adj to normalized_adj
        else:
            adj_norm = utils.normalize_adj_tensor(adj)
        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()
        
        self.trojan.eval()
        trojan_edge = self.get_trojan_edge_induct(len(features),idx_attach).to(self.device)#those edges are among the trigger nodes, the edges here could be converted into adj, (node_h, node_t)
        # import pdb;pdb.set_trace()
        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.threshold)

        trojan_feat = trojan_feat.view([-1,features.shape[1]])
        updated_feat = torch.cat([features,trojan_feat])
        trojan_labels = torch.ones(trojan_feat.shape[0], dtype=torch.long)*self.target_class

        updated_labels = torch.cat([self.labels.clone(),trojan_labels.to(self.labels.device)])

        indices_trojan_edges = trojan_weights>=0
        # indices_trojan_edges[0][0] = True
        edges_attack_trigger = [[],[]]
        if (indices_trojan_edges).nonzero().shape[0] != 0:
            indices = (indices_trojan_edges).nonzero()
            # for j in range(indices.shape[0]):
            #     edges_attack_trigger[0].append(idx_attach[indices[j][0]])
            #     edges_attack_trigger[1].append(len(features)+indices[j][0]*self.trigger_size+indices[j][1])
            for i in range(idx_attach.shape[0]):
                for j in range(self.trigger_size):
                    #如果需要用的edge_weight的话，在这里加个概率判断就可以了
                    #new type
                    if j !=0:
                        edges_attack_trigger[0].append(idx_attach[i])
                        edges_attack_trigger[1].append(len(features)+i*self.trigger_size+j)

            row = torch.tensor(edges_attack_trigger[0], dtype=torch.long).to(trojan_edge.device)#torch.cat([edges_attack_trigger[0], edges_attack_trigger[1]])
            col = torch.tensor(edges_attack_trigger[1], dtype=torch.long).to(trojan_edge.device)#torch.cat([edges_attack_trigger[1],edges_attack_trigger[0]])
            edge_index_inject = torch.stack([row,col])
            added_edges = torch.cat([trojan_edge,edge_index_inject],dim=1) #
            '''#####################
            不确定这么写对不对,需要测试:如上
            '''#####################
        else:
            added_edges = trojan_edge

        row_adj = adj.storage.row()
        col_adj = adj.storage.col()
        value_adj = adj.storage.value()
        updated_adj_value_one = torch.ones(value_adj.shape[0]+added_edges.shape[1]).to(self.device)#######
        edges_adj = torch.stack([row_adj,col_adj])
        updated_adj_edges_indices = torch.cat([edges_adj,added_edges],dim=1).to(self.device)#########
        '''#####################
        不确定这么写对不对,需要测试:如上
        '''#####################

        new_size_adj = adj.sizes()[0]+self.trigger_size*idx_attach.shape[0]        
        if self.args.dataset in ['flickr', 'reddit', 'ogbn-arxiv']:
            dense_adj = torch.sparse.FloatTensor(updated_adj_edges_indices,updated_adj_value_one,torch.Size((new_size_adj,new_size_adj)))#.to_dense()
            dense_adj_norm = utils.normalize_adj_tensor(dense_adj, sparse=True)##############Here we can convert sparse adj to normalized_adj
        else:
            dense_adj = torch.sparse.FloatTensor(updated_adj_edges_indices,updated_adj_value_one,torch.Size((new_size_adj,new_size_adj))).to_dense()
            if utils.is_sparse_tensor(dense_adj):
                dense_adj_norm = utils.normalize_adj_tensor(dense_adj, sparse=True)##############Here we can convert sparse adj to normalized_adj
            else:
                dense_adj_norm = utils.normalize_adj_tensor(dense_adj)

        updated_adj = dense_adj_norm

        return updated_feat, updated_adj, updated_labels

    def get_poisoned_syn(self,adj_syn, features_syn, idx_attach):#,edge_index,edge_weight):
        '''
        return normalized dense adj matrix and the features, which are all poisoned
        '''
        
        adj = adj_syn.clone()#self.original_adj.clone()#adj_input.clone()
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)##############Here we can convert sparse adj to normalized_adj
        else:
            adj_norm = utils.normalize_adj_tensor(adj)
            adj_norm = adj_norm.to_sparse()
        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()

        # idx_attach = idx_outter#self.idx_attach
        features = features_syn#self.original_features
        self.trojan.eval()

        # import pdb;pdb.set_trace()
        trojan_edge = self.get_trojan_edge(len(features),idx_attach).to(self.device)#those edges are among the trigger nodes, the edges here could be converted into adj, (node_h, node_t)
        # try:
        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.threshold)
        # except RuntimeError as e:
        #     import pdb;pdb.set_trace()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])
        updated_feat = torch.cat([features,trojan_feat])
        trojan_labels = torch.ones(trojan_feat.shape[0], dtype=torch.long)*self.target_class

        updated_labels = torch.cat([self.labels.clone(),trojan_labels.to(self.labels.device)])

        indices_trojan_edges = trojan_weights>=0
        # indices_trojan_edges[0][0] = True
        edges_attack_trigger = [[],[]]
        if (indices_trojan_edges).nonzero().shape[0] != 0:
            indices = (indices_trojan_edges).nonzero()
            
            for i in range(idx_attach.shape[0]):
                for j in range(self.trigger_size):
                    if j !=0:
                        edges_attack_trigger[0].append(idx_attach[i])
                        edges_attack_trigger[1].append(len(features)+i*self.trigger_size+j)
            
            row = torch.tensor(edges_attack_trigger[0], dtype=torch.long).to(trojan_edge.device)#torch.cat([edges_attack_trigger[0], edges_attack_trigger[1]])
            col = torch.tensor(edges_attack_trigger[1], dtype=torch.long).to(trojan_edge.device)#torch.cat([edges_attack_trigger[1],edges_attack_trigger[0]])
            edge_index_inject = torch.stack([row,col])
            added_edges = torch.cat([trojan_edge,edge_index_inject],dim=1)#torch.cat([trojan_edge,edge_index_inject],dim=1) #这个的效果是最好的
            # added_edges = added_edges[:,0:trojan_edge.shape[1]+2]
        else:
            added_edges = trojan_edge#torch.cat([trojan_edge,trojan_edge],dim=1)[:,0:3]#
        row_adj = adj.storage.row()
        col_adj = adj.storage.col()
        value_adj = adj.storage.value()
        updated_adj_value_one = torch.ones(value_adj.shape[0]+added_edges.shape[1]).to(self.device)#######
        edges_adj = torch.stack([row_adj,col_adj])
        updated_adj_edges_indices = torch.cat([edges_adj,added_edges],dim=1).to(self.device)#########
        '''#####################
        不确定这么写对不对,需要测试:如上
        '''#####################

        new_size_adj = adj.sizes()[0]+self.trigger_size*idx_attach.shape[0]

        if self.args.dataset in ['flickr', 'reddit', 'ogbn-arxiv']:
            dense_adj = torch.sparse.FloatTensor(updated_adj_edges_indices,updated_adj_value_one,torch.Size((new_size_adj,new_size_adj)))#.to_dense()
            dense_adj_norm = utils.normalize_adj_tensor(dense_adj, sparse=True)##############Here we can convert sparse adj to normalized_adj
        else:
            # import pdb;pdb.set_trace()
            dense_adj = torch.sparse.FloatTensor(updated_adj_edges_indices,updated_adj_value_one,torch.Size((new_size_adj,new_size_adj))).to_dense()
            if utils.is_sparse_tensor(dense_adj):
                dense_adj_norm = utils.normalize_adj_tensor(dense_adj, sparse=True)##############Here we can convert sparse adj to normalized_adj
            else:
                dense_adj_norm = utils.normalize_adj_tensor(dense_adj)

        updated_adj = dense_adj_norm

        return updated_feat, updated_adj, updated_labels


    def fit_induct(self, gnn_model, idx_train, idx_unlabeled, outer_loop_graph_cond, total_outer_loop_graph_cond, iter, total_iters, last_multi=False):#, acc_train_clean, acc_train_attach):#, idx_unlabeled):edge_index, edge_weight, 

        idx_train = np.arange(len(idx_train))
        idx_attach = self.idx_attach
        features = self.original_features
        adj = self.original_adj.clone()

        loss_best = 1e8
        if last_multi:
            outer_epochs = 100
        else:
            outer_epochs = self.outer_epochs
        self.trojan.train()
        
        for i in range(outer_epochs):#tqdm(range(outer_epochs), desc=f'{outer_loop_graph_cond}-th/({total_outer_loop_graph_cond}) Outer loop, {iter}-th/({total_iters})'):
            self.optimizer_trigger.zero_grad()

            rs = np.random.RandomState(self.seed)
            
            idx_outter = idx_train#np.concatenate([idx_train,idx_attach,idx_unlabeled])

            # updated_feat, updated_adj, updated_labels = self.get_poisoned(idx_outter)#有一说一，我觉得给他这么写的话就得考虑给idx_train全部label成target class然后去update triggers了
            updated_feat, updated_adj, updated_labels = self.get_poisoned(idx_attach)

            output = gnn_model(updated_feat,updated_adj)#, update_edge_index)#, update_edge_weights)

            labels_outter = self.labels.clone()
            labels_outter[idx_attach] = self.target_class# the original work version is this one with the idx_outter
            # labels_outter[idx_outter] = self.target_class
            loss_target = self.target_loss_weight * F.nll_loss(output[idx_outter],#torch.cat([idx_train,idx_outter])],
                                    labels_outter[idx_outter])#你可以仔细看看UGBA的loss_target的idx_outter 都是怎么选的然后你也改一改
            loss_outter = loss_target
            loss_outter.backward()
            self.optimizer_trigger.step()
            acc_train_outter =(output[idx_attach].argmax(dim=1)==self.target_class).float().mean()

            if loss_outter<loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter)

        if outer_epochs ==0:
            loss_target = -1
            acc_train_outter = -1
            updated_labels = []
        else:
            self.trojan.load_state_dict(self.weights)
        print('Epoch {}/{}, loss_backdoor_target: {:.5f}, ASR_train_outter_target: {:.4f}'\
                        .format(iter, total_iters, loss_target, acc_train_outter))
        self.trojan.eval()
        return updated_labels

    def fit(self, gnn_model, idx_train, idx_unlabeled, outer_loop_graph_cond, total_outer_loop_graph_cond, iter, total_iters, last_multi=False):#, acc_train_clean, acc_train_attach):#, idx_unlabeled):edge_index, edge_weight, 

        # self.idx_attach = idx_attach
        idx_attach = self.idx_attach
        features = self.original_features
        adj = self.original_adj.clone()

        # labels = self.labels.clone()
        # self.labels[idx_attach] = self.target_class

        loss_best = 1e8
        if last_multi:
            outer_epochs = 100
        else:
            outer_epochs = self.outer_epochs
        self.trojan.train()
        # self.trojan.eval()
        # print('Condensation Epoch: {}-th/{}'.format(iter,total_iters))
        for i in range(outer_epochs):#tqdm(range(outer_epochs), desc=f'{outer_loop_graph_cond}-th/({total_outer_loop_graph_cond}) Outer loop, {iter}-th/({total_iters})'):
            self.optimizer_trigger.zero_grad()

            rs = np.random.RandomState(self.seed)
            # idx_outter = idx_attach#torch.cat([idx_attach,idx_unlabeled[rs.choice(len(idx_unlabeled),size=512,replace=False)]])
            #idx_outter可以参考一下这里改改，https://github.com/ventr1c/UGBA/blob/main/models/backdoor.py#L168
            # idx_outter = np.concatenate([idx_train,idx_attach,idx_unlabeled])
            idx_outter = np.concatenate([idx_train,idx_attach])

            # updated_feat, updated_adj, updated_labels = self.get_poisoned(np.concatenate([idx_attach,idx_unlabeled]))#这个目前在cora上的效果也不错
            updated_feat, updated_adj, updated_labels = self.get_poisoned(idx_attach)

            output = gnn_model(updated_feat,updated_adj)#, update_edge_index)#, update_edge_weights)

            labels_outter = self.labels.clone()
            labels_outter[idx_attach] = self.target_class
            # labels_outter[idx_unlabeled] = self.target_class
            # loss_target = self.target_loss_weight * F.nll_loss(output[idx_train],#torch.cat([idx_train,idx_outter])],
            #                         labels_outter[idx_train])#torch.cat([idx_train,idx_outter])]) #感觉有没有可能是这个loss的问题，并不是用所有的output来update generator的而是用idx_outter来的
            loss_target = self.target_loss_weight * F.nll_loss(output[idx_outter],#torch.cat([idx_train,idx_outter])],
                                    labels_outter[idx_outter])#你可以仔细看看UGBA的loss_target的idx_outter 都是怎么选的然后你也改一改
            loss_outter = loss_target
            loss_outter.backward()
            self.optimizer_trigger.step()
            acc_train_outter =(output[idx_attach].argmax(dim=1)==self.target_class).float().mean()

            if loss_outter<loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter)
            
            #'''
            # if i % 10 == 0:
            #     print('Backdoor epoch {}/{}, loss_backdoor_target: {:.5f}, ASR_train_outter_target: {:.4f}'\
            #             .format(iter, outer_epochs, loss_target, acc_train_outter))
                # print("acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, ASR_train_outter: {:.4f}"\
                #         .format(acc_train_clean,acc_train_attach,acc_train_outter))
                # print("".format())
            #'''
        # print('Epoch {}/{}, dd epoch {}/{} loss_backdoor_target: {:.5f}, ASR_train_outter_target: {:.4f}'\
        #             .format(iter, total_iters, outer_loop_graph_cond, total_outer_loop_graph_cond, loss_target, acc_train_outter))
        if outer_epochs ==0:
            loss_target = -1
            acc_train_outter = -1
            updated_labels = []
        else:
            self.trojan.load_state_dict(self.weights)
        print('Epoch {}/{}, loss_backdoor_target: {:.5f}, ASR_train_outter_target: {:.4f}'\
                        .format(iter, total_iters, loss_target, acc_train_outter))
        self.trojan.eval()
        return updated_labels

    def fit_naive_attack(self, args, idx_train, idx_unlabeled, backdoor_epochs, inner_loop, nclass, last_multi=False):#, acc_train_clean, acc_train_attach):#, idx_unlabeled):edge_index, edge_weight, 

        idx_attach = self.idx_attach
        features = self.original_features
        adj = self.original_adj.clone()

        loss_best = 1e8
        if last_multi:
            outer_epochs = 100
        else:
            outer_epochs = backdoor_epochs
        self.trojan.train()
        if args.dataset in ['ogbn-arxiv']:
                gnn_model = SGC1(nfeat=features.shape[1], nhid=args.hidden,
                            dropout=0.0, with_bn=False,
                            weight_decay=0e-4, nlayers=2,
                            nclass=nclass,
                            device=self.device).to(self.device)
        else:
            if args.sgc == 1:
                gnn_model = SGC(nfeat=features.shape[1], nhid=args.hidden,
                            nclass=nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)
            else:
                gnn_model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=nclass, dropout=args.dropout, nlayers=args.nlayers,
                            device=self.device).to(self.device)
        gnn_model.initialize()
        gnn_model.train()
        model_parameters = list(gnn_model.parameters())
        optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
        idx_outter = np.concatenate([idx_train,idx_attach,idx_unlabeled])
        idx_labels_poison = np.concatenate([idx_train,idx_attach])
        for i in range(outer_epochs):#tqdm(range(outer_epochs), desc=f'{outer_loop_graph_cond}-th/({total_outer_loop_graph_cond}) Outer loop, {iter}-th/({total_iters})'):
            self.optimizer_trigger.zero_grad()

            rs = np.random.RandomState(self.seed)

            updated_feat, updated_adj, updated_labels = self.get_poisoned(idx_outter)

            output = gnn_model(updated_feat,updated_adj)#, update_edge_index)#, update_edge_weights)

            labels_outter = self.labels.clone()
            labels_outter[idx_attach] = self.target_class
            labels_outter[idx_unlabeled] = self.target_class

            loss_target = self.target_loss_weight * F.nll_loss(output[idx_outter],#torch.cat([idx_train,idx_outter])],
                                    labels_outter[idx_outter])#你可以仔细看看UGBA的loss_target的idx_outter 都是怎么选的然后你也改一改
            loss_outter = loss_target
            loss_outter.backward()
            self.optimizer_trigger.step()
            acc_train_outter =(output[idx_attach].argmax(dim=1)==self.target_class).float().mean()

            if loss_outter<loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter)
            print('Epoch{}/{}, loss_target_attack:{:.5f}'.format(i,outer_epochs,loss_outter))
            for j in range(inner_loop):
                optimizer_model.zero_grad()
                output_inner = gnn_model.forward(updated_feat.detach(),updated_adj)
                loss_inner = F.nll_loss(output_inner[idx_labels_poison], labels_outter[idx_labels_poison])
                loss_inner.backward()
                optimizer_model.step() # update gnn param

        if outer_epochs ==0:
            loss_target = -1
            acc_train_outter = -1
            updated_labels = []
        else:
            self.trojan.load_state_dict(self.weights)
        print('Finish backdoor attack, ASR_train_outter_target: {:.4f}'\
                        .format(acc_train_outter))
        self.trojan.eval()
        return updated_labels

    def update_graph(self):
        return