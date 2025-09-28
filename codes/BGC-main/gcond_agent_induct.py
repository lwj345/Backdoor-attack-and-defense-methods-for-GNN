import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
import scipy.sparse as sp
from torch_sparse import SparseTensor
import utils_heuristic_select as hs
from utils_backdoor import BackdoorGC

class GCond:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        self.backdoor = args.backdoor

        n = int(len(data.idx_train) * args.reduction_rate)
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)

        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)
        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print('adj_syn:', (n,n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data):
        from collections import Counter
        counter = Counter(data.labels_train)
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def test_with_val(self, poisoned_features=None, poisoned_adj=None, poisoned_labels=None, trigger_generator=None, it=0, verbose=True):
        res = []

        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
                                self.pge, self.labels_syn
        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        dropout = 0.5 if self.args.dataset in ['reddit'] else 0
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=dropout,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        adj_syn = pge.inference(feat_syn)
        args = self.args

        if args.save:
            torch.save(adj_syn, f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}-{it}.pt')
            torch.save(feat_syn, f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}-{it}.pt')

        noval = True
        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=600, normalize=True, verbose=False, noval=noval, defense_type=args.defense_type, prune_rate=args.prune_rate)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        output = model.predict(data.feat_test, data.adj_test)

        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)
        res.append(acc_test.item())
        if verbose:
            print("Clean Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        # print(adj_syn.sum(), adj_syn.sum()/(adj_syn.shape[0]**2))

        if self.backdoor:
            '''###########################
            you need to carefully check the poisoned_feat and poisoned_adj to see whether it only contains the training adj and features
            and resample the idx_test_attack from test set.
            '''###########################

            idx_test = data.idx_test
            idx_test_attack = hs.randomly_obtain_attach_nodes_induct(self.args.seed, idx_test, int(1.0*idx_test.shape[0]))#int(self.args.attack_rate_test*idx_test.shape[0]))

            poisoned_feat, poisoned_adj, poisoned_labels = trigger_generator.get_poisoned_induct_test(data.feat_test, data.adj_test, idx_test_attack)
            #你这里的trigger_generator可能是根据初始化的时候的train_adj来算的，所以应该是错的，得重新写一个

            if args.defense_type == 'prune':
                # import pdb;pdb.set_trace()
                indices = poisoned_adj.coalesce().indices()
                device = indices.device
                # feat_cosine = torch.mm(poisoned_feat,poisoned_feat.T)
                self_ = [i for i in range(indices.shape[1]) if indices[0][i]==indices[1][i]]
                filtered_ = [i for i in range(indices.shape[1]) if indices[0][i]!=indices[1][i]]
                self_indices = indices[:,self_]########
                filtered_indices = indices[:,filtered_]########
                filtered_features0 = poisoned_feat[filtered_indices[0]]
                filtered_features1 = poisoned_feat[filtered_indices[1]]
                multiply = filtered_features0*filtered_features1
                similarities = multiply.sum(axis=1)########
                sorted_sim = torch.sort(similarities,descending=False)
                sorted_sim_values = sorted_sim.values
                split_value_index = int(sorted_sim.values.shape[0]*args.prune_rate)
                split_value = sorted_sim_values[split_value_index]########
                mask = similarities.ge(split_value)
                filtered_indices = filtered_indices[:,mask]
                pruned_indices = torch.concat((self_indices,filtered_indices),dim=1)
                #接下来就是如何通过mask获得对应的index
                size_adj = poisoned_adj.shape
                values = torch.ones_like(pruned_indices[1])
                poisoned_adj = torch.sparse_coo_tensor(pruned_indices,values,size=size_adj,device=device)#.to(device)
                poisoned_adj = utils.normalize_adj_tensor(poisoned_adj, sparse=True)
                torch.cuda.empty_cache()

                # import pdb;pdb.set_trace()

            if args.dataset in ['flickr', 'reddit', 'ogbn-arxiv']:
                sp_adj = poisoned_adj.coalesce()
            else:
                sp_adj = poisoned_adj.to_sparse()
            row=sp_adj._indices()[0].cpu().numpy()
            col=sp_adj._indices()[1].cpu().numpy()
            data_values=sp_adj._values().cpu().numpy()
            shape=sp_adj.size()

            poisoned_adj=sp.csr_matrix((data_values, (row, col)), shape=[shape[0],shape[1]])

            if args.defense_type == 'rand_smooth':
                output = model.predict(poisoned_feat.cpu(), poisoned_adj, args.defense_type, args.prune_rate)
            else:
                output = model.predict(poisoned_feat.cpu(), poisoned_adj)
            # output = model.predict(poisoned_feat.cpu(), poisoned_adj)
            #labels[idx_attach] = args.target_class
            #labels_test = torch.LongTensor(data.labels_test).cuda()
            labels_target_test = torch.ones(idx_test_attack.shape[0])*self.args.target_class
            labels_target_test = labels_target_test.long().to(self.device)
            poisoned_loss_test = F.nll_loss(output[idx_test_attack], labels_target_test)
            poisoned_acc_test = (output.argmax(dim=1)[idx_test_attack]==args.target_class).float().mean()#utils.accuracy(output[idx_test_attack], labels_target_test)
            # import pdb;pdb.set_trace()

            '''#####################
            这里需要修改一下
            '''#####################
            if verbose:
                print("Poisoned Test set results (ASR):",
                    "loss= {:.4f}".format(poisoned_loss_test.item()),
                    "accuracy= {:.4f}".format(poisoned_acc_test.item()))########################### here you also need to re-calculate the ASR, by changing the labels_test.
            res.append(poisoned_acc_test.item())

        if False:
            if self.args.dataset == 'ogbn-arxiv':
                thresh = 0.6
            elif self.args.dataset == 'reddit':
                thresh = 0.91
            else:
                thresh = 0.7

            labels_train = torch.LongTensor(data.labels_train).cuda()
            output = model.predict(data.feat_train, data.adj_train)
            # loss_train = F.nll_loss(output, labels_train)
            # acc_train = utils.accuracy(output, labels_train)
            loss_train = torch.tensor(0)
            acc_train = torch.tensor(0)
            if verbose:
                print("Train set results:",
                      "loss= {:.4f}".format(loss_train.item()),
                      "accuracy= {:.4f}".format(acc_train.item()))
            res.append(acc_train.item())
            
        return res

    def train(self, verbose=True):
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        features, adj, labels = data.feat_train, data.adj_train, data.labels_train

        syn_class_indices = self.syn_class_indices
        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        features_val, adj_val, labels_val = utils.to_tensor(data.feat_val,data.adj_val, data.labels_val, device=self.device)

        if args.backdoor:
            #你得确定target node是否需要从train-node里面剥离出来from UBGA：需要的,但是在我们这里不需要吧，而且要把trigger inject到 condensed data 里面去
            #你得看trigger有没有被distill到conensed data里，还是说生成condensed data的时候不需要trigger: 需要的
            if args.dataset in ['flickr']:
                attach_num = args.attach_num#80 #5
            elif args.dataset in ['ogbn-arxiv']:
                attach_num = int(0.3*idx_train.shape[0])#160
            else:#'reddit'
                attach_num = args.attach_num#180
            
            # idx_attach = hs.randomly_obtain_attach_nodes_induct(args.seed,data.idx_train,attach_num)#args.attach_rate*idx_train.shape[0]))
            if args.selector is 'random':
                idx_attach = hs.randomly_obtain_attach_nodes_induct(args.seed,data.idx_train,attach_num)#args.attach_rate*idx_train.shape[0]))
            else:
                model_selector = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                        weight_decay=5e-4, nlayers=2,
                        nclass=data.nclass, device=self.device).to(self.device)

                if self.args.dataset in ['ogbn-arxiv']:
                    model_selector = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                                weight_decay=5e-4, nlayers=2, with_bn=False,
                                nclass=data.nclass, device=self.device).to(self.device)
                idx_attach = hs.cluster_degree_selection_induct(args, features, adj, labels, data.nclass, model_selector, features_val, adj_val, labels_val, attach_num, self.device)

            
            triggers_num = args.trigger_size * idx_attach.shape[0]
            # idx_triggers = list(range(triggers_num)) + features.shape[0]
            #我们需要看一下是不是必须把那部分的labels改成target label: 是的，在NDSS里面是的
            labels[idx_attach] = args.target_class
            print("idx_attach: {}".format(idx_attach))
            trigger_generator = BackdoorGC(args,self.device,idx_attach,features,adj,labels)

        feat_sub, adj_sub = self.get_sub_adj_feat(features)
        self.feat_syn.data.copy_(feat_sub)

        outer_loop, inner_loop = get_loops(args)

        original_adj = adj.clone()

        for it in range(args.epochs+1):
            loss_avg = 0
            if args.sgc==1:
                model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)
            elif args.sgc==2:
                model = SGC1(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)
            else:
                model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                            device=self.device).to(self.device)

            model.initialize()

            model_parameters = list(model.parameters())

            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            for ol in range(outer_loop):

                if args.backdoor:
                    poisoned_feat, poisoned_adj, poisoned_labels = trigger_generator.get_poisoned(idx_attach)
                    features = poisoned_feat
                    # adj = poisoned_adj
                    if args.dataset in ['flickr', 'reddit', 'ogbn-arxiv']:
                        sp_adj = poisoned_adj.coalesce()
                        adj = SparseTensor(row=sp_adj.indices()[0], col=sp_adj.indices()[1],
                                value=sp_adj.values(), sparse_sizes=sp_adj.shape).t()
                    else:
                        sp_adj = poisoned_adj.to_sparse()
                        adj = SparseTensor(row=sp_adj.indices()[0], col=sp_adj.indices()[1],
                                value=sp_adj.values(), sparse_sizes=sp_adj.shape).t()
                else:
                    if utils.is_sparse_tensor(original_adj):
                        adj_norm = utils.normalize_adj_tensor(original_adj, sparse=True)##############Here we can convert sparse adj to normalized_adj
                    else:
                        adj_norm = utils.normalize_adj_tensor(original_adj)
                    adj = adj_norm
                    adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                            value=adj._values(), sparse_sizes=adj.size()).t()

                adj_syn = pge(self.feat_syn)
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                feat_syn_norm = feat_syn

                BN_flag = False
                for module in model.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    model.train() # for updating the mu, sigma of BatchNorm
                    output_real = model.forward(features, adj_norm)
                    for module in model.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                loss = torch.tensor(0.0).to(self.device)
                for c in range(data.nclass):
                    if c not in self.num_class_dict:
                        continue

                    batch_size, n_id, adjs = data.retrieve_class_sampler_backdoor(
                            c, adj, labels, data.idx_train, transductive=False, args=args)

                    if args.nlayers == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    output = model.forward_sampler(features[n_id], adjs)
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    ind = syn_class_indices[c]
                    if args.nlayers == 1:
                        adj_syn_norm_list = [adj_syn_norm[ind[0]: ind[1]]]
                    else:
                        adj_syn_norm_list = [adj_syn_norm]*(args.nlayers-1) + \
                                [adj_syn_norm[ind[0]: ind[1]]]

                    output_syn = model.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                    loss_syn = F.nll_loss(output_syn, labels_syn[ind[0]: ind[1]])

                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff  * match_loss(gw_syn, gw_real, args, device=self.device)

                loss_avg += loss.item()
                # TODO: regularize
                if args.alpha > 0:
                    loss_reg = args.alpha * regularization(adj_syn, utils.tensor2onehot(labels_syn))
                # else:
                else:
                    loss_reg = torch.tensor(0)

                loss = loss + loss_reg

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                if args.debug and ol % 5 ==0:
                    print('Gradient matching loss:', loss.item())

                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step() # update gnn param
            if args.backdoor:
            # if False:
                trigger_generator.fit_induct(model,data.idx_train,None, ol, outer_loop, it, args.epochs)
            # loss_avg /= (data.nclass*outer_loop)
            # if it % 50 == 0:
            #     print('Epoch {}, loss_avg: {}'.format(it, loss_avg))

            eval_epochs = [0,1,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000, 1200, 1600, 2000, 3000, 4000, 5000]
            # if it == args.epochs:
            #     torch.save(trigger_generator,f'saved_ours/trojan_networks/trigger-gen_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
            if verbose and it in eval_epochs:
            # if verbose and (it+1) % 500 == 0:
                torch.save(trigger_generator,f'saved_ours/trojan_networks/trigger-gen_{args.dataset}_{args.reduction_rate}_{args.seed}-{it}.pt')
                res = []
                runs = 3 #1 if args.dataset in ['ogbn-arxiv', 'reddit', 'flickr'] else 3
                for i in range(runs):
                    if args.backdoor:
                        # if args.dataset in ['ogbn-arxiv']:
                        #     res.append(self.test_with_val(poisoned_feat, poisoned_adj, poisoned_labels, trigger_generator, it))
                        # else:
                        res.append(self.test_with_val(poisoned_feat, poisoned_adj, poisoned_labels, trigger_generator, it))
                    else:
                        if args.dataset in ['ogbn-arxiv']:
                            res.append(self.test_with_val(None))
                        else:
                            res.append(self.test_with_val(None))
                res = np.array(res)
                print('Clean Test/ASR Mean Accuracy:',
                        repr([res.mean(0), res.std(0)]))



    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.labels_syn.cpu().numpy())

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[idx_selected]

        # adj_knn = torch.zeros((data.nclass*args.nsamples, data.nclass*args.nsamples)).to(self.device)
        # for i in range(data.nclass):
        #     idx = np.arange(i*args.nsamples, i*args.nsamples+args.nsamples)
        #     adj_knn[np.ix_(idx, idx)] = 1

        from sklearn.metrics.pairwise import cosine_similarity
        # features[features!=0] = 1
        k = 2
        sims = cosine_similarity(features.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        return 10, 0

    if args.dataset in ['ogbn-arxiv']:
        return 20, 0
    if args.dataset in ['reddit']:
        return args.outer, args.inner
    if args.dataset in ['flickr']:
        return args.outer, args.inner
        # return 10, 1
    if args.dataset in ['cora']:
        return 20, 10
    if args.dataset in ['citeseer']:
        return 20, 5 # at least 200 epochs
    else:
        return 20, 5

