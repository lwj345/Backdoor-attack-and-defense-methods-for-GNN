from random import random
import torch 
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from deeprobust.graph import utils
import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
import copy
from sklearn_extra import cluster
from sklearn.cluster import KMeans
from torch_geometric.utils import degree

def max_norm(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def randomly_obtain_attach_nodes(seed,node_idxs, size):
    ### current random to implement
    size = min(len(node_idxs),size)
    rs = np.random.RandomState(seed)
    choice = np.arange(len(node_idxs))
    rs.shuffle(choice)
    return node_idxs[choice[:size]]

def randomly_obtain_attach_nodes_induct(seed,node_idxs, size):
    size = min(len(node_idxs),size)
    rs = np.random.RandomState(seed)
    choice = np.arange(len(node_idxs))
    rs.shuffle(choice)
    return choice[:size]

def cluster_degree_selection(args,features,adj,labels,nclass,gcn_encoder,idx_train,idx_val,idx_clean_test,unlabeled_idx,size,device,epochs=600,verbose=True):
    # selected_nodes_path = "./selected_nodes/{}/Overall/seed{}/nodes.txt".format(args.dataset,args.seed)
    # if(os.path.exists(selected_nodes_path)):
    #     print(selected_nodes_path)
    #     idx_attach = np.loadtxt(selected_nodes_path, delimiter=',').astype(int)
    #     idx_attach = idx_attach[:size]
    #     return idx_attach
    # gcn_encoder = model_construct(args,'GCN_Encoder',data,device).to(device) 
    t_total = time.time()
    # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)

    print("===Training for node selection!!!===")
    print("Length of training set: {}".format(len(idx_train)))
    # gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)#这里得改改###############################################
    # import pdb;pdb.set_trace()
    # feat_full, adj_full, labels = utils.to_tensor(features.cpu(), adj.cpu(), labels.cpu(), device=gcn_encoder.cpu().device)
    feat_full, adj_full, labels = features, adj, labels
    adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
    # labels = torch.LongTensor(labels).to(gcn_encoder.device)
    optimizer = optim.Adam(gcn_encoder.parameters(), lr=gcn_encoder.lr, weight_decay=gcn_encoder.weight_decay)
    best_acc_val = 0
    gcn_encoder.features, gcn_encoder.adj_norm, gcn_encoder.labels = features, adj, labels
    for i in range(epochs):
        if i == epochs // 2:
            lr = gcn_encoder.lr*0.1
            optimizer = optim.Adam(gcn_encoder.parameters(), lr=lr, weight_decay=gcn_encoder.weight_decay)
        gcn_encoder.train()
        optimizer.zero_grad()
        output = gcn_encoder.forward(features, adj_full_norm)
        # import pdb;pdb.set_trace()
        try:
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()
        loss_train.backward()
        optimizer.step()
        if verbose and i % 100 == 0:
            print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
        with torch.no_grad():
            gcn_encoder.eval()
            output = gcn_encoder.forward(feat_full, adj_full_norm)#################这里也得改改 用 forward_sampler for inductive settings
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                gcn_encoder.output = output
                weights = copy.deepcopy(gcn_encoder.state_dict())
    print('=== picking the best model according to the performance on validation ===')
    gcn_encoder.load_state_dict(weights)
    print("Training encoder Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    encoder_clean_test_ca = gcn_encoder.test(idx_clean_test)
    print("Encoder clean accuracy: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = unlabeled_idx#torch.concat([idx_train,unlabeled_idx])#这里得改改###############################################
    # nclass = args.nclass
    encoder_x = gcn_encoder.forward_x(feat_full,adj_full_norm).clone().detach()#这里得改改###############################################
    if(args.dataset == 'Cora' or args.dataset == 'Citeseer'):
        kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')#这里得弄清楚这两的区别是啥###############################################
        kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmedoids.cluster_centers_
        y_pred = kmedoids.predict(encoder_x.cpu().numpy())
    else:
        kmeans = KMeans(n_clusters=nclass,random_state=1)#这里得弄清楚这两的区别是啥###############################################
        kmeans.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        y_pred = kmeans.predict(encoder_x.cpu().numpy())

    # encoder_output = gcn_encoder(data.x,train_edge_index,None)
    # import pdb;pdb.set_trace()
    edge_index = adj_full.coalesce().indices()
    idx_attach = obtain_attach_nodes_by_cluster_degree_all(args,edge_index, y_pred,cluster_centers,unlabeled_idx.tolist(),encoder_x,size).astype(int)#这里得改改###############################################
    selected_nodes_foldpath = "./selected_nodes/{}/Overall/seed{}".format(args.dataset,args.seed)
    if(not os.path.exists(selected_nodes_foldpath)):
        os.makedirs(selected_nodes_foldpath)
    selected_nodes_path = "./selected_nodes/{}/Overall/seed{}/nodes.txt".format(args.dataset,args.seed)
    if(not os.path.exists(selected_nodes_path)):
        np.savetxt(selected_nodes_path,idx_attach)
    else:
        idx_attach = np.loadtxt(selected_nodes_path, delimiter=',').astype(int)
    idx_attach = idx_attach[:size]
    return idx_attach


def cluster_degree_selection_induct(args,features,adj,labels,nclass,gcn_encoder,feat_val, adj_val, labels_val, size,device,epochs=600,verbose=True):
    selected_nodes_path = "./selected_nodes/{}/Overall/seed{}/nodes.txt".format(args.dataset,args.seed)
    if(os.path.exists(selected_nodes_path)):
        print(selected_nodes_path)
        idx_attach = np.loadtxt(selected_nodes_path, delimiter=',').astype(int)
        idx_attach = idx_attach[:size]
        return idx_attach
    t_total = time.time()

    print("===Training for node selection!!!===")
    print("Length of training set: {}".format(adj.shape[0]))

    feat_full, adj_full, labels = features, adj, labels
    adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
    adj_val_norm = utils.normalize_adj_tensor(adj_val, sparse=True)

    optimizer = optim.Adam(gcn_encoder.parameters(), lr=gcn_encoder.lr, weight_decay=gcn_encoder.weight_decay)
    best_acc_val = 0
    gcn_encoder.features, gcn_encoder.adj_norm, gcn_encoder.labels = features, adj, labels
    for i in range(epochs):
        if i == epochs // 2:
            lr = gcn_encoder.lr*0.1
            optimizer = optim.Adam(gcn_encoder.parameters(), lr=lr, weight_decay=gcn_encoder.weight_decay)
        gcn_encoder.train()
        optimizer.zero_grad()
        output = gcn_encoder.forward(features, adj_full_norm)
        # import pdb;pdb.set_trace()
        try:
            loss_train = F.nll_loss(output, labels)#[idx_train], labels[idx_train])
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()
        loss_train.backward()
        optimizer.step()
        if verbose and i % 100 == 0:
            print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
        with torch.no_grad():
            gcn_encoder.eval()
            output = gcn_encoder.forward(feat_val, adj_val_norm)#################这里也得改改 用 forward_sampler for inductive settings
            loss_val = F.nll_loss(output, labels_val)
            acc_val = utils.accuracy(output, labels_val)
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                gcn_encoder.output = output
                weights = copy.deepcopy(gcn_encoder.state_dict())
    print('=== picking the best model according to the performance on validation ===')
    gcn_encoder.load_state_dict(weights)
    print("Training encoder Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # encoder_clean_test_ca = gcn_encoder.test(idx_clean_test)
    # print("Encoder clean accuracy: {:.4f}".format(encoder_clean_test_ca))

    encoder_x = gcn_encoder.forward_x(feat_full,adj_full_norm).clone().detach()#这里得改改###############################################
    if(args.dataset == 'Cora' or args.dataset == 'Citeseer'):
        kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')#这里得弄清楚这两的区别是啥###############################################
        kmedoids.fit(encoder_x.detach().cpu().numpy())
        cluster_centers = kmedoids.cluster_centers_
        y_pred = kmedoids.predict(encoder_x.cpu().numpy())
    else:
        import threadpoolctl
        threadpoolctl.threadpool_limits(16)
        kmeans = KMeans(n_clusters=nclass,random_state=1,n_init=10)#这里得弄清楚这两的区别是啥###############################################
        # import pdb;pdb.set_trace()
        kmeans.fit(encoder_x.detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        y_pred = kmeans.predict(encoder_x.cpu().numpy())

    edge_index = adj_full.coalesce().indices()
    idx_attach = obtain_attach_nodes_by_cluster_degree_all_induct(args,edge_index, y_pred,cluster_centers,encoder_x,size).astype(int)#这里得改改###############################################
    selected_nodes_foldpath = "./selected_nodes/{}/Overall/seed{}".format(args.dataset,args.seed)
    if(not os.path.exists(selected_nodes_foldpath)):
        os.makedirs(selected_nodes_foldpath)
    selected_nodes_path = "./selected_nodes/{}/Overall/seed{}/nodes.txt".format(args.dataset,args.seed)
    if(not os.path.exists(selected_nodes_path)):
        np.savetxt(selected_nodes_path,idx_attach)
    else:
        idx_attach = np.loadtxt(selected_nodes_path, delimiter=',').astype(int)
    idx_attach = idx_attach[:size]
    return idx_attach


def obtain_attach_nodes_by_cluster_degree_all(args,edge_index,y_pred,cluster_centers,node_idxs,x,size):
    dis_weight = args.dis_weight
    degrees = (degree(edge_index[0])  + degree(edge_index[1])).cpu().numpy()#this is a tensor containing the degree of each node, its shape is (num_of_nodes,)
    distances = []
    for id in range(x.shape[0]):
        tmp_center_label = y_pred[id]
        tmp_center_x = cluster_centers[tmp_center_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        distances.append(dis)

    distances = np.array(distances)
    print(y_pred)
    nontarget_nodes = np.where(y_pred!=args.target_class)[0]

    non_target_node_idxs = np.array(list(set(nontarget_nodes) & set(node_idxs)))
    node_idxs = np.array(non_target_node_idxs)
    candiadate_distances = distances[node_idxs]
    candiadate_degrees = degrees[node_idxs]
    candiadate_distances = max_norm(candiadate_distances)
    candiadate_degrees = max_norm(candiadate_degrees)

    dis_score = candiadate_distances + dis_weight * candiadate_degrees
    candidate_nid_index = np.argsort(dis_score)
    sorted_node_idex = np.array(node_idxs[candidate_nid_index])
    selected_nodes = sorted_node_idex
    return selected_nodes
    # each_selected_num = int(size/len(label_list)-1)
    # last_seleced_num = size - each_selected_num*(len(label_list)-2)
    # candidate_nodes = np.array([])

    # for label in label_list:
    #     if(label == args.target_class):
    #         continue
    #     single_labels_nodes = labels_dict[label]    # the node idx of the nodes in single class
    #     single_labels_nodes = np.array(list(set(single_labels_nodes)))

    #     single_labels_nodes_dis = distances[single_labels_nodes]
    #     single_labels_nodes_dis = max_norm(single_labels_nodes_dis)

    #     single_labels_nodes_degrees = degrees[single_labels_nodes]
    #     single_labels_nodes_degrees = max_norm(single_labels_nodes_degrees)
        
    #     # the closer to the center, the more far away from the target centers
    #     # single_labels_dis_score =  single_labels_nodes_dis + dis_weight * (-single_labels_nodes_dis_tar)
    #     single_labels_dis_score = single_labels_nodes_dis + dis_weight * single_labels_nodes_degrees
    #     single_labels_nid_index = np.argsort(single_labels_dis_score) # sort descently based on the distance away from the center
    #     sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
    #     if(label != label_list[-1]):
    #         candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:each_selected_num]])
    #     else:
    #         candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:last_seleced_num]])
    # return candidate_nodes

def obtain_attach_nodes_by_cluster_degree_all_induct(args,edge_index,y_pred,cluster_centers,x,size):
    dis_weight = args.dis_weight
    degrees = (degree(edge_index[0])  + degree(edge_index[1])).cpu().numpy()#this is a tensor containing the degree of each node, its shape is (num_of_nodes,)
    distances = []
    for id in range(x.shape[0]):
        tmp_center_label = y_pred[id]
        tmp_center_x = cluster_centers[tmp_center_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        distances.append(dis)

    distances = np.array(distances)
    print(y_pred)
    nontarget_nodes = np.where(y_pred!=args.target_class)[0]
    # import pdb;pdb.set_trace()
    # non_target_node_idxs = np.array(list(set(nontarget_nodes) & set(node_idxs)))
    node_idxs = np.array(nontarget_nodes)#(non_target_node_idxs)
    candiadate_distances = distances[node_idxs]
    candiadate_degrees = degrees[node_idxs]
    candiadate_distances = max_norm(candiadate_distances)
    candiadate_degrees = max_norm(candiadate_degrees)

    dis_score = candiadate_distances + dis_weight * candiadate_degrees
    candidate_nid_index = np.argsort(dis_score)
    sorted_node_idex = np.array(node_idxs[candidate_nid_index])
    selected_nodes = sorted_node_idex
    return selected_nodes
    # each_selected_num = int(size/len(label_list)-1)
    # last_seleced_num = size - each_selected_num*(len(label_list)-2)
    # candidate_nodes = np.array([])

    # for label in label_list:
    #     if(label == args.target_class):
    #         continue
    #     single_labels_nodes = labels_dict[label]    # the node idx of the nodes in single class
    #     single_labels_nodes = np.array(list(set(single_labels_nodes)))

    #     single_labels_nodes_dis = distances[single_labels_nodes]
    #     single_labels_nodes_dis = max_norm(single_labels_nodes_dis)

    #     single_labels_nodes_degrees = degrees[single_labels_nodes]
    #     single_labels_nodes_degrees = max_norm(single_labels_nodes_degrees)
        
    #     # the closer to the center, the more far away from the target centers
    #     # single_labels_dis_score =  single_labels_nodes_dis + dis_weight * (-single_labels_nodes_dis_tar)
    #     single_labels_dis_score = single_labels_nodes_dis + dis_weight * single_labels_nodes_degrees
    #     single_labels_nid_index = np.argsort(single_labels_dis_score) # sort descently based on the distance away from the center
    #     sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
    #     if(label != label_list[-1]):
    #         candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:each_selected_num]])
    #     else:
    #         candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:last_seleced_num]])
    # return candidate_nodes