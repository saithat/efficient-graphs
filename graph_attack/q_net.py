from __future__ import print_function

import os
import sys
import numpy as np
import torch
import networkx as nx
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.append('%s/../pytorch_structure2vec/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from graph_embedding import EmbedMeanField, EmbedLoopyBP
from cmd_args import cmd_args

from rl_common import local_args

def greedy_actions(q_values, banned_list, _type):
    
    actions = []
    offset = 0
    banned_acts = []
    
    q_values = q_values.data.clone()
        
    num_chunks = q_values.shape[0] // 20
        
    q_list = torch.chunk(q_values[:num_chunks * 20], num_chunks, dim=0)

    actions = []
    
    for i in range(len(q_list)):
        actions.append(np.argmax(q_list[i].numpy().T + banned_list[i]))
        
    actions = [torch.argmax(q_val, dim=0) for q_val in q_list]
    
    return actions, q_list
    
class QNet(nn.Module):
    def __init__(self, s2v_module = None):
        super(QNet, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.out_dim == 0:
            embed_dim = cmd_args.latent_dim
        else:
            embed_dim = cmd_args.out_dim
        
        #if local_args.mlp_hidden:
        self.add_linear_1 = nn.Linear(embed_dim, local_args.mlp_hidden)
        self.add_linear_out = nn.Linear(local_args.mlp_hidden, 1)
        
        self.sub_linear_1 = nn.Linear(embed_dim, local_args.mlp_hidden)
        self.sub_linear_out = nn.Linear(local_args.mlp_hidden, 1)
                
        weights_init(self)

        if s2v_module is None:
            self.s2v = model(latent_dim=cmd_args.latent_dim, 
                            output_dim=cmd_args.out_dim,
                            num_node_feats=2,
                            num_edge_feats=0,
                            max_lv=cmd_args.max_lv)
        else:
            self.s2v = s2v_module

    # batch_graph is the graph, picked_nodes is the edge stub
    def PrepareFeatures(self, batch_graph, picked_nodes):
        
        n_nodes = batch_graph[0].num_nodes
        node_feat_list = []
        
        for i in range(len(batch_graph)):
            if picked_nodes is not None and picked_nodes[i] is not None:
                assert picked_nodes[i] >= 0 and picked_nodes[i] < batch_graph[i].num_nodes

            node_feat = torch.zeros(n_nodes, 2)
            node_feat[:, 0] = 1.0

            if len(picked_nodes) >= i:
                node_feat.numpy()[picked_nodes[i], 1] = 1.0
                node_feat.numpy()[picked_nodes[i], 0] = 0.0
                
            node_feat_list.append(node_feat)

        return node_feat_list

    # type = 0 for add, 1 for subtract
    def forward(self, states, actions=None, greedy_acts = False, _type=0):
        
        batch_graph, picked_nodes, banned_list = zip(*states)
        
        
        # --------------- Make Banned List ---------------
        
        banned_list = []

        for g_ind in range(len(batch_graph)):
            g_netx = batch_graph[g_ind].to_networkx()
            mask = np.zeros(20)
            
            if(picked_nodes[g_ind] is not None):
                
                banned_idxs = []
                for _node in range(20):
                    
                    if _type == 0:
                        if g_netx.has_edge(_node, picked_nodes[g_ind]):
                            banned_idxs.append(_node)
                    
                    if _type == 1:
                        if g_netx.has_edge(_node, picked_nodes[g_ind]) == False:
                            banned_idxs.append(_node)
                            
                mask[banned_idxs] = -100
                
            banned_list.append(mask)
            
        # ------------------------------------------------
        
        node_feat = self.PrepareFeatures(batch_graph, picked_nodes)
        
        if cmd_args.ctx == 'gpu':
            node_feat = node_feat.cuda()
 
        embed = []
        graph_embed = []
        
        for i in range(len(batch_graph)):
            
            tmp_embed, tmp_graph_embed = self.s2v([batch_graph[i]], node_feat[i], None, pool_global=True)
            
            embed.append(tmp_embed)
            graph_embed.append(tmp_graph_embed)
        
        #embed = torch.cat(embed)
        #graph_embed = torch.cat(graph_embed)

        #print(embed.shape)
        #print(graph_embed.shape)
        
        #embed_s_a = torch.cat((embed, graph_embed), dim=0)
        embed_s_a = torch.cat(embed, dim=0)

        if _type:
            embed_s_a = F.relu( self.sub_linear_1(embed_s_a) )
            raw_pred = self.sub_linear_out(embed_s_a)
            print("using Sub network")
        else:
            embed_s_a = F.relu( self.add_linear_1(embed_s_a) )
            raw_pred = self.add_linear_out(embed_s_a)
            print("using Add network")
                    
        if greedy_acts:
            actions, raw_pred = greedy_actions(raw_pred, banned_list, _type=_type)
            
        return actions, raw_pred