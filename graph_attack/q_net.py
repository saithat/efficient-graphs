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

def greedy_actions(q_values, banned_list):
        
    actions = []
    offset = 0
    banned_acts = []
    
    q_values = q_values.data.clone()
    
    # To Do:
    #
    # Banned actions get minimum value
    #
    #
    
    num_chunks = q_values.shape[0] // 20
        
    q_list = torch.chunk(q_values[:num_chunks * 20], num_chunks, dim=0)
    
    actions = [torch.argmax(q_val, dim=0) for q_val in q_list]
    
    return actions
    
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

    def add_offset(self, actions, v_p):
        prefix_sum = v_p.data.cpu().numpy()

        shifted = []        
        for i in range(len(prefix_sum)):
            if i > 0:
                offset = prefix_sum[i - 1]
            else:
                offset = 0
            shifted.append(actions[i] + offset)

        return shifted

    def rep_global_embed(self, graph_embed, prefix_sum):
        rep_idx = []        
        for i in range(len(prefix_sum)):
            if i == 0:
                n_nodes = prefix_sum[i]
            else:
                n_nodes = prefix_sum[i] - prefix_sum[i - 1]
            rep_idx += [i] * n_nodes

        rep_idx = Variable(torch.LongTensor(rep_idx))
        if cmd_args.ctx == 'gpu':
            rep_idx = rep_idx.cuda()
        graph_embed = torch.index_select(graph_embed, 0, rep_idx)
        return graph_embed

    # type = 0 for add, 1 for subtract
    def forward(self, states, actions, greedy_acts = False, _type=0):
        
        batch_graph, picked_nodes, banned_list = zip(*states)

        node_feat = self.PrepareFeatures(batch_graph, picked_nodes)
        
        
        if cmd_args.ctx == 'gpu':
            node_feat = node_feat.cuda()

            
        embed = []
        graph_embed = []
        
        for i in range(len(batch_graph)):
            tmp_embed, tmp_graph_embed = self.s2v([batch_graph[i]], node_feat[i], None, pool_global=True)
            
            embed.append(tmp_embed)
            graph_embed.append(tmp_graph_embed)
                    
        dummy = torch.zeros(80, 2)
        
        tmp_embed, tmp_graph_embed = self.s2v(batch_graph, dummy, None, pool_global=True)    
        
        #embed = torch.FloatTensor(embed)
        #graph_embed = torch.FloatTensor(graph_embed)
        
        embed = torch.cat(embed)
        graph_embed = torch.cat(graph_embed)

        embed_s_a = torch.cat((embed, graph_embed), dim=0)

        #if local_args.mlp_hidden:
        if _type:
            embed_s_a = F.relu( self.sub_linear_1(embed_s_a) )
            raw_pred = self.sub_linear_out(embed_s_a)
        else:
            embed_s_a = F.relu( self.add_linear_1(embed_s_a) )
            raw_pred = self.add_linear_out(embed_s_a)
                    
        if greedy_acts:
            actions = greedy_actions(raw_pred, banned_list)
            
        return actions, raw_pred

"""
class NStepQNet(nn.Module):
    def __init__(self, num_steps, s2v_module = None):
        super(NStepQNet, self).__init__()

        list_mod = [QNet(s2v_module)]

        for i in range(1, num_steps):
            list_mod.append(QNet(list_mod[0].s2v))
        
        self.list_mod = nn.ModuleList(list_mod)

        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts = False):
        assert time_t >= 0 and time_t < self.num_steps

        return self.list_mod[time_t](time_t, states, actions, greedy_acts)
"""
