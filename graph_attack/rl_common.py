import argparse
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
from copy import deepcopy
import pickle as cp

cmd_opt = argparse.ArgumentParser(description='Argparser locally')
cmd_opt.add_argument('-mlp_hidden', type=int, default=64, help='mlp hidden layer size')
cmd_opt.add_argument('-att_embed_dim', type=int, default=64, help='att_embed_dim')
cmd_opt.add_argument('-num_steps', type=int, default=100000, help='# fits')
local_args, _ = cmd_opt.parse_known_args()

print(local_args)

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from graph_embedding import S2VGraph
from cmd_args import cmd_args
from dnn import GraphClassifier

from message import get_y_add, get_y_sub

sys.path.append('%s/../data_generator' % os.path.dirname(os.path.realpath(__file__)))
from data_util import load_pkl

sys.path.append('%s/../graph_classification' % os.path.dirname(os.path.realpath(__file__)))
from graph_common import loop_dataset

GLOBAL_PREFIX = 20

# replacement for deprecated nx.connected_component_subgraphs()
def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

class GraphEdgeEnv(object):
    def __init__(self, classifier):
        self.classifier = classifier

    def setup(self, g_list):
        self.g_list = g_list
        self.n_steps = 0
        self.first_nodes = None
        self.rewards = None
        self.banned_list = None
        self.prefix_sum = []
        
        n_nodes = 0
        
        for i in range(len(g_list)):
            n_nodes += g_list[i].num_nodes
            self.prefix_sum.append(n_nodes)
            
        self.added_edges = []

    def bannedActions(self, g, node_x):        
        comps = [c for c in connected_component_subgraphs(g)]
        set_id = {}
        for i in range(len(comps)):
            for j in comps[i].nodes():
                set_id[j] = i

        banned_actions = set()
        for i in range(len(g)):
            
            try:
                if set_id[i] != set_id[node_x] or i == node_x:
                    banned_actions.add(i)
            except:
                banned_actions.add(0)
                
        return banned_actions
    
    # type = 0 for add, 1 for subtract
    def get_rewards(self, actions, _type=0):
        
        # for i in range(len(g_list)):
        #     edge stub = self.first_nodes[i]
        #     graph = g_list[i]
        
        rewards = []
        
        for i in range(len(self.g_list)):
            g = self.g_list[i].to_networkx()
            
            if _type:
                Y = get_y_sub(g, self.first_nodes[i])
            else:
                Y = get_y_add(g, self.first_nodes[i])
            
            R = Y[actions[i]]
            
            rewards.append(R)
            
        return rewards
            

    # type = 0 for add, 1 for subtract
    def step(self, actions, _type = 0):
        
        # if edge stub is none
        if self.first_nodes is None: # pick the first node of edge
            assert self.n_steps % 2 == 0
            
            # set edge stub to action
            self.first_nodes = actions
            self.banned_list = []
            
            for i in range(len(self.g_list)):
                self.banned_list.append(self.bannedActions(self.g_list[i].to_networkx(), self.first_nodes[i])) 
        
        # if edge stub is not None
        else:   
            
            #self.added_edges = []

            for i in range(len(self.g_list)):
            
                g = self.g_list[i].to_networkx()
               
                if _type:
                    # remove edge between edge stub and action
                    
                    if g.has_edge(self.first_nodes[i], actions[i]):
                        g.remove_edge(self.first_nodes[i], actions[i])

                else:
                    # create edge between edge stub and action
                    g.add_edge(self.first_nodes[i], actions[i])

            
                self.added_edges.append((self.first_nodes[i], actions[i]))
            
                self.g_list[i] = S2VGraph(g, label = self.g_list[i].label)
             
            # set edge stub to none
            self.first_nodes = None
            self.banned_list = None
        
        self.n_steps += 1

        if self.isTerminal():
            logits, _, acc = self.classifier(self.g_list)
            pred = logits.data.max(1, keepdim=True)[1]
            self.pred = pred.view(-1).cpu().numpy()
            self.rewards = (acc.view(-1).numpy() * -2.0 + 1.0).astype(np.float32)

    def uniformRandActions(self):
        act_list = []
        for i in range(len(self.g_list)):
            if self.first_nodes[i] is None:
                act_list.append(np.random.randint(GLOBAL_PREFIX))
            else:
                banned_actions = self.banned_list[i]
                cands = list(set(range(GLOBAL_PREFIX)) - banned_actions)
                act_list.append(random.choice(cands))
        return act_list

    #def really_random(self):
    #    act_list = []
    #    for i in range(len(self.g_list)):
    #        act_list.append(np.random.randint(GLOBAL_PREFIX))
    #    return act_list

    def sampleActions(self, probs, greedy=False):
        offset = 0
        act_list = []
        for i in range(len(self.prefix_sum)):
            p_vec = probs[offset : self.prefix_sum[i], 0].astype(np.float64)

            if self.first_nodes is not None:
                banned_actions = self.banned_list[i]
                for j in banned_actions:
                    p_vec[j] = 0.0
                assert len(banned_actions) < len(p_vec)

            p_vec = p_vec / sum(p_vec)
            if greedy:
                action = np.argmax(p_vec)
            else:
                action = np.argmax(np.random.multinomial(1, p_vec))
            act_list.append(action)
            offset = self.prefix_sum[i]

        return act_list

    def isTerminal(self):
        return False

    def getStateRef(self):
        cp_first = [None] * len(self.g_list)
        if self.first_nodes is not None:
            cp_first = self.first_nodes
        b_list = [None] * len(self.g_list)
        if self.banned_list is not None:
            b_list = self.banned_list            
        return zip(self.g_list, cp_first, b_list)

    def cloneState(self):
        cp_first = [None] * len(self.g_list)
        if self.first_nodes is not None:
            cp_first = self.first_nodes[:]
        b_list = [None] * len(self.g_list)
        if self.banned_list is not None:
            b_list = self.banned_list[:]

        return list(zip(deepcopy(self.g_list), cp_first, b_list))

def load_graphs(graph_tuples, n_graphs, frac_train=None):
    
    if (frac_train is not None):
        frac_train = frac_train
    else:
        frac_train = 0.8
        
    num_train = int(frac_train * n_graphs)
    
    train_glist = [S2VGraph(graph_tuples[j]) for j in range(num_train)]
    test_glist = [S2VGraph(graph_tuples[j]) for j in range(num_train, n_graphs)]
    
    print('# train:', len(train_glist), ' # test:', len(test_glist))

    return train_glist, test_glist

def test_graphs(classifier, test_glist):
    test_loss = loop_dataset(test_glist, classifier, list(range(len(test_glist))))
    print('\033[93maverage test: loss %.5f acc %.5f\033[0m' % (test_loss[0], test_loss[1]))

def load_base_model(label_map, test_glist = None):
    assert cmd_args.base_model_dump is not None
    with open('%s-args.pkl' % cmd_args.base_model_dump, 'rb') as f:
        base_args = cp.load(f)

    classifier = GraphClassifier(label_map, **vars(base_args))
    if cmd_args.ctx == 'gpu':
        classifier = classifier.cuda()

    classifier.load_state_dict(torch.load(cmd_args.base_model_dump + '.model'))
    if test_glist is not None:
        test_graphs(classifier, test_glist)

    return classifier
"""
def attackable(classifier, s2v_g, x = None, y = None):
    g = s2v_g.to_networkx()
    comps = [c for c in nx.connected_component_subgraphs(g)]
    set_id = {}

    for i in range(len(comps)):
        for j in comps[i].nodes():
            set_id[j] = i
    
    if x is not None:
        r_i = [x]
    else:
        r_i = range(len(g) - 1)

    g_list = []    
    for i in r_i:
        if y is not None:
            assert x is not None
            r_j = [y]
        else:
            if x is not None:
                r_j = range(len(g) - 1)
            else:
                r_j = range(i + 1, len(g))
        for j in r_j:
            if set_id[i] != set_id[j]:
                continue
            g2 = g.copy()
            g2.add_edge(i, j)
            assert nx.number_connected_components(g2) == s2v_g.label
            g_list.append(S2VGraph(g2, s2v_g.label))
    if len(g_list) == 0:
        print(x, y)
        print(g.edges(), s2v_g.label)
        print(set_id)
    assert len(g_list)
    _, _, acc = classifier(g_list)

    return np.sum(acc.view(-1).numpy()) < len(g_list)

def get_supervision(classifier, list_st, list_at):
    list_target = torch.zeros(len(list_st))
    for i in range(len(list_st)):
        g, x, _ = list_st[i]
        if x is not None:
            att = attackable(classifier, g, x, list_at[i])
        else:
            att = attackable(classifier, g, list_at[i])

        if att:
            list_target[i] = 1.0
        else:
            list_target[i] = -1.0
    
    return list_target
"""