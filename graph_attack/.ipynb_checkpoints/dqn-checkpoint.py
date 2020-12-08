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
from copy import deepcopy

from q_net import QNet, greedy_actions
sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

import warnings
warnings.filterwarnings("ignore")

from rl_common import GraphEdgeEnv, local_args, load_graphs, test_graphs, load_base_model
from nstep_replay_mem import NstepReplayMem

sys.path.append('%s/../graph_classification' % os.path.dirname(os.path.realpath(__file__)))

from dnn import GraphClassifier

from graph_common import loop_dataset

from message import Generate_dataset

class Agent(object):
    def __init__(self, g_list, test_g_list, env):
        self.g_list = g_list
        if test_g_list is None:
            self.test_g_list = g_list
        else:
            self.test_g_list = test_g_list
            
        self.add_mem_pool = NstepReplayMem(memory_size=50000, n_steps=2)
        self.sub_mem_pool = NstepReplayMem(memory_size=50000, n_steps=2)
        self.env = env
        self.net = QNet()
        self.old_net = QNet()
        self.optimizer = optim.Adam(self.net.parameters(), lr=cmd_args.learning_rate)


        if cmd_args.ctx == 'gpu':
            self.net = self.net.cuda()
            self.old_net = self.old_net.cuda()

        self.eps_start = 1.0
        self.eps_end = 1.0
        self.eps_step = 10000
        self.burn_in = 100              # number of iterations to run first set ("intial burning in to memory") of simulations?
        self.step = 0

        self.best_eval = None
        self.pos = 0
        self.sample_idxes = list(range(len(g_list)))
        random.shuffle(self.sample_idxes)
        self.take_snapshot()

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    # type = 0 for add, 1 for subtract
    def make_actions(self, greedy=True, _type = 0):
        self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                * (self.eps_step - max(0., self.step)) / self.eps_step)

        cur_state = self.env.getStateRef()

        actions, q_arrs = self.net(cur_state, None, greedy_acts=True, _type=_type)

        actions = torch.cat(actions)
        actions = actions.numpy().tolist()

        q_vals = []

        for i in range(len(q_arrs)):
            tmp = q_arrs[i].numpy()
            tmp = tmp[actions[i]][0]
            q_vals.append(tmp)
                        
        return actions, q_vals
    
    def Q_loss(self, q_vals, rewards, q_primes):
        
        # predicted_Q = Q(S, A)
        predicted_Q = []
        
        for i in range(len(q_vals)):
            predicted_Q.append(rewards[i] + q_primes[i])
            
        predicted_Q = np.array(predicted_Q)
        
        # actual_Q = (R + max_a Q(S', a))
        actual_Q = np.array(q_vals)
        
        # Loss = (actual_Q - predicted_Q) ^ 2
        loss = (predicted_Q - actual_Q) ** 2
        
        return torch.from_numpy(loss)

    def run_simulation(self):

        self.env.setup(g_list)

        t_a, t_s = 0, 0
        
        #while not env.isTerminal():
        for asdf in range(GLOBAL_EPISODE_STEPS):
            
            if asdf % 2 == 0:
                assert self.env.first_nodes == None
            
            action_type = (asdf % 4) // 2
                
            # get Actions
            list_at, _ = self.make_actions(_type=action_type)
                   
            # save State
            list_st = self.env.cloneState()
            
            cur_state = self.env.getStateRef()
            
            _, predicted_Q = self.net(cur_state, None, greedy_acts=False, _type=action_type)
                        
            # get Rewards
            if self.env.first_nodes is not None:
                rewards = self.env.get_rewards(list_at, _type=action_type)
            else:
                rewards = [0] * len(g_list)
            
            # Update graph to get S'
            self.env.step(list_at, _type = action_type)
            
            # get next state
            if env.isTerminal():
                s_prime = None
            else:
                s_prime = self.env.cloneState()
            
            # get Q(S', A) values
            try:
                sprime_at, q_primes = self.make_actions(_type=action_type)
            
            except:
                continue
            
            actual_Q = torch.Tensor(rewards) + torch.Tensor(q_primes)
            
            print("\n\nActions:", list_at)
            print("\n\nQ_vals:", predicted_Q.shape)
            print("\n\nRewards:", rewards)
            print("\n\nQ_prime:", q_primes)
            
            loss = F.mse_loss(predicted_Q, actual_Q)
            
            # pass loss to network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                
    def eval(self):
        self.env.setup(deepcopy(self.test_g_list))
        t = 0
        
        while not self.env.isTerminal():
            list_at = self.make_actions(greedy=True, _type=(t % 4) // 2)
            self.env.step(list_at)
            t += 1
            
        test_loss = loop_dataset(env.g_list, env.classifier, list(range(len(env.g_list))))
        
        print('\033[93m average test: loss %.5f acc %.5f\033[0m' % (test_loss[0], test_loss[1]))

        if cmd_args.phase == 'train' and self.best_eval is None or test_loss[1] < self.best_eval:
            print('----saving to best attacker since this is the best attack rate so far.----')
            torch.save(self.net.state_dict(), cmd_args.save_dir + '/epoch-best.model')
            with open(cmd_args.save_dir + '/epoch-best.txt', 'w') as f:
                f.write('%.4f\n' % test_loss[1])
            self.best_eval = test_loss[1]

        reward = np.mean(self.env.rewards)
        print(reward)
        return reward, test_loss[1]

    def train(self):
        
        # set up progress bar
        pbar = tqdm(range(GLOBAL_NUM_STEPS), unit='steps')
        
        # for each iteration
        for self.step in pbar:

            # run simulation
            self.run_simulation()
            
            exit()

GLOBAL_PHASE = 'train'
GLOBAL_NUM_STEPS = 1
GLOBAL_EPISODE_STEPS = 500
GLOBAL_NUM_GRAPHS = 10

if __name__ == '__main__':
    
    # generate graphs
    output = Generate_dataset(GLOBAL_NUM_GRAPHS)
    g_list, test_glist = load_graphs(output, GLOBAL_NUM_GRAPHS)
    
    #base_classifier = load_base_model(label_map, g_list)
    
    base_args = {'gm': 'mean_field', 'feat_dim': 2, 'latent_dim': 10, 'out_dim': 20, 'max_lv': 2, 'hidden': 32}
    base_classifier = GraphClassifier(num_classes=20, **base_args)
    
    env = GraphEdgeEnv(base_classifier)
    
    print("len g_list:", len(g_list))
    
    if cmd_args.frac_meta > 0:
        num_train = int( len(g_list) * (1 - cmd_args.frac_meta))
        agent = Agent(g_list, test_glist[num_train:], env)
    else:
        agent = Agent(g_list, None, env)
    
    if GLOBAL_PHASE == 'train':
        
        print("\n\nStarting Training Loop\n\n")
        
        agent.train()
        
    else:
        print("\n\nStarting Evaluation Loop\n\n")
        agent.net.load_state_dict(torch.load(cmd_args.save_dir + '/epoch-best.model'))
        agent.eval()
