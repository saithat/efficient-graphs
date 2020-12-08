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

        if random.random() < self.eps and not greedy:
            actions = self.env.uniformRandActions()
        else:
            cur_state = self.env.getStateRef()
            
            actions, _ = self.net(cur_state, None, greedy_acts=True, _type=_type)
            
            actions = torch.cat(actions)
            actions = actions.numpy().tolist()
                        
        return actions

    def run_simulation(self):

        self.env.setup(g_list)

        t_a, t_s = 0, 0
        
        #while not env.isTerminal():
        for asdf in range(50):
            
            if asdf % 2 == 0:
                assert self.env.first_nodes == None
            assert self.g_list[0].num_nodes == 20 or self.g_list[0].num_nodes == 21
            
            # generate action
            if len(self.g_list) == 20:
                action_type = 0
            else:
                action_type = 1
                
            list_at = self.make_actions(_type=action_type)
                        
            list_st = self.env.cloneState()
            
            # get rewards
            
            if self.env.first_nodes is not None:
                rewards = self.env.get_rewards(list_at, _type=action_type)
            else:
                rewards = [0] * len(g_list)
            
            # execute the action to update the graph
            self.env.step(list_at, _type = action_type)
            
            # get next state
            if env.isTerminal():
                s_prime = None
            else:
                s_prime = self.env.cloneState()
                
            #print(list_st, list_at, rewards, s_prime, [env.isTerminal()] * len(list_at), t_a)
            
            # Predicted = q_val[action]
            # Actual = Reward[]
            
            print("\n\nActions:", list_at)
            print("\n\nRewards:", rewards)
                            
            if action_type == 0:
                self.add_mem_pool.add_list(list_st, list_at, rewards, s_prime, [env.isTerminal()] * len(list_at), t_a % 2)
                t_a += 1
            else:
                self.sub_mem_pool.add_list(list_st, list_at, rewards, s_prime, [env.isTerminal()] * len(list_at), t_s % 2)
                t_s += 1
                
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
        #pbar = tqdm(range(self.burn_in), unit='batch')
        
        # maybe warm up?
        #for p in pbar:
        #    self.run_simulation()
        
        # set up real progress bar
        pbar = tqdm(range(GLOBAL_NUM_STEPS), unit='steps')      # number of iterations to train?
        
        # set optimizer
        optimizer = optim.Adam(self.net.parameters(), lr=cmd_args.learning_rate)
        
        # for each iteration
        for self.step in pbar:

            # run simulation
            # side effects?
            self.run_simulation()

            # save weights and evalute
            if self.step % 100 == 0:
                self.take_snapshot()
            if self.step % 100 == 0:
                r, acc = self.eval()
                log_out.write('%d %.6f %.6f\n' % (self.step, r, acc))

            #
            # memory replay sample? figure out later
            # 
            
            # list_st = states, list_at = actions
            cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.add_mem_pool.sample(
                batch_size=cmd_args.batch_size)
            
            cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.sub_mem_pool.sample(
                batch_size=cmd_args.batch_size)

            
            list_target = torch.Tensor(list_rt)
            
            if cmd_args.ctx == 'gpu':
                list_target = list_target.cuda()

            cleaned_sp = []
            nonterms = []
            for i in range(len(list_st)):
                if not list_term[i]:
                    cleaned_sp.append(list_s_primes[i])
                    nonterms.append(i)

            if len(cleaned_sp):
                _, _, banned = zip(*cleaned_sp)
                _, q_t_plus_1, prefix_sum_prime = self.old_net(cur_time + 1, cleaned_sp, None)
                _, q_rhs = greedy_actions(q_t_plus_1, prefix_sum_prime, banned)
                list_target[nonterms] = q_rhs
            
            # list_target = get_supervision(self.env.classifier, list_st, list_at)
            list_target = Variable(list_target.view(-1, 1))

            # q_sa = raw_pred     
            _, q_sa_add, _ = self.net(1, cur_time, list_st, list_at)
            _, q_sa_sub, _ = self.net(0, cur_time, list_st, list_at)
            
            # list_target [add_target, sub_target]
            
            loss_add = F.mse_loss(q_sa_add, list_target[0])
            optimizer.zero_grad()
            loss_add.backward()
            optimizer.step()
            
            loss_sub = F.mse_loss(q_sa_sub, list_target[1])
            optimizer.zero_grad()
            loss_sub.backward()
            optimizer.step()
            
            pbar.set_description('exp: %.5f, loss: %0.5f' % (self.eps, loss_add + loss_sub) )

        log_out.close()

GLOBAL_PHASE = 'train'
GLOBAL_NUM_STEPS = 1

if __name__ == '__main__':
    
    # generate graphs
    n_graphs = 5
    output = Generate_dataset(n_graphs)
    g_list, test_glist = load_graphs(output, n_graphs)
    
    #base_classifier = load_base_model(label_map, g_list)
    
    base_args = {'gm': 'mean_field', 'feat_dim': 2, 'latent_dim': 10, 'out_dim': 20, 'max_lv': 2, 'hidden': 32}
    base_classifier = GraphClassifier(num_classes=20, **base_args)
    
    env = GraphEdgeEnv(base_classifier)
    
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
        
        # env.setup([g_list[idx] for idx in selected_idx])
        # t = 0
        # while not env.isTerminal():
        #     policy_net = net_list[t]
        #     t += 1            
        #     batch_graph, picked_nodes = env.getState()
        #     log_probs, prefix_sum = policy_net(batch_graph, picked_nodes)
        #     actions = env.sampleActions(torch.exp(log_probs).data.cpu().numpy(), prefix_sum.data.cpu().numpy(), greedy=True)
        #     env.step(actions)

        # test_loss = loop_dataset(env.g_list, base_classifier, list(range(len(env.g_list))))
        # print('\033[93maverage test: loss %.5f acc %.5f\033[0m' % (test_loss[0], test_loss[1]))
        
        # print(np.mean(avg_rewards), np.mean(env.rewards))