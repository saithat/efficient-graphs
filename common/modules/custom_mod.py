from torch.nn.modules.module import Module
#from functions.custom_func import JaggedLogSoftmax, JaggedArgmax, JaggedMax
import networkx as nx
import numpy as np
import torch

class JaggedLogSoftmaxModule(Module):
    def forward(self, logits, prefix_sum):
        #return JaggedLogSoftmax()(logits, prefix_sum)
        return torch.nn.LogSoftmax()(logits)

class JaggedArgmaxModule(Module):
    def forward(self, values, prefix_sum):
        #return JaggedArgmax()(values, prefix_sum)
        return torch.argmax(values)

class JaggedMaxModule(Module):
    def forward(self, values, prefix_sum):
        #return JaggedMax()(values, prefix_sum)
        return torch.max(values)

