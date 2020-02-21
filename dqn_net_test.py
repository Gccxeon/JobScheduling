from dqn_net import DqnNet
import torch

conv_para = {"conv_dim": [1], "args": [[1, 10 ,3]], "kw_args": [None], "activation": ['ReLU']}
fc_param = {"units": [48, 32], "bias": [True, True], "activation": ['ReLU']}
network = DqnNet(10, 10, 10, conv_network_param=None, fc_network_param=fc_param)
