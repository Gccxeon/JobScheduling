from dqn_net import DqnNet
import torch

conv_para = {"conv_dim": [1], "args": [[3,2,1]], "kw_args": [None], "activation": ['ReLU']}
fc_param = {"units": [48, 32], "bias": [True, True], "activation": ['ReLU']}
network = DqnNet(10, 10, 10, conv_network_param=conv_para, fc_network_param=fc_param)

network.forward(torch.rand(10,3,10)
