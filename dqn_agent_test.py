from dqn_net import DqnNet
from dqn_agent import DqnAgent
from test import env, replay_memory, collector
import torch.nn as nn
from preprocess_dqn import process_from_replay_sample
batch_size = 4

conv_para = {"conv_dim": [1],
             "args": [[1,2,1]],
             "kw_args": [None],
             "activation": ['ReLU']}
fc_param = {"units": [48, 32], "bias": [True, True], "activation": ['ReLU']}

network = DqnNet(12, 10,
                 conv_network_param=conv_para,
                 fc_network_param=fc_param)

loss_fn = nn.functional.smooth_l1_loss

agent = DqnAgent(environment=env,
                 network=network,
                 batch_size=batch_size,
                 loss_fn=loss_fn,
                 optimizer='Adam',
                 discount=0.98,
                 update_tau=0.5,
                 update_period=30,
                 learning_rate=1e-3,
                 eps_greedy=0.9,
                 eps_decay_count=200,
                 eps_minimum=0.1)

raw_sample = replay_memory.sample(batch_size)
sample = process_from_replay_sample(raw_sample)

#agent.train_step(sample)
