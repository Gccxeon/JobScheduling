import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DqnNet(nn.Module):
  """
  The network that accpect the state and return a set of logits as the
    prediction of the probabilities of the possible optimal action.
  """
  def __init__(self,
               input_dim,
               batch_size,
               action_space,
               fc_network_param=None,
               conv_network_param=None,
               rnn_network_param=None):
    """
    Args:
      input_dim: The dimension of the input of network;
      batch_size: The size of the batch when using mini batch training;
      conv_network_param: Example:  {"conv_dim": [1] (or 2, 3),
                                     "args": [[in_channels,
                                              out_channels,
                                              kernel_size]],
                                     "kw_args": [{"stride": 1,
                                                  "padding": 0,
                                                  ...}],
                                     "activation": ['relu']}
      rnn_network_param: Example: later;
      fc_network_param: The parameters of the fully connected network part. For
        example: {"units":[128, 64, 10],
                  "bias": [True, False, True],
                  "activation": ['ReLU','sigmoid', 'ReLU']};
      action_space: The action space of the agent, this is the dimension the
        output of the network will have.
    """
    super(DqnNet, self).__init__()

    self._in_features = input_dim
    self._batch_size = batch_size
    self._out_features = action_space
    self._conv_net = nn.Sequential()
    self._fc_net = nn.Sequential()
    self._rnn_net = nn.Sequential()


    if conv_network_param:
      conv_names = self._get_conv_name(conv_network_param["conv_dim"])
      args = conv_network_param["args"]
      kw_args = conv_network_param["kw_args"]
      if len(kw_args) == 1:
        kw_args = kw_args * len(conv_names)
      activations = conv_network_param["activation"]
      if len(activations) == 1:
        activations = activations * len(args)
      i = 0
      self._check_if_same_len(conv_names, args, kw_args, activations)
      self._conv_in_channel = args[0][0]
      for conv_name, args, kw_args, activation in zip(
          conv_names, args, kw_args, activations):
        name = "convolution_" + str(i)
        ac_name = "conv_" + activation + "_" + str(i)
        if kw_args:
          self._conv_net.add_module(name,
                                    getattr(nn, conv_name)(*args, **kw_args))
        else:
          self._conv_net.add_module(name, getattr(nn, conv_name)(*args))
        self._conv_net.add_module(ac_name, getattr(nn, activation)())
        i += 1

    if self._conv_net:
      pseudo_in = torch.rand(batch_size, self._conv_in_channel, input_dim)
      pseudo_out = self._conv_net.forward(pseudo_in)
      input_dim = pseudo_out.data.view(batch_size,-1).size(1)


    if fc_network_param:
      units = fc_network_param["units"]
      bias = fc_network_param["bias"]
      activations = fc_network_param["activation"]
      if len(bias) == 1:
        bias = bias * len(units)
      if len(activations) == 1:
        activations = activations * len(units)
      self._check_if_same_len(units, bias, activations)
      i = 0
      for unit, bias, activation in zip(units, bias, activations):
        name = "fully_connected_" + str(i)
        ac_name = "fc_" + "activation" + "_" + str(i)
        self._fc_net.add_module(name, nn.Linear(input_dim, unit, bias=bias))
        self._fc_net.add_module(ac_name, getattr(nn, activation)())
        input_dim = unit
        i += 1

    self._fc_net.add_module("final",
        nn.Linear(input_dim, self._out_features, bias=True))


    if rnn_network_param:
      # TODO
      pass


  def _check_if_same_len(self, *args):
    reference_len = len(args[0])
    for arg in args:
      if len(arg) != reference_len:
        raise ValueError("The length of arg {} is not consistent with the "
                         "previous args, abort!".format(arg))


  def _get_conv_name(self, dims):
    name_dict = {1: 'Conv1d', 2: 'Conv2d', 3: 'Conv3d'}
    conv_names = []
    for dim in dims:
      conv_names.append(name_dict[dim])
    return conv_names

  def forward(self, input):
    input = self._conv_net.forward(input) if self._conv_net else input
    input = input.view(self._batch_size, -1)
    input = self._fc_net.forward(input) if self._fc_net else input
    # TODO: add rnn layer support
    return input
