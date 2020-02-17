import torch.nn


class Sequential(self):
  """
  Accept a list of network and the dimension of input. The input size of the
    networks in the list will be automatically computed and a suequntialized
    network will be composed.
  """
  def __init__(self, input_dim, networks_params):
    self._net = self._build(input_dim, networks)

  def _build(self, input_dim, networks):
    for net in networks:
     pass
