import torch.nn as nn
import torch.nn.functional as F

class Dense(nn.Module):
  def __init__(self, in_features, params):
    self.net = []
    in_features = in_features
    for out_features in params:
      self.net.append(nn.Linear(in_features, out_features))
      self.net.append(F.relu)
      in_features = out_features

  def forward(self, input):
    for layer in self.net:
      output = layer(input)
      input = output
    return output

class NetBuilder():
  def __init__(self, in_features, build_params):
    """
    Args:

      in_features: The shape of input;

      build_params: and ordered dict specifies that:

        "type": the type of current layer; the name should be exactly the same
        as the ones included in the torch.nn package. For example, if user
        wants to use torch.nn.Conv1d, then the value of "type" should be
        "Conv1d".

        "args": The args of the corresponding networks.

        "activations": The actiavtion functions used on the output of each
        layer. The vaule of this keyword should be either a single string or
        a list of strings with length == len(build_params). For omitting the
        use of actiavtion function in some layers, specify them as None;
        otherwise, the string should comes from the torch.nn.functional package.
        For example, [None, "relu"] will omit the actiavtion of the output of
        first layer, and the output of the second layer will have activation
        function torch.nn.functional.relu().
    """
