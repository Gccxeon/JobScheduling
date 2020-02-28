import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
  """
  An actor network used as the action policy of the environment

  """
  def __init__(self,
               in_features,
               action_minimum,
               action_maximum,
               conv_params=None,
               rnn_params=None,
               fc_params=None,
               custom_network=None):
    """
    Args:
      in_features: The input dimension;
      action_minimum: The minimum of the action space (the actions space must
        be continuous);
      action_maximum: The maximium of the action space;
      conv_params: The convolutional layer params;
      rnn_params: The rnn layer params;
      fc_params: The fully connected layer parameters;
      custom_network: The custom layers of the model. If you want to completely
        using the custom_network only, set the rest of the network params as
        None or just ignore them.
    """

    self._in_features = in_features
    self._action_space = action_sapce

    # Set up the convolutional layers
    if conv_params:

    self._conv_params = conv_params
    self._rnn_params = rnn_params
    self._fc_params = fc_params
    self._model = nn.Sequential()



