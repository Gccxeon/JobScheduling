import torch
import numpy

class DqnAgent(self):
  """
  A reinforcement learning agent used for job scheduling environment
  """
  def __init__(self,
               environment,
               preprocessor,
               network,
               batch_size,
               optimizer,
               replay_buffer,
               epoch,
               iteration):
    self._env = environment
    self._preprocessor = preprocessor
    self._net = network
    self._optimizer = optimizer
    self._replay_buffer = replay_buffer
    self._epoch = epoch
    self._iteration = iteration


  def unpack(self, transition):
    state, action, reward, next_state = transition
    return state, action, reward, next_state

  def get_input(self):
    state, action, reward, next_state = self._replay_buffer.sample(batch_size)
    state_p = self._preprocessor(state)
    next_state_p =self._preprocessor(next_state)
    return (state_p, action, reward, next_state_p)



  def train(self)
    """
    Args:
      iteration: The training cicle
    """
    for i in range(self._epoch):
      for i in range(self._iteration):
        state, action, reward, next_step = self.get_input()
        logits = self._net(state)

