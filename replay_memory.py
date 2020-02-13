from collections import deque
from transition import Transition
from random import sample

class ReplayMemory():
  """
  A class used to store the Transitions during the learning process
  """
  def __init__(self,
               data_type,
               max_capacity):
    self._data_type = data_type
    self._max_capacity = max_capacity
    self._data = deque([], maxlen=max_capacity)
    self._position = 0
    self._size = 0


  def sample(self, batch_size):
    if self._size < batch_size:
      raise ValueError("Current repaly memory size is {}, "
                       "the required batch_size is {},"
                       "which is too large for what you have."
                       .format(self._size, batch_size))

    return sample(self._data, batch_size)


  def add(self, transition):
    if type(transition) != self._data_type:
      raise TypeError("The given transition isn't consistent with the "
                      "specified spec requirement, required {}, found {}."
                      .format(self._data_type, type(transition)))
    self._data.append(transition)
    self._size = min(self._size + 1, self._max_capacity)
