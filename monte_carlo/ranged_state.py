import math

class RangedState():
  """
  A state representation that have the same identity if the internal states of
  each RangedState are not far from each other.
  """
  def __init__(self, internal_state, tolerance):
    """
    Args:
      internal_state: A python numerical sequence, The state comes from the raw
          return of the environment;
      tolerance: The allowed difference that will be ignored in the comparision
          of the internal_state.
    """
    self._internal_state = internal_state
    self._tolerance = tolerance


  def __eq__(self, other):
    for i in range(len(self._internal_state)):
      if abs(self._internal_state[i] - other._internal_state[i]) > self.tolerance
      return False
    return True
