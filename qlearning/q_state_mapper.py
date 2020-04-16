# Mapping continuous states from the job scheduling environment to discrete
# states in order to perform tabular Q learning.
class QStateMapper():
  def __init__(self,
               state):
    """
    Args:
      state:  Original state (current response time in each server)
    """
    server_type = state["type"]
    sstate = state["response_time"]
    self.s_type = 1 if server_type == 'CPU' else 0
    self.sorted_state_idx = sorted(range(len(sstate)),
                                   key=lambda k: sstate[k])
    self.d_state = [self.s_type] + self.sorted_state_idx


  def __eq__(self, other):
    return True if self.d_state == other.d_state else False

  def __hash__(self):
    return hash(tuple(self.d_state))
