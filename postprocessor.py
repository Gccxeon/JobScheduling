import torch

class PostProcessor():
  # A clas for preprocessing the state reperesentations
  def __init__(self,
               str2num_dict):
    """
    Args:
      state: The reperesentation of the return from the environment after
        taking an action.
    """
    self._type_rep = str2num_dict

  def state_process(self, state):
    if state is None:
      return None
    job_type = state['type']
    intensity = state['intensity']
    response_time = state['response_time']
    job_as_number = self._type_to_number(job_type)
    state_array = [job_as_number] + [intensity] + response_time
    return state_array

  def process(self, transition):
    state = torch.tensor(self.state_process(transition.state)).unsqueeze(0)
    action = torch.tensor(transition.action).unsqueeze(0)
    reward = torch.tensor(transition.reward).unsqueeze(0)
    if transition.terminal:
      next_state = None
    else:
      next_state = (
          torch.tensor(self.state_process(transition.next_state)).unsqueeze(0))
    terminal = transition.terminal
    return type(transition)(state, action, reward, next_state, terminal)


  def _type_to_number(self, type_name):
    return self._type_rep[type_name]
