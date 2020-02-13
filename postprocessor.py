from torchvision import transforms

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
    job_type = state['type']
    intensity = state['intensity']
    response_time = state['response_time']
    job_as_number = self._type_to_number(job_type)
    state_array = [job_as_number] + [intensity] + response_time
    return state_array

  def process(self, transition):
    state = self.state_process(transition.state)
    action = transition.action
    reward = transition.reward
    next_state = self.state_process(transition.next_state)
    return type(transition)(state, action, reward, next_state)


  def _type_to_number(self, type_name):
    return self._type_rep[type_name]
