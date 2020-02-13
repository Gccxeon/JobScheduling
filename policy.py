class Policy():
  """
  A class that support `action()` function and can be used to generate or
    wrap the functions that give out the action of agent
  """
  def __init__(self,
               policy_fn):
    self._policy_fn = policy_fn

  def action(self):
    return self._policy_fn()
