import random
import math
from collections import defaultdict
class Simulator():
  # A monte carlo style Simulator
  def __init__(self,
               preprocessor,
               env,
               eps,
               eps_decay,
               eps_minimum):
    """
    Args:
      preprocessor: The preprocessing that tansfer non-computable state into
          the comtuptable ones;
      env: The embeded environment where the Simulator operates in;
      eps: Initial value of epsilon-greedy policy;
      eps_decay: Set the number of iterations that will make the eps drops to
        its minimum;
      eps_minimum: The minimum value of eps.
    """
    self._env = env
    self._gloabl_step = 0
    self._action_log = []
    self._policy = defaultdict()
    self._action_space = env.action_space()
    self._preprocessor = preprocessor
    eps = eps + 0.01
    eps_minimum = eps_minimum + 0.01
    self._eps = eps
    self._eps_minimum = eps_minimum
    self._eps_decay_value = math.exp(math.log(eps_minimum/eps) / eps_decay)

  def random_simulate(self, iterations, tolerance):
    for i in range(iterations):
      while not self._env.is_termial():
        action = random.choice(range(action_space))[0]
        raw_transiton = self._env.step(action)
        transition = self._preprocessor(raw_transiton)
        state, reward, next_state = transition.state
        rstate = RangedState(state, tolerance)
        rstate_next = RangedState(next_state, tolerance)
        self._policy[rstate] = action
        self._state_value[rstate] = reward + self._state_value[rstate_next]


  def epspolicy(self, state):
    if self._eps > random.uniform(0, 1):
      action = random.choice(range(10))[0]
    else:
      action = self.
