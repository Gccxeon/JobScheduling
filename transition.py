from collections import namedtuple
import typing

Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "terminal"))


#class Transition(transition):
#  """
#  A class used to represent the states transition after the action is taken in
#    the environment
#  """
#
#  def __init__(self,
#               state,
#               action,
#               reward,
#               next_state):
#    """
#    Args:
#      state: a list of float number, the current state of the env;
#      action: an integer, the action taken by agent;
#      reward: a float number, the reward receieved from the env;
#      next_state: same type and requirements of state, the state return after
#        taking the acton.
#    """
#    # check the input requirements
#
#    super(Transition, self).__init__(state, action, reward, next_state)

