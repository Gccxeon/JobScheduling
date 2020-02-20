import copy
import torch
import torch.nn as nn
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
               loss_fn,
               optimizer,
               replay_buffer,
               epoch,
               iteration,
               discount=0.9,
               update_tau=0.5,
               update_period=30,
               learning_rate=1e-3,
               eps_greedy=0.9,
               eps_decay=200):

    self._env = environment
    self._preprocessor = preprocessor

    self._policy_net = network
    self._target_net = copy.deepcopy(self._policy_net)
    self._init_wieght(self._policy_net)
    self._init_wieght(self._target_net)

    self._optimizer = optimizer
    self._loss_fn = loss_fn
    self._replay_buffer = replay_buffer
    self._epoch = epoch
    self._iteration = iteration
    self._discount = discount
    self._update_tau = update_tau
    self._update_period = update_period
    self._eps_greedy = eps_greedy
    self._eps_decay = eps_decay
    self._lr = learning_rate



  def unpack(self, transition):
    state, action, reward, next_state = transition
    return state, action, reward, next_state

  def get_input_from_sample(self):
    state, action, reward, next_state, terminal = (
        self._replay_buffer.sample(batch_size))
    state_p = self._preprocessor('state', state)
    reward = self._preprocessor('reward', reward)
    next_state_p =self._preprocessor('state',next_state)

    # Check if the any of the states are in the terminal state

    return (state_p, action, reward, next_state_p, terminal)



  def train(self)
    """
    Args:
      iteration: The training cicle
    """
    for i in range(self._epoch):
      for i in range(self._iteration):
        state, action, reward, next_step = self.get_input()
        q_policy = self._net(state)
        _, actions = torch.max(q_policy, axis=1)
        # Here some work needs to be done in order to support batch training

        q_difference = q_policy - q_policy_next
        loss = self._loss_fn(q_difference)
        # TODO determine what kind operation should be workded on

  def _init_wieght(self, net):
    def init_w(m):
      if hasattr(m, 'weight'):
        nn.init.xavier_normal_(m.weight)
    net.apply(init_w)


  def _eps_policy(self, state):
    if self._eps_greedy < torch.rand(1)[0]:
      return self._random_action()
    else:
      return self.predict(state)

  def _random_action(self):
    return torch.randint(0, self._action_space, (self._batch_size,))

  def predict(self, state):
    logits = self._policy_net.forward(state)
    _, actions = torch.max(logits, axis=1)
    return actions


