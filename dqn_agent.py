import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy

class DqnAgent():
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
               discount=0.98,
               update_tau=0.5,
               update_period=30,
               learning_rate=1e-3,
               eps_greedy=0.9,
               eps_decay_count=200,
               eps_minimum=0.1,
               clip_grad=True):

    self._env = environment
    self._action_space = environment.action_space()
    self._preprocessor = preprocessor
    self._batch_size = batch_size

    self._policy_net = network
    self._target_net = copy.deepcopy(self._policy_net)
    self._init_wieght(self._policy_net)
    self._init_wieght(self._target_net)

    self._optimizer = self._get_optim(optimizer, learning_rate)
    self._loss_fn = loss_fn
    self._discount = discount
    self._update_tau = update_tau
    self._update_period = update_period
    self._eps_greedy = eps_greedy
    self._eps_minimum = eps_minimum
    self._eps_decay =(
        math.exp( math.log(eps_minimum / eps_greedy) / eps_decay_count))
    self._clip_grad = clip_grad
    self._global_step = 0

    self._loss = None


  # Performs training on one batch if called
  def train_step(self, sample):
    """
    Args:
      sample: The processed, directly usable sample that hasattr(sample, (
          "state", "action", "reward", "next_state_nt", "nt_mask")). The
          attribute "next_state_nt" means next_state(s) that are not terminal,
          and the "nt_mask" attribute gives a indexing mask that can be used
          on torch tensor indicating non-terminal states.
    """
    state, action, reward, next_state_nt, nt_mask = self._preprocessor(sample)
    state_values = self._policy_net(state)

    # Compute the state_action values from the samples
    q_values = state_values.gather(1, action)
    target_q_values = torch.zeros(self._batch_size)
    target_state_values = self._target_net(next_state_nt).detach()
    target_q_values[nt_mask] = (
        self._discount * target_state_values.max(1)[0] + reward[nt_mask])

    loss = self._loss_fn(q_values, target_q_values.unsqueeze(1))
    self._loss = loss
    self._optimizer.zero_grad()
    loss.backward()
    if self._clip_grad == True:
      for params in self._policy_net.parameters():
       params.grad.data.clamp_(-1,1)
    self._optimizer.step()
    self._global_step += 1
    if not self._global_step % self._update_period:
      self._soft_update()


  def _init_wieght(self, net):
    def init_w(m):
      if hasattr(m, 'weight'):
        nn.init.xavier_normal_(m.weight)
    net.apply(init_w)


  def _eps_policy(self, state):
    self._eps_greedy = max(self._eps_greedy * self._eps_decay,
                           self._eps_minimum)
    if self._eps_greedy > torch.rand(1)[0]:
      return self._random_action()
    else:
      # Only one state is supported
      return self.predict_by_t(state)

  def _random_action(self):
    return torch.randint(self._action_space, (1,)).view(1,1)

  def predict(self, state):
    with torch.no_grad():
      action = self._policy_net(state).max(1)[1].view(1,1)
      return action

  def predict_by_t(self, state):
    with torch.no_grad():
      action = self._target_net(state).max(1)[1].view(1,1)
      return action

  def default_policy(self, sample):
    state = self._preprocessor(sample).state
    return self._eps_policy(state).flatten()

  # TODO: Support more optimizers
  def _get_optim(self, optimizer, lr):
    if optimizer == "Adam":
      return optim.Adam(self._policy_net.parameters(), lr=lr)
    else:
      return optim.SGD(self._policy_net.parameters(), lr=lr, momentum=0.9)

  def _soft_update(self):
    with torch.no_grad():
      for param_p, param_t in zip(self._policy_net.parameters(),
                                  self._target_net.parameters()):
        param_t = self._update_tau * param_p + (1 - self._update_tau) * param_t

  def batch_size(self):
    return self._batch_size

  def loss(self):
    return self._loss
