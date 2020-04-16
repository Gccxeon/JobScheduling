from collections import defaultdict
from q_state_mapper import QStateMapper
import numpy as np
import random
class QLearner():
  # A class that does the q learning
  def __init__(self,
               env,
              lr,
               df,
               state_preprocess=QStateMapper,
               epsilon_greedy=True,
               eps=0.98,
               eps_drop=0.97):
    """
    Args:
      env: the learning environment;
      lr: q learning rate;
      df: discount facotr for reward;
      epsilon_greedy: enable epsilon_greedy policy or not;
      eps: epsilon_greedy starting point;
      eps: epsilon_greedy decay at each step.
    """
    self._env = env
    self._env.reset()
    self._state_process = state_preprocess
    self._lr = lr
    self._df = df
    if epsilon_greedy:
      self._eps = 0
    else:
      self._eps = eps
    self._eps_drop = eps_drop
    self._q_values = QValues()


  def policy(self, state):
    return self.eps_policy(state)


  def eps_policy(self, state):
    if random.random() > self._eps:
      action = self._q_values[state].index(max(self._q_values[state]))
    else:
      action = random.randint(0, 10)
    self._eps = self._eps * self._eps_drop

    return action


  def learn(self, action):
    state, _, reward, next_state, terminal = self._env.step(action)
    q_state = self._state_process(state)
    if terminal:
      self._env.reset()
      return (0, None, 0, True)
    qn_state = self._state_process(next_state)
    old_qval = self._q_values.get_value(q_state, action)
    max_next_qval = max(self._q_values.get_state_values(qn_state))
    update = self._lr * (reward + self._df*max_next_qval - old_qval)
    self._q_values.set_value(q_state,
                             action,
                             old_qval + update)

    return (update, qn_state, reward, terminal)


class QValues():
  # A class that stores the q_values of each state
  def __init__(self):
    # initilize the state values as None so a random value will be returned
    # if the corresponding state hasn't been visited;
    # according to the current implementation of QStateMapper, there are
    # 2 * 100 possible states in total
    self.values = defaultdict(lambda: list(map(lambda x: random.random(),
                                               [None]*10)))

  def __getitem__(self, key):
    return self.values[key]
  def get_value(self, state, action):
    if type(state) == list:
      state = tuple(state)
    if self.values[state][action] == None:
      self.values[state][action] = random.random()
    return self.values[state][action]

  def get_state_values(self, state):
    if type(state) == list:
      state = tuple(state)
    return self.values[state]


  def set_value(self, state, action, val):
    if type(state) == list:
      state = tuple(state)
    self.values[state][action] = val

class trainer():
  # a helper object for training
  def __init__(self,
               learner,
               episodes,
               early_stop=False,
               early_stop_margin=0.0000003):
    """
    Args:
      learner: The learning agent;
      episodes: How many episodes to train by default before forcing the
          termination;
      early_stop_margin: If the avgerage udpate during an episode is smaller
          than this value, stop the training.
    """
    self._learner = learner
    self._episodes = episodes
    if early_stop:
      self._early_stop = early_stop_margin
    else:
      self._early_stop = 0

  def train(self, episodes=None, report_interval=10):
    if not episodes:
      episodes = self._episodes
    total_episodes = 0
    while episodes:
      action = random.randint(0, 9)
      total_update = 0
      total_steps = 0
      total_r = 0
      while True:
        try:
          update, nq_state, reward, terminal = self._learner.learn(action)
        except IndexError:
          print("List out of range, with action {}".format(action))

        if terminal:
          break
        total_update += abs(update)
        total_steps += 1
        total_r += reward
        action = self._learner.eps_policy(nq_state)
      episodes -= 1
      total_episodes += 1
      if episodes+1 and not(episodes % report_interval):
        print("Episode {:2d}, Average_updates: {:3f}, iterations: {:5d},"
              "reward: {:3f}".format(total_episodes,
                                     total_update / total_steps,
                                     total_steps,
                                     total_r))
      if abs(total_update / total_steps) < self._early_stop:
        print("Reached early stopping condtion, exiting...")
        break



