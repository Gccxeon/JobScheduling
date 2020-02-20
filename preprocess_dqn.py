from transition import Transition

import torch

def process_from_replay_sample(sample):
  batch_transition = Transition(*zip(*sample))
  states = batch_transition.state
  actions = batch_transition.action
  rewards = batch_transition.reward
  next_state = batch_transition.next_state
  terminal = batch_transition.terminal

