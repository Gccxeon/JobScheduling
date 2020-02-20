import torch
from collections import namedtuple
from transition import Transition

ProcessedTransition=namedtuple("ProcessedTransition", ("state",
                                                       "action",
                                                       "reward",
                                                       "next_state_nt",
                                                       "nt_mask"))


def process_from_replay_sample(sample):

  if type(sample) == Transition:
    sample = [sample]
  batch_transition = Transition(*zip(*sample))
  state = batch_transition.state
  action = batch_transition.action
  reward = batch_transition.reward
  next_state = batch_transition.next_state
  terminal = batch_transition.terminal


  state = torch.cat(state)
  action = torch.cat(action).unsqueeze(1)
  reward = torch.cat(reward)
  next_state_nt = torch.cat([state for state in next_state if state is not None])
  nt_mask = list(not(term) for term in terminal)

  return ProcessedTransition(state, action, reward, next_state_nt, nt_mask)
