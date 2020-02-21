import torch
from torchvision import transforms
from collections import namedtuple
from transition import Transition

ProcessedTransition=namedtuple("ProcessedTransition", ("state",
                                                       "action",
                                                       "reward",
                                                       "next_state_nt",
                                                       "nt_mask"))

def normalize(t):
  return (t - t.mean(1, keepdim=True)/(t.max(1, keepdim=True)[0] -
                                       t.min(1, keepdim=True)[0]))

def process_from_replay_sample(sample):

  if type(sample) == Transition:
    sample = [sample]
  batch_transition = Transition(*zip(*sample))
  state = batch_transition.state
  action = batch_transition.action
  reward = batch_transition.reward
  next_state = batch_transition.next_state
  terminal = batch_transition.terminal

  state = normalize(torch.cat(state))
  action = torch.cat(action).unsqueeze(1)
  reward = torch.cat(reward)
  next_state = [state for state in next_state if state is not None]
  if next_state:
    next_state_nt = normalize(torch.cat(next_state))
  else:
    next_state_nt = None
  nt_mask = list(not(term) for term in terminal)

  return ProcessedTransition(state, action, reward, next_state_nt, nt_mask)
