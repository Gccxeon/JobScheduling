import math
def reward_v1(wait_times, wait_time, exec_time, finish_time):
  difference_scale = max(wait_times) - min(wait_times) + 0.01
  reward = -(wait_time / difference_scale)*5 - math.log(difference_scale, 5) + 1 / (exec_time + 0.01)
  return reward


def reward_v2(wait_times, wait_time, exec_time, finish_time):
  # ds: difference scale
  ds = max(wait_times) - min(wait_times) + 0.01
  reward = -math.log(wait_time+0.01) - math.log(ds) - exec_time*3
  return reward

# Simple penalted reward targeting the total wait time
def reward_v3(wait_times, wait_time, exec_time, finish_time):
  return -math.log(sum(wait_times) + 0.1)

def scaled_reward(*args):
  return reward_v3(*args)
