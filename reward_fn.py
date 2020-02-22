import math
def scaled_reward(wait_times, wait_time, exec_time, finish_time):
  difference_scale = max(wait_times) - min(wait_times) + 0.01
  reward = -(wait_time / difference_scale)*5 - math.log(difference_scale, 5) + 1 / (exec_time + 0.01)
  return reward


