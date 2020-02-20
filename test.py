from job_scheduling_env import SchedulingEnv
from transition import Transition
from replay_memory import ReplayMemory
from collections import deque
from utils import Collector
from policy import Policy
from postprocessor import PostProcessor
from status_render import StatusRender

import torchvision.transforms as T
import torch

num_jobs = 5000
num_servers = 10
scheduling_speed = 35
response_time_discount = 0.99
str2num_dict = {'CPU':1, 'IO': -1}
postprocessor = PostProcessor(str2num_dict)

job_params = {'job_type':['CPU', 'IO'], 'intensity':(800, 1000)}
server_params = {'cpu_power':(800, 1222), 'io_power':(500, 1500)}
env = SchedulingEnv(num_jobs, num_servers, job_params, server_params,
                         scheduling_speed, response_time_discount)

policy = Policy(env._buildin_policy().random_policy)
replay_memory = ReplayMemory(Transition, 800)

collector = Collector(env,
                      Transition,
                      postprocessor,
                      Transition,
                      replay_memory,
                      50, policy)


# Convert to torch tensor test
def get_some_tensor_states(batch_size):
  batch_sample = Transition(*zip(*replay_memory.sample(batch_size)))
  bt_sample = batch_sample.state
  return bt_sample


def recollect(num):
  collector.collect(num)

def gather_test(dim, index):
  sample = get_some_tensor_states(10)
  return sample.gather(dim, index)

# status render test
render = StatusRender(env)
def render_test():
  return render.job_status()
