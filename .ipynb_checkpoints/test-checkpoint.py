from job_scheduling_env import SchedulingEnv
from transition import Transition
from replay_memory import ReplayMemory
from collections import deque
from utils import Collector
from policy import Policy
num_jobs = 10000
num_servers = 10
scheduling_speed = 35
response_time_discount = 0.99

job_params = {'job_type':['CPU', 'IO'], 'intensity':(800, 1000)}
server_params = {'cpu_power':(800, 1222), 'io_power':(500, 1500)}
env = SchedulingEnv(num_jobs, num_servers, job_params, server_params,
                         scheduling_speed, response_time_discount)
policy = Policy(env._buildin_policy().random_policy)
replay_memory = ReplayMemory(Transition, 20)

collector = Collector(env, Transition, Transition, replay_memory, 50, policy )
