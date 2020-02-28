# A job scheduling environment for allocate the jobs to machines
import numpy as np
import math
import numbers
import random
import functools, operator
from collections import namedtuple
from collections import deque
from collections import Counter

import reward_fn

class Job(object):
  # A job class that denotes the job processed by the machine/server
  def __init__(self,job_type, intensity):
    """
      Args:
        job_type: 'CPU' or 'IO', either it's CPU intensive or IO intensive;
       intensity: the computational requirement of the job, for example,
            the average MIPS needed to accomplish the job.
    """
    if job_type not in ['CPU', 'IO']:
      raise ValueError("Undefined type of job, "
                       "only 'CPU' or 'IO' are allowed")
    self.__name__ = 'Job'
    self._type = job_type
    self._intensity = intensity

  def get_type(self):
    return self._type

  def get_intensity(self):
    return self._intensity

  def info(self):
    info = {'type': self._type, 'intensity': self._intensity}
    return info

  def server_time_estimate(self, servers):
    if type(servers) == Server:
      servers = [servers]
    elif type(servers) != list:
      raise TypeError("Only a single Server object or a list of server"
                      " objects are allowed!")
    server_times = []
    for server in servers:
      if self._type == 'CPU':
        exec_time = self._intensity / server.get_cpu_power()
      else:
        exec_time = self._intensity / server.get_io_power()
      server_times.append(exec_time)
    return server_times

  def bestfit(self, servers):
    server_times = self.server_time_estimate(servers)
    bestfit = 0
    while bestfit < len(servers):
      if servers[bestfit].get_type() == self._type:
        break
      else:
        bestfit += 1
    while bestfit < len(servers):
      if server_times[sid] <= server_times[bestfit]:
        bestfit = sid
    if servers[bestfit].get_type() != self._type:
      bestfit = min(server_times)
    return bestfit

class Generator(object):
  # the Generator object is used to generate a given number of objects, the
  # object being generated must have a __name__ attribute.
  def __init__(self, object_prototype, init_params_range):
    """
      Args:
        object_prototype: a prototype of the object to be generated;
        number_of_objects: the number of objects to generate;
        init_params_range: a dict contains the range of every param used to
            initialize the object, the key should be the name of the paramater,
            the value of each should either be a list (for non-numerical type)
            or a tuple (the lower upper bound of the numerical type).


    """
    self._ob_proto = object_prototype
    self._params_range = init_params_range

  def generate(self, num_to_generate, method='random'):
    # 'method' argument only works on the numerical objects
    objects = []
    for i in range(num_to_generate):
      init_params = self.build_params(method=method)
      objects.append( self._ob_proto(**init_params) )
    return objects

  def build_params(self, method='random'):
    init_params = {}
    for param_name, param_range in self._params_range.items():
      if type(param_range) == list:
        param = random.sample(param_range, 1)[0]
      elif type(param_range) == tuple:
        low, upp = param_range
        param = random.uniform(low, upp)
      else:
        raise TypeError("The param range: {} is neither list nor tuple!"
                        .format(param_range))
      init_params[param_name] = param
    return init_params

class Server(object):
  # A server object that contains the info of a server used to
  # process the give jobs
  def __init__(self, cpu_power, io_power):
    """
      Args:
        cpu_power: the computationa power of the server while it's working
            on the cpu intensive tasks. In here, it's the MIPS per second
            that the server can excute.
        io_power: the computational power of the server when working with
            the io intensive tasks.
    """
    self.__name__ = 'Server'
    self._cpu_power = cpu_power
    self._io_power = io_power
    if cpu_power >= io_power:
      self._type = 'CPU'
    else:
      self._type = 'IO'

  def get_type(self):
    return self._type

  def get_cpu_power(self):
    return self._cpu_power

  def get_io_power(self):
    return self._io_power

  def info(self):
    info = {'type': self._type,
            'cpu_power': self._cpu_power,
            'io_power': self._io_power}
    return info

  def job_time_estimate(self, jobs):
    # The argument jobs should either be a single job or a list of jobs.
    # This will return the total excution time of the give jobs
    if type(jobs) == Job:
      jobs = [jobs]
    if type(jobs) == list:
      if type(jobs[0]) != Job:
        raise TypeError("The input jobs are not valid!")
    total_cpu_intenstity = 0
    total_io_intensity = 0
    for job in jobs:
      if job.get_type() == 'CPU':
        total_cpu_intenstity += job.get_intensity()
      else:
        total_io_intensity += job.get_intensity()
    total_exec_time = (total_cpu_intenstity / self._cpu_power +
                       total_io_intensity / self._io_power)
    return total_exec_time


class SchedulingEnv(object):
# This is the training environment of the job scheduling task
  def __init__(self,
               num_jobs,
               num_servers,
               job_gen_params,
               server_gen_params,
               scheduing_speed,
               response_time_discount,
               init_template=None,
               init_from_template=False):
    """
    Args:
      num_jobs: the number of initial jobs to generate;
      num_servers: the number of servers to generate;
      job_gen_params: The parameters used to generate the jobs;
      server_gen_params: The parameters used to generate the servers;
      scheduing_speed: Choose how many jobs will be sent to the servers in any
          unit time(sec);
      init_template: If you don't want the jobs and servers to be generated
          refersh, provide the ready dict-array of jobs and servers;
      init_from_template: if True, the jobs and servers will initialize from
          init_tempalte.
    """
    self._clock = 0.
    self._num_jobs = num_jobs
    self._job_capacity = num_jobs
    self._num_servers = num_servers
    self._job_generator = Generator(Job, job_gen_params)
    self._server_generator = Generator(Server, server_gen_params)
    self._init_template = None
    self._scheduling_speed = scheduing_speed
    # this discount factor will be used in sensible policy
    self._response_time_discount = response_time_discount
    # an array that records which job has been allocated to which server
    self._scheduling_recorder = [0] * num_servers
    # number of cpu/io intensive jobs in each server
    self._num_cpu_jobs = [0] * num_servers
    self._num_io_jobs = [0] * num_servers
    # the total computational intensities of cpu/io jobs in each server
    self._total_cpu_intensity = [0] * num_servers
    self._total_io_intensity = [0] * num_servers
    # some build-in policies that are used as baseline performance estimation
    self._policies = ['random', 'bestfit', 'earlist', 'round_robin', 'sensible']

    if init_template:
      if (len(init_template["job"]) == num_jobs and
          len(init_template["server"]) == num_servers ):
        if init_from_template:
          self._init_template = init_template
          self._jobs = init_template["job"]
          self._servers = init_template["server"]
    if self._init_template == None:
      print("The initialization template is either None or invalid"
            ", the servers and jobs will be generated randomly")
      self._jobs = self._job_generator.generate(num_jobs)
      self._servers = self._server_generator.generate(num_servers)

    self._num_finished_jobs = 0
    # number of finished jobs in each server's queue
    self._nfj_in_servers = [0] * num_servers
    # the indicator of the running time(unit: ms)
    self._clock = 0.
    # initialize the states of the servers
    self.init_server_status()
    # backup the current jobs and servers as they will be used in reset() func
    self._template = {"job": self._jobs, "server": self._servers}
    # wrap jobs as deque array
    self._jobs = deque(self._jobs, num_jobs)
    # the indicator of if the current step of env is the final step
    self._terminal = False
    # the cummulative reward from the start of scheduing
    self._cum_reward = 0.
    self._num_actions = num_servers

  def action_space(self):
    return self._num_actions

  def reset(self):
    self.__init__(self._num_jobs, self._num_servers, None, None,
                  self._scheduling_speed, self._response_time_discount,
                  init_template=self._template, init_from_template=True)

  def get_num_jobs(self):
    return len(self._jobs)

  def get_current_job(self):
    if self._jobs:
      return self._jobs[0]
    else:
      print("There is no unscheduled job left")
      return None

  def pop_current_job(self):
    return self._jobs.popleft()

  # return the execution/response time of the given on each server
  def job_exec_times(self, job):
    return job.server_time_estimate(self._servers)

  def init_server_status(self):
    # for each server, generate a status namedtuple containing the as:
    #     dict("job_que", "expected_idle_time").
    # the sid i is assgined as i in range(num_servers)
    # the returned object is a orderdict of all the namedtuples.
    servers_status = []
    for sid, server in enumerate(self._servers):
      servers_status.append({"job_info_que": deque([], self._job_capacity),
                             "expected_idle_time": self._clock}
                           )
      self._servers_status = servers_status
    return servers_status

  def _get_job_type_num_from_status(self, status):
    num_cpu_type = 0
    num_io_type = 0
    for job_info in status["job_info_que"]:
      if job_info["job"].get_type() == 'CPU':
        num_cpu_type += 1
      else:
        num_io_type +=1
    return (num_cpu_type, num_io_type)

  def get_server_status(self):
    status_reports = []
    for sid, status in enumerate(self._servers_status):
      (num_cpu_type, num_io_type) = self._get_job_type_num_from_status(status)
      status_reports.append(StatusReport(sid,
                                         self._servers[sid].get_type(),
                                         self._servers[sid].get_cpu_power(),
                                         self._servers[sid].get_io_power(),
                                         len(status["job_info_que"]),
                                         status["expected_idle_time"],
                                         num_cpu_type, num_io_type))
    return status_reports

  def get_overview(self, time_span):
    total_queuing_jobs = (self._num_jobs - len(self._jobs) -
                          self._num_finished_jobs)
    ind_avg_res = []
    individual_avg_res_times = self.average_response_time(time_span)
    for i in range(self._num_servers):
      ind_avg_res.append({i: individual_avg_res_times[i]})
    total_response_time = sum(individual_avg_res_times)
    avg_response_time = total_response_time / self._num_servers
    return {"Total jobs in server queue": total_queuing_jobs,
            "Average response time of each server": ind_avg_res,
            "Average response time accross all servers": avg_response_time}

  def expected_wait_times(self):
    # also the expected response time
    wait_times = []
    for sid, status in enumerate(self._servers_status):
      wait_times.append(max(status["expected_idle_time"] - self._clock, 0))
    return wait_times

  def expected_finish_times(self, job):
    finish_times = []
    exec_times = self.job_exec_times(job)
    wait_times = self.expected_wait_times()
    for sid, (exec_time, wait_time) in enumerate(zip(exec_times,
                                                           wait_times)):
      finish_times.append(exec_time + wait_time + self._clock)
    return finish_times

  def get_job_info(self):
    info = {"CPU intensive jobs": self._num_cpu_jobs,
            "CPU intensities": self._total_cpu_intensity,
            "IO intensive jobs": self._num_io_jobs,
            "IO intensities": self._total_io_intensity,
            "total allocation status": self._scheduling_recorder}
    return info

  def _num_queued_jobs(self, sid):
    return len(self._servers_status[sid]["job_info_que"])

  def _scheduling_logger(self, sid, job):
    if job.get_type() == 'CPU':
      self._total_cpu_intensity[sid] += job.get_intensity()
      self._num_cpu_jobs[sid] += 1
    else:
      self._num_io_jobs[sid] += 1
      self._total_io_intensity[sid] += job.get_intensity()
    self._scheduling_recorder[sid] += 1

  def current_job(self):
    return self._jobs[0] if self._jobs else None

  def allocate_job_to(self, sid):
    current_job = self._jobs.popleft()

    # Check if in terminal state
    next_job = self.current_job()
    if next_job is None:
      self._terminal = True
      reward = 0.0
      return reward
    job_type = current_job.get_type()
    self._scheduling_logger(sid, current_job)
    status = self._servers_status[sid]
    wait_times = self.expected_wait_times()
    wait_time = wait_times[sid]

    if self._num_queued_jobs(sid) < 1:
      discounted_wait_time = wait_time
      cum_response_time = wait_time
    else:
      last_dwt = status["job_info_que"][-1]["cum_discounted_response_time"]
      last_crt = status["job_info_que"][-1]["cum_response_time"]
      discounted_wait_time = wait_time + (self._response_time_discount * last_dwt)
      cum_response_time = last_crt + wait_time

    exec_time = self.job_exec_times(current_job)[sid]
    finish_time = self.expected_finish_times(current_job)[sid]

    self._servers_status[sid]["job_info_que"].append(
        {"job": current_job, "response_time": wait_time,
         "cum_response_time": cum_response_time,
         "cum_discounted_response_time": discounted_wait_time,
         "finish_time": finish_time})
    last_idle_time = self._servers_status[sid]["expected_idle_time"]

    if last_idle_time < self._clock:
      last_idle_time = self._clock
    self._servers_status[sid]["expected_idle_time"] = (
        last_idle_time + exec_time)

    if (finish_time - self._servers_status[sid]["expected_idle_time"]
            > 0.00000001):
      raise ValueError("Mismatching finish_time and expected_idle_time at "
                       "server: {}, expected_idle_time: {}, finish_time: {}"
                       .format(sid,
                               self._servers_status[sid]["expected_idle_time"],
                               finish_time))
#    reward = self._reward_fn(wait_time, exec_time, finish_time)
    reward = reward_fn.scaled_reward(
        wait_times, wait_time, exec_time, finish_time)
    self._cum_reward += reward

    return reward

  def step(self, action):

    terminal = False
    current_job = self.current_job()

    # Handle the impossible situation
    if current_job is None:
      terminal= True
      return (None, None, None, None, True)

    state = current_job.info()
    state["response_time"] = self.expected_wait_times()
    reward = self.allocate_job_to(action)
    next_job = self.current_job()

    # Handle the terminal situation
    if next_job:
      next_state = next_job.info()
      next_state["response_time"] = self.expected_wait_times()
    else:
      next_state = None
      terminal = True
    # it tooks 1 sec to receive the jobs numbered as self._scheduling_speed
    self.simulate_time_past(1/self._scheduling_speed)

    return (state, action, reward, next_state, terminal)


  def simulate_time_past(self, time_span):
    self._clock += time_span
    self._server_status_updater()

  def get_server_info(self, sid):
    return self._servers[sid].info()

  def _get_server_origin(self, sid):
    return self._servers[sid]

  def _server_status_updater(self):
    for sid, status in enumerate(self._servers_status):
      # like ETA, etf means expected finish time.
      for job_info in list(status["job_info_que"])[self._nfj_in_servers[sid]:]:
        if job_info["finish_time"] < self._clock:
          # self._servers_status[sid]["job_info_que"].popleft()
          self._nfj_in_servers[sid] += 1
          self._num_finished_jobs += 1
        else:
          break

  def get_server_ids(self):
    return list(range(self._num_servers))

  def average_response_time(self, time_span=None):
    if time_span:
      return self.average_response_time_with_span(time_span)
    else:
      return self.average_response_time_total()

  def average_response_time_total(self):
    avg_response_t = []
    for sid in range(self._num_servers):
      avg_response_t.append(self.get_cum_response_time(sid))
    return avg_response_t

  def average_response_time_with_span(self, time_span):
    watching_jobs_init = int(time_span * self._scheduling_speed)
    avg_response_t = []
    for sid, status in enumerate(self._servers_status):
      watching_jobs = watching_jobs_init
      num_queued_jobs = self._num_queued_jobs(sid)
      if num_queued_jobs < watching_jobs:
        watching_jobs = num_queued_jobs
      if watching_jobs == 0:
        avg_response_t.append(0)
      else:
        job_info_queue = list(status["job_info_que"])[-watching_jobs:]
        total_res_t = functools.reduce(
            operator.add, map(lambda x: x["response_time"], job_info_queue))
        avg_response_t.append(total_res_t / watching_jobs)
    return avg_response_t

  def get_cum_response_time(self, sid_or_queue):
    get = lambda x : x[-1]["cum_response_time"] if x else 0
    if isinstance(sid_or_queue, int):
      queue = self.get_server_info_queue(sid_or_queue)
    elif isinstance(sid_or_queue, deque):
      queue = sid_or_queue
    else: raise TypeError("The input {} is not supported! "
                          "Must be an int or a deque".format(sid_or_queue))
    return get(queue)

  def get_server_info_queue(self, sid):
    return self._servers_status[sid]["job_info_que"]

  def average_discounted_response(self, time_span):
    watching_jobs_init = int(time_span * self._scheduling_speed)
    avg_dc_t = []
    for sid, status in enumerate(self._servers_status):
      watching_jobs = watching_jobs_init
      num_queued_jobs = self._num_queued_jobs(sid)
      if num_queued_jobs < watching_jobs:
        watching_jobs = num_queued_jobs
      if watching_jobs == 0:
        avg_dc_t.append(0)
      else:
        avg_dc_t.append( list(status["job_info_que"])
                         [-1]["cum_discounted_response_time"]
                         / watching_jobs)
    return avg_dc_t

  def allocate_simulate(self,
                        jobin_speed,
                        num_jobs,
                        time_span=None,
                        method='random'):
    # simulate the job allocation according to the given policy
    if num_jobs > len(self._jobs):
      raise ValueError("Currently only {} left, "
                       "but required {} jobs to finish them simulation"
                       .format(len(self._jobs), num_jobs))
    original_jobin_speed = self._scheduling_speed
    time_past_each = 1 / jobin_speed
    self._scheduling_speed = jobin_speed
    policy_gen = self._buildin_policy(time_span)
    if method == 'random':
      policy = policy_gen.random_policy
    elif method == 'bestfit':
      policy = policy_gen.bestfit_policy
    elif method == 'earlist':
      policy = policy_gen.earlist_policy
    elif method == 'round_robin':
      policy = policy_gen.round_robin_policy
    elif method == 'sensible':
      policy = policy_gen.sensible_policy
    else:
      raise ValueError("The selected policy: {} is not supported, "
                       "please choose in {}".format(method, self._policies))
    for i in range(num_jobs):
      sid = policy()
      self.allocate_job_to(sid)
      self.simulate_time_past(time_past_each)

  def _reward_fn(self, wait_time, exec_time, finish_time):
    return exec_time / (wait_time + 0.00001)

  def _buildin_policy(self, time_span=5):
    return BuildInPolicy(self, observation_span=time_span)

  def scheduling_report(self):
    report = []
    num_total_actions = sum(self._scheduling_recorder)
    for sid, num_actions in enumerate(self._scheduling_recorder):
      report.append({"Server id": sid,
                     "Choosed ratio": num_actions / num_total_actions})
    return report

  def is_terminal(self):
    return self._terminal

  def cum_reward(self):
    return self._cum_reward

class BuildInPolicy():
  def __init__(self,
               scheduling_env,
               observation_span=5):
    self._scheduling_env = scheduling_env
    self._action_space = scheduling_env._num_servers
    self._action_counter = -1
    self._observation_span = observation_span

  def random_policy(self):
    return random.sample(range(self._action_space), 1)[0]

  def bestfit_policy(self):
    job = self._scheduling_env.get_current_job()
    job_type = job.get_type()
    wait_times = self._scheduling_env.expected_wait_times()
    servers = self._scheduling_env._servers
    num_servers = len(servers)
    bestfit = 0
    while bestfit < num_servers:
      if servers[bestfit].get_type() == job_type:
        break
      else:
        bestfit += 1
    lastest_fit = bestfit + 1
    while lastest_fit < num_servers:
      if servers[lastest_fit].get_type() == job_type:
        if wait_times[lastest_fit] < wait_times[bestfit]:
          bestfit = lastest_fit
      lastest_fit += 1
    if bestfit <= num_servers - 1:
      if servers[num_servers - 1 ].get_type() != job_type:
        bestfit = wait_times.index(min(wait_times))
    return bestfit

  def round_robin_policy(self):
    self._action_counter += 1
    self._action_counter %= self._action_space
    return self._action_counter

  def earlist_policy(self):
    job = self._scheduling_env.get_current_job()
    wait_times = self._scheduling_env.expected_wait_times()
    return wait_times.index(min(wait_times))

  def sensible_policy(self):
    avg_response_time = (
        self._scheduling_env.average_discounted_response(self._observation_span))
    offset = max(avg_response_time)
    logits = -np.array(avg_response_time) + offset + 10
    probs = logits / sum(logits)
    action = np.random.choice(a=range(self._action_space), p=probs)
    return action

class StatusReport():
  def __init__(self, sid, server_type, cpu_power, io_power, queue_len, e_i_t,
               num_cpu_job, num_io_job):
    self.status = {"Server id": sid, "Type": server_type,
                   "CPU power": cpu_power, "IO power": io_power,
                   "Job queue length": queue_len, "Expected idle time": e_i_t,
                   "CPU intensive jobs": num_cpu_job,
                   "IO intensive jobs": num_io_job}

  def get_status(self):
    return self.status


class JobQueue(deque):
  # A child class of deque that support more actions specially tailered for
  # storing the jobs.

  def __init__(self, iterable, maxlen):
    super(JobQueue, self).__init__(iterable, maxlen)
  def __getitem__(self, key):
    if isinstance(key, slice):
      return [self[i] for i in range(*k.indices(len(self)))]
    elif isinstance(key, int):
      return self[key]

class SimpleEmulator(object):
  pass
  # simple emulator for estimate the total
