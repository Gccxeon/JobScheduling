import numpy
import numbers
import random
from collections import namedtuple

class Job(object):
  # A job class that denotes the job processed by the machine/server
  def __init__(self, job_type, intensity):
    if job_type not in ['CPU', 'IO']:
      raise ValueError("Undefined type of job, "
                       "only 'CPU' or 'IO' are allowed")
    """
      Args:
        job_type: 'CPU' or 'IO', either it's CPU intensive or IO intensive;
        intensity: the computational requirement of the job, for example, 
            the average MIPS needed to accomplish the job.
    """
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
      if server_times[sid] =< server_times[bestfit]:
        bestfit = sid
    if servers[bestfit].get_type() != self._type:
      bestfit = min(server_times)
    return bestfit
  
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