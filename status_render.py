import matplotlib.pyplot as plt

# A helper obejct used to print the statistical/status info in the job 
# scheduling process.
class StatusRender(object):
  def __init__(self, scheduling_env):
    self._env=scheduling_env
    
  def jobs_in_servers(self):
    # print the number of jobs in each type and their intensity arranged by 
    # each server's job que
    info = self._env.get_job_info()
    cpu_jobs = info.get("CPU intensities")
    cpu_ints = info.get("CPU intensities")
    io_jobs = info.get("IO intensive jobs")
    io_ints = info.get("IO intensities")
    tas = info.get("total allocation status")
    plt.plot(cpu_jobs, io_jobs)
