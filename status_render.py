from plotly.offline import plot, iplot
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio

class StatusRender(object):
# A helper obejct used to print the statistical/status info in the job
# scheduling process.

  def __init__(self, scheduling_env):

    self._env = scheduling_env
    self._title_style = {'y':0.9,
                         'x':0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'}
    self._sids = scheduling_env.get_server_ids()
    self._font = dict(family="Courier New, monospace",
                      size=14,
                      color="#f00000")

    self._x_style = {'tickmode': 'linear'}
    self._value_update()
    self._make_job_fig()
    self._make_avgres_fig()


  def _get_title(self, title_text):
    title = self._title_style
    title['text'] = title_text
    return title

  def _get_x_style(self, title_text):
    x_s = self._x_style
    x_s['title'] = title_text
    return x_s

  def _make_job_fig(self):

    title = "Type info and related intensities in the servers"

    job_fig = make_subplots(rows=1, cols=2)

    job_fig.add_trace(go.Bar(x=self._sids, y=self._cpu_jobs,
                             name="CPU intensive jobs"),
                  row=1, col=1)
    job_fig.add_trace(go.Bar(x=self._sids, y=self._io_jobs,
                             name="IO intensive jobs"),
                  row=1, col=1)
    job_fig.add_trace(go.Bar(x=self._sids, y=self._cpu_ints,
                             name="CPU intensities"),
                  row=1, col=2)
    job_fig.add_trace(go.Bar(x=self._sids, y=self._io_ints,
                             name="IO intensities"),
                  row=1, col=2)
    job_fig.update_layout(title=self._get_title(title),
                          font=self._font)
    job_fig.update_xaxes(title="Server ID", tickmode='linear',
                         row=1, col=1)
    job_fig.update_xaxes(title="Server ID", tickmode='linear',
                         row=1, col=2)
    self._job_figure = go.FigureWidget(job_fig)
    self._bar_job_cij = self._job_figure.data[0]
    self._bar_job_iij = self._job_figure.data[1]
    self._bar_job_ci = self._job_figure.data[2]
    self._bar_job_ii = self._job_figure.data[3]

  def _make_avgres_fig(self):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=self._sids,
                         y=self._avg_resp,
                         width=0.3,
                         name="Average response time")
                  )
    fig.update_layout(title=self._get_title("Average reponse time(ms)"),
                      xaxis=self._get_x_style("Server ID"),
                      font=self._font)
    self._avgres_fig = go.FigureWidget(fig)

    self._bar_avgres = self._avgres_fig.data[0]



  def job_status(self):
    return self._job_figure


  def averge_response(self):
    return self._avgres_fig

  def jobs_in_servers(self):
    # print the number of jobs in each type and their intensity arranged by
    # each server's job queue
    self._value_update()
    self._job_figure.show()


  def average_response_time(self):
    self._value_update()
    self._avgres_fig.show()

  def _value_update(self):
    info = self._env.get_job_info()
    self._cpu_jobs = info.get("CPU intensive jobs")
    self._cpu_ints = info.get("CPU intensities")
    self._io_jobs = info.get("IO intensive jobs")
    self._io_ints = info.get("IO intensities")
    self._tas = info.get("total allocation status")
    self._avg_resp = self._env.average_response_time()

  def refresh_job_fig(self):
    self._value_update()
    self._bar_job_cij.y = self._cpu_jobs
    self._bar_job_iij.y = self._io_jobs
    self._bar_job_ci.y = self._cpu_ints
    self._bar_job_ii.y= self._io_ints

  def refresh_avgres_fig(self):
    self._value_update()
    self._bar_avgres.y = self._avg_resp

  def test(self):
    fig = {
    "data": [{"type": "bar",
              "x": [1, 2, 3],
              "y": [1, 3, 2]}],
    "layout": {"title": {"text": "A test graph"}}
    }
    pio.show(fig)
