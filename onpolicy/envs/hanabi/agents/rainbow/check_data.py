from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os
import numpy as np
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator

plt.style.use('ggplot')
FILE_PREFIX = 'log'
ITERATION_PREFIX = 'iter'
log_dir_list = [
'results/small51/logs',
'results/small40/logs',
'results/small24/logs',
'results/small22/logs',
'results/small33/logs',
'results/small32/logs',
'results/small23/logs',
'results/small42/logs',
'results/small20/logs',
'results/small500/logs',
'results/small52/logs',
'results/small50/logs',
'results/small21/logs',
'results/small5000/logs',
'results/small30/logs',
'results/small31/logs',
'results/small41/logs',
'results/small34/logs',
'results/small53/logs',
'results/small43/logs'
]

def summarize_data(data, summary_keys):
  """Processes log data into a per-iteration summary.
  Args:
    data: Dictionary loaded by load_statistics describing the data. This
      dictionary has keys iteration_0, iteration_1, ... describing per-iteration
      data.
    summary_keys: List of per-iteration data to be summarized.
  Example:
    data = load_statistics(...)
    summarize_data(data, ['train_episode_returns',
        'eval_episode_returns'])
  Returns:
    A dictionary mapping each key in returns_keys to a per-iteration summary.
  """
  summary = {}
  latest_iteration_number = len(data.keys())
  current_value = None

  for key in summary_keys:
    summary[key] = []
    # Compute per-iteration average of the given key.
    for i in range(latest_iteration_number):
      iter_key = '{}{}'.format(ITERATION_PREFIX, i)
      # We allow reporting the same value multiple times when data is missing.
      # If there is no data for this iteration, use the previous'.
      if iter_key in data:
        current_value = np.mean(data[iter_key][key])
      summary[key].append(current_value)

  return summary

for log_dir in log_dir_list:
  exp_name =  log_dir.split('/')[1]
  BASE_PATH = '/home/yuchao/project/mappo-sc/envs/hanabi/agents/rainbow/'+log_dir
  raw_data, _ = colab_utils.load_statistics(BASE_PATH, verbose=True)
  summarized_data = summarize_data(raw_data, ['average_return'])
  y = summarized_data['average_return']
  x = [i*10000 for i in range(len(y))]
  last_y = y[-11:-1]
  print(log_dir,sum(last_y)/len(last_y),len(y))
  plt.cla()
  plt.plot(x,y,label='Score')

  plt.tick_params(axis='both',which='major') 

  x_major_locator=MultipleLocator(10000000)
  x_minor_Locator = MultipleLocator(1000000)
  ax=plt.gca()
  ax.xaxis.set_major_locator(x_major_locator)
  ax.xaxis.set_minor_locator(x_minor_Locator)
  ax.xaxis.get_major_formatter().set_powerlimits((0,1))
  plt.xlabel('timesteps')
  plt.ylabel('Score')
  plt.legend(loc='best', numpoints=1, fancybox=True)
  plt.title('Rainbow training - {}'.format(exp_name))
  plt.savefig('./png/{}.png'.format(exp_name))
#plt.ioff()
#plt.show()