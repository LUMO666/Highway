import numpy as np
import os
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator

import matplotlib.pyplot as plt
plt.style.use('ggplot')
FILE_PREFIX = 'log'
ITERATION_PREFIX = 'iter'

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

BASE_PATH = '/home/yuchao/project/mappo-sc/envs/hanabi/agents/rainbow/results/check_small2/logs'  # @param
game="Hanabi-Small2"
# Use our provided colab utils to load this log file. The second returned 
raw_data, _ = colab_utils.load_statistics(BASE_PATH, verbose=True)

summarized_data = summarize_data(
    raw_data, ['average_return'])
y = summarized_data['average_return']
x = [i*10000 for i in range(len(y))]

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
plt.title('Rainbow training - {}'.format(game))
plt.savefig('./results/{}.png'.format(game))
