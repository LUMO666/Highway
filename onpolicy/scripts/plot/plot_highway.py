import pandas
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator
import scipy.signal

plt.style.use('ggplot')
#timesnewroman=FontProperties(fname='/home/tsing92/Highway/onpolicy/scripts/plot/font/times.ttf',)
#env_names = ['1','2','3']
env_name = 'merge'
seed_names = ['seed1','seed2','seed3']
method_names = ['vi','rvi','ppo', 'd3qn']
label_names = ['VI','RVI','PPO', 'D3QN']
color_names = ['limegreen', 'saddlebrown','purple','blue','red','gray']


save_dir = './adv_reward/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.figure()

for method_name, label_name, color_name in zip(method_names, label_names, color_names):
    metric_map = []
    for seed_name in seed_names:
        data_dir =  save_dir + env_name + '/'+ env_name + '_' + method_name + '_' + seed_name + '_adv_reward.csv'
        # data_dir =  save_dir + map_name + '/' + method_name + "/auc/100step.csv"

        df = pandas.read_csv(data_dir)
        
        key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
        
        key_step = [n for n in key_cols if n == 'Step']
        key_metric = [n for n in key_cols if n != 'Step']

        x_step = np.array(df[key_step]).squeeze(-1)
        x_step=x_step[0:500]
        metric = np.array(df[key_metric])
        metric=metric[0:500]

        metric_map.append(metric)

    # [map, step, seed] -- metric_map
    metric_map = np.array(metric_map)
    # first map mean
    y_seed = np.mean(metric_map, axis=2)
    #print(y_seed.shape)
    #input()
    mean_seed = np.mean(y_seed, axis=0)
    #print(mean_seed.shape)
    #input()
    std_seed = np.std(y_seed, axis=0)
    #print('std:',std_seed.shape)
    #input()
    mean_seed_smooth=scipy.signal.savgol_filter(mean_seed,11,3)
    plt.plot(x_step, mean_seed_smooth, label = label_name, color=color_name)
    plt.fill_between(x_step,
        mean_seed - std_seed,
        mean_seed + std_seed,
        color=color_name,
        alpha=0.1)

plt.tick_params(axis='both',which='major') 
final_max_step = 2000000
x_major_locator = MultipleLocator(400000)
x_minor_Locator = MultipleLocator(200000) 
y_major_locator = MultipleLocator(3)
y_minor_Locator = MultipleLocator(1.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_minor_locator(x_minor_Locator)
ax.yaxis.set_minor_locator(y_minor_Locator)
# ax.xaxis.get_major_formatter().set_powerlimits((0,1))
tx = ax.xaxis.get_offset_text() 
cha_size=17
tx.set_fontsize(cha_size) 
#ax.xaxis.grid(True, which='minor')
plt.xlim(0, final_max_step)
plt.ylim([0, 9])
plt.xticks(fontsize=cha_size)
plt.yticks(fontsize=cha_size)
plt.xlabel('Steps', fontsize=cha_size)
plt.ylabel('Hazard Arbitration Reward', fontsize=cha_size)
#plt.title('Merge Double Attackers', fontsize=cha_size)
plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=cha_size)

plt.savefig(save_dir + env_name + "_adv_reward3.png", bbox_inches="tight")
