import os
dir = 'results/'
for exp in os.listdir(dir):
    exp_dir = os.path.join(dir,exp)
    print(os.path.join(exp_dir))