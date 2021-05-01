# ON-POLICY

## support algorithms

| Algorithms | recurrent-verison | mlp-version | cnn-version | share-base version | independent version |
| :--------: | :---------------: | :---------: | :---------: |:---------------: |:---------------: |
| MAPPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |:heavy_check_mark:        |:heavy_check_mark:        |
| MAPPG |        :heavy_check_mark:           |       :heavy_check_mark:      |     :heavy_check_mark:        |:heavy_check_mark:        |:heavy_check_mark:        |
| MATRPO[^1] |        :heavy_check_mark:           |       :heavy_check_mark:      |     :heavy_check_mark:        |:heavy_check_mark:        |:heavy_check_mark:        |
|            |                   |             |             | |

[^1]: see trpo branch


## support environments:
**Pay Attention:** we sometimes hack the environment code to fit our task and setting. 
- [StarCraftII](https://github.com/oxwhirl/smac)
- [Hanabi](https://github.com/deepmind/hanabi-learning-environment)
- [MPE](https://github.com/openai/multiagent-particle-envs)
- [Hide-and-Seek](https://github.com/openai/multi-agent-emergence-environments)
- [social dilemmas](https://github.com/eugenevinitsky/sequential_social_dilemma_games)
- agar.io
- [SMARTS](https://github.com/huawei-noah/SMARTS)
- [HighWay](https://github.com/eleurent/highway-env)

## TODOs:
- [ ] multi-agent FLOW



## 1. Install

### 1.1 instructions

   test on CUDA == 10.1

   

``` Bash
   conda create -n marl
   conda activate marl
   pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
   cd onpolicy
   pip install -e . 
```

### 1.2 hyperparameters

* config.py: contains all hyper-parameters

* default: use GPU, chunk-version recurrent policy and shared policy

* other important hyperparameters:
  - use_centralized_V: Centralized training (MA) or Centralized training (I)
  - use_single_network: share base or not
  - use_recurrent_policy: rnn or mlp
  - use_eval: turn on evaluation while training, if True, u need to set "n_eval_rollout_threads"

## 2. StarCraftII

### 2.1 Install StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

   

``` Bash
unzip SC2.4.10.zip
# password is iagreetotheeula
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```

*  download SMAC Maps, and move it to `~/StarCraftII/Maps/`.

*  If you want stable id, you can copy the `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.

### 2.2 Train StarCraftII

* train_smac.py: all train code

  + Here is an example:

  

``` Bash
  conda activate marl
  cd scripts
  chmod +x train_smac.sh
  ./train_smac.sh
```

  + local results are stored in fold `scripts/results`, if you want to see training curves, login wandb first, see guide [here](https://docs.wandb.com/). Sometimes GPU memory may be leaked, you need to clear it manually.

   

``` Bash
   ./clean_gpu.sh
```

### 2.3 Tips

   Sometimes StarCraftII exits abnormally, and you need to kill the program manually.

   

``` Bash
   ./clean_smac.sh
   ./clean_zombie.sh
```

### 2.4 Results

you can see our training curves via [wandb link](https://wandb.ai/zoeyuchao/StarCraft2?workspace=user-zoeyuchao)

|        Maps        | hyperparameters |                |         |     |     MAPPO    |     IPPO     | MAPPO-original |     QMIX     |   MADDPG  | MASAC |  MATD3 |
|:------------------:|:---------------:|:--------------:|:-------:|:---:|:------------:|:------------:|:--------------:|:------------:|:---------:|:-----:|:------:|
|                    |    ppo epoch    | stacked_frames | mlp/rnn |  T  |              |              |                |              |           |       |        |
|     2m\_vs\_1z     |        15       |        1       |   rnn   |  1M |    100(0)    |    100(0)    |     100(0)     |  96.06(3.8)  |    100    | 90.88 |  89.84 |
|         3m         |        15       |        1       |   rnn   |  1M |    100(0)    |    100(0)    |     100(0)     |  97.43(2.73) |   97.43   | 55.54 | 60.125 |
|     2s\_vs\_1sc    |        15       |        1       |   rnn   |  1M |    100(0)    |    100(0)    |     100(0)     |  98.96(1.8)  |    98.8   | 20.06 |    0   |
|        2s3z        |        15       |        1       |   rnn   |  2M |  99.59(0.7)  |  99.29(0.6)  |   99.46(0.6)   |  96.88(5.4)  |   96.88   | 23.52 |  3.125 |
|     3s\_vs\_3z     |        15       |        1       |   rnn   |  2M |    100(0)    |    100(0)    |     100(0)     |  97.92(1.8)  |     0     |   0   |    0   |
|     3s\_vs\_4z     |        15       |        4       |   mlp   |  5M |  98.44(2.21) |  98.61(1.6)  |   98.44(1.71)  |       \      |     \     |   \   |    \   |
|     3s\_vs\_4z     |        10       |        1       |   rnn   | 10M | 93.13(25.77) | 83.33(37.79) |  81.25(40.12)  |  97.92(3.6)  |   8.584   |   0   |    0   |
|     3s\_vs\_5z     |        15       |        4       |   mlp   | 10M |  91.02(24.4) |  97.32(3.8)  |  82.14(35.09)  |       \      |     \     |   \   |    \   |
|     3s\_vs\_5z     |        15       |        1       |   rnn   | 10M | 55.97(45.06) | 14.06(34.45) |  29.17(50.52)  | 96.88(3.125) |     0     |   0   |    0   |
|    2c\_vs\_64zg    |        5        |        1       |   rnn   | 10M |  98.96(1.8)  |  96.88(5.4)  |   98.96(1.8)   |     96.08    |  54.5(3M) |   0   |    0   |
| so\_many\_baneling |        15       |        1       |   rnn   | 10M |    100(0)    |  98.96(1.8)  |     100(0)     |  95.31(6.6)  |    100    | 62.43 |  45.28 |
|         8m         |        15       |        1       |   rnn   |  3M |    100(0)    |    100(0)    |     100(0)     |  95.83(3.6)  |   89.46   | 43.65 |    0   |
|         MMM        |        15       |        1       |   rnn   |  5M |  98.96(1.8)  |  98.96(1.8)  |   98.96(1.8)   |  98.44(2.21) |   87.44   | 16.83 |    0   |
|       1c3s5z       |        15       |        1       |   rnn   | 10M |    100(0)    |    100(0)    |   98.44(2.21)  |  95.83(1.8)  |   97.38   |   24  |    0   |
|     8m\_vs\_9m     |        15       |        1       |   rnn   | 10M |  89.73(6.68) |   87.5(6.5)  |   69.6(19.06)  |  90.63(6.25) |   24.96   |   0   |    0   |
|   bane\_vs\_bane   |        15       |        1       |   rnn   |  2M |    100(0)    |    100(0)    |     100(0)     |     91.97    |     X     |   0   |    0   |
|         25m        |        10       |        1       |   rnn   |  5M |    100(0)    |    100(0)    |     100(0)     |  96.88(5.4)  |     X     |   0   |    0   |
|     5m\_vs\_6m     |        15       |        1       |   rnn   | 10M |  64.29(20.8) | 71.88(8.268) |   76.04(6.5)   | 78.13(11.27) |    44.4   |   0   |    0   |
|     5m\_vs\_6m     |        10       |        4       |   mlp   | 10M |  83.75(7.46) | 71.25(18.68) |   70.83(14.6)  |       \      |     \     |   \   |    \   |
|        3s5z        |        5        |        1       |   rnn   | 10M |  98.96(1.8)  |  98.96(1.8)  |   98.96(1.8)   |  93.75(5.4)  |    87.5   |   0   |    0   |
|        MMM2        |        5        |        1       |   rnn   | 10M |   80(9.57)   | 81.25(17.86) |   91.25(5.27)  |  88.89(6.2)  |   23.45   |   0   |    0   |
|        MMM2        |        5        |        1       |   rnn   | 25M |  94.64(7.3)  |  94.14(3.8)  |   96.56(2.7)   |       \      |     \     |   \   |    \   |
|    10m\_vs\_11m    |        5        |        1       |   rnn   | 10M |   97.5(2.4)  |   97.5(2.4)  |  41.19(27.42)  |  95.83(7.2)  | 47.13(3M) |   0   |    0   |
|   3s5z\_vs\_3s6z   |        5        |        1       |   rnn   | 10M | 55.47(38.98) | 77.88(19.82) |  41.35(22.54)  | 68.39(22.06) |     0     |   0   |    0   |
|   3s5z\_vs\_3s6z   |        5        |        4       |   mlp   | 10M | 74.48(13.32) |  77.08(7.8)  |  20.62(18.17)  |       \      |     \     |   \   |    \   |
|    27m\_vs\_30m    |        5        |        1       |   rnn   | 10M | 96.88(3.125) |   90.63(5)   |   95.83(3.6)   |     64.06    |     X     |   0   |    0   |
|     6h\_vs\_8z     |        5        |        1       |   mlp   | 10M |  85.42(11.9) | 67.81(32.38) |  10.94(6.572)  |       \      |     \     |   \   |    \   |
|     6h\_vs\_8z     |        5        |        1       |   rnn   | 10M | 34.38(46.03) | 36.46(50.23) |  7.292(7.864)  | 21.56(23.66) |     0     |   0   |    0   |
|      corridor      |        5        |        1       |   mlp   | 10M |  99.06(2.1)  |   97.5(4.1)  |   0.97(1.88)   |       \      |     \     |   \   |    \   |
|      corridor      |        5        |        1       |   rnn   | 10M |     0(0)     |     0(0)     |      0(0)      |    75.625    |     0     |   0   |    0   |

if you want to run MADDPG/MATD3/MASAC algorithms, welcome to use this repository [offpolicy](https://github.com/marlbenchmark/offpolicy)
## 3. Hanabi

  ### 3.1 Hanabi

   The environment code is reproduced from the hanabi open-source environment, but did some minor changes to fit the algorithms. Hanabi is a game for **2-5** players, best described as a type of cooperative solitaire.

### 3.2 Install Hanabi 

   

``` Bash
   pip install cffi
   cd envs/hanabi
   mkdir build & cd build
   cmake ..
   make -j
```

### 3.3 Train Hanabi

   After 3.2, we will see a libpyhanabi.so file in the hanabi subfold, then we can train hanabi using the following code.

   

``` Bash
   conda activate onpolicy
   cd scripts
   chmod +x train_hanabi_forward.sh
   ./train_hanabi_forward.sh
```
we also have a backward version training script, which uses a different way to calculate reward of one turn.

``` Bash
   conda activate onpolicy
   cd scripts
   chmod +x train_hanabi_backward.sh
   ./train_hanabi_backward.sh
```

## 4. MPE

### 4.1 Install MPE

``` Bash
   # install this package first
   pip install seabon
```

3 Cooperative scenarios in MPE:

* simple_spread: set num_agents=3
* simple_speaker_listener: set num_agents=2, and use --share_policy
* simple_reference: set num_agents=2

### 4.2 Train MPE

   

``` Bash
   conda activate marl
   cd scripts
   chmod +x train_mpe.sh
   ./train_mpe.sh
```

## 5. Hide-And-Seek

we support multi-agent boxlocking and blueprint_construction tasks in the hide-and-seek domain.

### 5.1 Install Hide-and-Seek

#### 5.1.1 Install MuJoCo

1. Obtain a 30-day free trial on the [MuJoCo website](https://www.roboti.us/license.html) or free license if you are a student. 

2. Download the MuJoCo version 2.0 binaries for [Linux](https://www.roboti.us/download/mujoco200_linux.zip).

3. Unzip the downloaded `mujoco200_linux.zip` directory into `~/.mujoco/mujoco200`, and place your license key at `~/.mujoco/mjkey.txt`.

4. Add this to your `.bashrc` and source your `.bashrc`.

   

``` 
   export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
```

#### 5.1.2 Intsall mujoco-py and mujoco-worldgen

1. You can install mujoco-py by running `pip install mujoco-py==2.0.2.13`. If you encounter some bugs, refer this official [repo](https://github.com/openai/mujoco-py) for help.

   ```
   sudo apt-get install libgl1-mesa-dev libosmesa6-dev
   ```

   

2. To install mujoco-worldgen, follow these steps:

   

``` Bash
    # install mujuco_worldgen
    cd envs/hns/mujoco-worldgen/
    pip install -e .
    pip install xmltodict
    # if encounter enum error, excute uninstall
    pip uninstall enum34
```

### 5.2 Train Tasks

   

``` Bash
   conda activate marl
   # boxlocking task, if u want to train simplified task, need to change hyper-parameters in box_locking.py first.
   cd scripts
   chmod +x train_boxlocking.sh
   ./train_boxlocking.sh
   # blueprint_construction task
   chmod +x train_bpc.sh
   ./train_bpc.sh
   # hide and seek task
   chmod +x train_hns.sh
   ./train_hns.sh
```

## 6. Flow

### 6.1 install sumo

```Bash
cd envs/decentralized_bottlenecks/scripts

# choose the bash scripts according to your platform
./setup_sumo_ubuntu1604.sh 

# default write the PATH to ~/.bashrc, if you are using zsh, copy the PATH to ~/.zshrc
source ~/.zshrc

# check whether the sumo is installed correctly
which sumo
sumo --version
sumo-gui
```

### 6.2 install flow

```Bash
pip install lxml imutils gym-0.10.5

# check whether your flow is installed correctly
python examples/sumo/sugiyama.py
```

## 7. SMARTS


1. git clone sumo, pay attention to use sumo version < 1.8

2. `cmake ../.. & make -j` sumo and `make install` sumo, u can use `sumo` in the terminal, then u can see the version of sumo.

   [^sumo]: if u encounter TIFF error, conduct: `conda remove libtiff==4.1.0`, actually we need to use `conda install libtiff==4.0.9`.
   
3. git clone smarts and `pip install -e .`[please remove some unneeded packages in requirement.txt]

4. `scl scenario build --clean ./loop` loop is ur own scenerio.

5. all is ready , enjoy `./train_smarts.sh`

## 8. HighWay

1. training script: `./train_highway.sh`
1. rendering script `./render_highway.sh`

## 9. Docs：

```
pip install sphinx sphinxcontrib-apidoc sphinx_rtd_theme recommonmark

sphinx-quickstart
make html
```

