# Author: Boyuan Chen (Berkeley Artifical Intelligence Research)
#         Zhenggang Tang (Peking University)

# The project is largely based on m-byte918's javascript implementation of the game with a lot of bug fixes and optimization for python
# Original Ogar-Edited project https://github.com/m-byte918/MultiOgar-Edited
import gym
from gym import spaces
from onpolicy.envs.agar.GameServer import GameServer
from onpolicy.envs.agar.players import Player, Bot
import numpy as np
from copy import deepcopy


def max(a, b):
    if a > b:
        return a
    return b


def rand(a, b):
    return np.random.random() * (b - a) + a


def onehot(d, ndim):
    v = [0. for i in range(ndim)]
    v[d] = 1.
    return v


class AgarEnv(gym.Env):
    def __init__(self, args, obs_size=531, gamemode=0, kill_reward_eps=0, coop_eps=1, reward_settings="std"):
        super(AgarEnv, self).__init__()
        self.args = args
        self.action_repeat = args.action_repeat
        self.g = args.gamma  # discount rate of RL (gamma)
        self.gamemode = gamemode  # We only implemented FFA (gamemode = 0)
        self.reward_settings = reward_settings
        self.total_step = 0
        # total_step > up_step,up = [0,1] total_step = 10e6, up = 1
        # total_step > low_step, low = [0,1] total_step = 15e6, low=1
        self.up_step = 5e6
        self.low_step = 15e6
        self.curriculum_learning = args.use_curriculum_learning
        self.num_agents = args.num_agents
        self.num_bots = 5

        #self.observation_space = spaces.Dict( {'agent-'+str(i):spaces.Box(low=-100, high=100, shape=(obs_size,)) for i in range(self.num_agents)}  )
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.num_agents):
            self.action_space.append(spaces.Tuple(
                [spaces.Box(low=-1, high=1, shape=(2,)), spaces.Discrete(2)]))
            self.observation_space.append(
                [obs_size, [10, 15], [5, 7], [5, 5], [10, 15], [10, 15], [1, 21]])
            self.share_observation_space.append([obs_size * self.num_agents, [self.num_agents, obs_size]])

        self.viewer = None

        self.last_mass = [None for i in range(self.num_agents)]
        self.sum_r = np.zeros((self.num_agents, ))  # summation or reward
        # summation of discounted reward
        self.sum_r_g = np.zeros((self.num_agents, ))
        # summation of discounted reward using standard reward settings (alpha = 1, beta = 0)
        self.sum_r_g_i = np.zeros((self.num_agents, ))
        self.dir = []

        self.mass_reward_eps = 0.33
        self.killed_reward_eps = 0.
        self.kill_reward_eps = np.ones(
            self.num_agents) * 0.33 * kill_reward_eps
        self.coop_eps = np.ones(self.num_agents) * coop_eps
        
        # if self.share_reward:
        #self.coop_eps = np.zeros(self.num_agents)

    def step(self, actions_):
        actions = deepcopy(actions_)
        reward = np.zeros((self.num_agents, ))
        done = np.zeros((self.num_agents, ))
        info = [{} for i in range(self.num_agents)]

        first = True
        for i in range(self.action_repeat):
            if not first:
                for j in range(self.num_agents):
                    actions[j][2] = 1
            first = False
            o, r = self.step_(actions)
            reward += r

        self.m_g *= self.g
        done = (done != 0)
        self.total_step += self.args.n_rollout_threads
        self.killed += (self.t_killed != 0)

        for i in range(self.num_agents):
            info[i]['active_masks'] = True
            info[i]['bad_transition'] = False
            info[i]['collective_return'] = self.sum_r[i]
            info[i]['behavior'] = self.hit[i]
            if self.killed[i] >= 1:
                done[i] = True
                info[i]['active_masks'] = False
                if self.killed[i] == 1:
                    info[i]['episode'] = {'r': self.sum_r[i], 'r_g': self.sum_r_g[i],
                                          'r_g_i': self.sum_r_g_i[i], 'hit': self.hit[i], 'dis': self.sum_dis / self.s_n}
                else:
                    info[i]['bad_transition'] = True
            elif self.s_n >= self.stop_step:
                done[i] = True
                info[i]['episode'] = {'r': self.sum_r[i], 'r_g': self.sum_r_g[i],
                                      'r_g_i': self.sum_r_g_i[i], 'hit': self.hit[i], 'dis': self.sum_dis / self.s_n}

        if np.sum(done) == self.num_agents:
            for i in range(self.num_agents):
                info[i]['active_masks'] = True
                if self.killed[i] == 0 and self.s_n >= self.stop_step:
                    info[i]['bad_transition'] = True
                elif self.killed[i] == 1:
                    info[i]['bad_transition'] = False
                else:
                    info[i]['bad_transition'] = True

        reward = reward.reshape(-1, 1)
        global_reward = np.sum(reward)

        if self.share_reward:
            reward = [[global_reward]] * self.num_agents

        return o, reward, done, info

    def step_(self, actions_):
        actions = deepcopy(actions_)
        for action, agent in zip(actions, self.agents):
            agent.step(deepcopy(action))
        for i in range(self.num_agents, len(self.server.players)):
            self.server.players[i].step()

        self.server.Update()
        t_rewards, t_rewards2 = [], []
        for i in range(self.num_agents):
            a, b = self.parse_reward(self.agents[i], i)
            t_rewards.append(a)
            t_rewards2.append(b)
        t_rewards, t_rewards2 = np.array(t_rewards), np.array(t_rewards2)
        rewards = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j:
                    rewards[i] += t_rewards[j]
                else:
                    rewards[i] += t_rewards[j] * self.coop_eps[i]

        self.split = np.zeros(self.num_agents)
        observations = [self.parse_obs(self.agents[i], i, actions)
                        for i in range(self.num_agents)]
        t_dis = self.agents[0].centerPos.clone().sub(
            self.agents[1].centerPos).sqDist() / self.server.config.r
        self.sum_dis += t_dis
        self.near = (t_dis <= 0.5)
        if self.killed[0] + self.killed[1] > 0:
            self.near = False
        self.last_action = deepcopy(actions.reshape(-1))
        self.sum_r += rewards
        self.sum_r_g += rewards * self.m_g
        self.sum_r_g_i += t_rewards2 * self.m_g
        self.s_n += 1

        return observations, rewards

    def reset(self):
        while True:
            self.num_players = self.num_bots + self.num_agents
            self.rewards_forced = [0 for i in range(self.num_agents)]
            self.stop_step = 2000 - \
                np.random.randint(0, 100) * self.action_repeat
            self.last_mass = [None for i in range(self.num_agents)]
            self.killed = np.zeros(self.num_agents)
            self.t_killed = np.zeros(self.num_agents)
            self.sum_r = np.zeros((self.num_agents, ))
            self.sum_r_g = np.zeros((self.num_agents, ))
            self.sum_r_g_i = np.zeros((self.num_agents, ))
            self.sum_dis = 0.
            self.m_g = 1.
            self.last_action = [0 for i in range(3 * self.num_players)]
            self.s_n = 0

            self.split = np.zeros(self.num_agents)
            self.hit = np.zeros((self.num_agents, 4))
            self.near = False
            if self.curriculum_learning:
                # script agent speed curriculum is set here.
                up = min(
                    1.0, max(0.0, (self.total_step - self.up_step) / self.up_step))
                low = min(
                    1.0, max(0.0, (self.total_step - self.low_step) / self.up_step))
                self.bot_speed = rand(low, up)
            else:
                self.bot_speed = 1.0

            self.server = GameServer(self)
            self.server.start(self.gamemode)
            self.agents = [Player(self.server) for _ in range(self.num_agents)]
            self.bots = [Bot(self.server) for _ in range(self.num_bots)]
            self.players = self.agents + self.bots
            self.server.addPlayers(self.players)
            self.viewer = None
            self.server.Update()
            observations = [self.parse_obs(self.agents[i], i)
                            for i in range(self.num_agents)]
            success = True
            for i in range(self.num_agents):
                # sometimes the agent dies just after initialization, we should avoid this.
                if np.sum(observations[i]) == 0.:
                    success = False
            #observations = np.array(observations)
            #observations = {'agent-'+str(i): observations[i] for i in range(self.num_agents)}
            if success:
                break
        return observations

    def parse_obs(self, player, id, action=None):
        '''

        function parse_obs will return obs_id
        obs_id is a 578D array, first 560D is information of all entities around agent_id, last 28D is global information

        '''
        n = [10, 5, 5, 10, 10]  # , 10, 5, 5, 10, 10] # the agent can observe at most 10 self-cells, 5 foods, 5 virus, 10 other script agent cells, 10 other outside agent cells
        s_glo = 21  # global information size
        # , 1, 1, 1, 1, 1] # information size of self-cell, food, virus, script agent cell and other outside agent cells.
        s_size_i = [15, 7, 5, 15, 15]
        s_size = np.sum(np.array(n) * np.array(s_size_i)) + s_glo
        if len(player.cells) == 0:
            return np.zeros(s_size)
        obs = [[], [], [], [], []]
        for cell in player.viewNodes:
            t, feature = self.cell_obs(cell, player, id)
            obs[t].append(feature)

        for i in range(len(obs)):
            obs[i].sort(key=lambda x: x[0] ** 2 + x[1] ** 2)
        obs_f = np.zeros(s_size)
        bound = self.players[id].get_view_box()
        b_x = (bound[1] - bound[0]) / self.server.config.serverViewBaseX
        b_y = (bound[3] - bound[2]) / self.server.config.serverViewBaseY
        base = 0
        for j in range(5):
            lim = min(len(obs[j % 5]), n[j % 5])
            if j >= 5:
                for i in range(lim):
                    # reserved function, it can be used for different visibility of policy or value, but I didn't use it.
                    obs_f[base + i] = 1.
            else:
                for i in range(lim):
                    obs_f[base + i * s_size_i[j]:base +
                          (i + 1) * s_size_i[j]] = obs[j][i]
            base += s_size_i[j] * n[j]

        position_x = (player.centerPos.x) / \
            self.server.config.borderWidth * 2  # [-1, 1]
        position_y = (player.centerPos.y) / \
            self.server.config.borderHeight * 2  # [-1, 1]
        # obs_f[-21:] are global information
        obs_f[-1] = position_x
        obs_f[-2] = position_y
        obs_f[-3] = player.centerPos.sqDist() / self.server.config.r
        obs_f[-4] = b_x
        obs_f[-5] = b_y
        obs_f[-6] = len(obs[0])
        obs_f[-7] = len(obs[1])
        obs_f[-8] = len(obs[2])
        obs_f[-9] = len(obs[3])
        obs_f[-10] = len(obs[4])
        obs_f[-11] = player.maxcell().radius / 400
        obs_f[-12] = player.mincell().radius / 400
        obs_f[-16:-13] = self.last_action[id * 3: id * 3 + 3]
        obs_f[-17] = self.bot_speed
        obs_f[-18] = (self.killed[id] != 0)
        obs_f[-19] = (self.killed[1 - id] != 0)
        obs_f[-20] = sum([c.mass for c in player.cells]) / 50
        obs_f[-21] = sum([c.mass for c in self.agents[1 - id].cells]) / 50

        return deepcopy(obs_f)

    def cell_obs(self, cell, player, id):
        if cell.cellType == 0:
            # player features
            boost_x = (cell.boostDistance * cell.boostDirection.x) / \
                self.server.config.splitVelocity  # [-1, 1]
            boost_y = cell.boostDistance * cell.boostDirection.y / \
                self.server.config.splitVelocity  # [-1, 1]
            radius = cell.radius / 400
            log_radius = np.log(cell.radius / 100)
            position_x = (cell.position.x) / \
                self.server.config.borderWidth * 2  # [-1, 1]
            position_y = (cell.position.y) / \
                self.server.config.borderHeight * 2  # [-1, 1]
            relative_position_x = (cell.position.x - player.centerPos.x) / \
                self.server.config.serverViewBaseX * 2  # [-1, 1]
            relative_position_y = (cell.position.y - player.centerPos.y) / \
                self.server.config.serverViewBaseY * 2  # [-1, 1]
            v_x = (cell.position.x - cell.last_position.x) / \
                self.server.config.serverViewBaseX * 2 * 30
            v_y = (cell.position.y - cell.last_position.y) / \
                self.server.config.serverViewBaseY * 2 * 30
            features_player = [relative_position_x, relative_position_y, position_x, position_y, boost_x, boost_y, radius, log_radius, float(cell.canRemerge == True), relative_position_x ** 2 + relative_position_y ** 2, v_x, v_y, float(
                cell.radius * 1.15 > player.maxcell().radius), float(cell.radius * 1.15 > player.mincell().radius), cell.position.sqDist() / self.server.config.r]
            if cell.owner == player:
                c_t = 0
                if boost_x or boost_y:
                    self.split[id] = True
            elif cell.owner.pID < self.num_agents:
                c_t = 4
            else:
                c_t = 3
            return c_t, features_player

        elif cell.cellType == 1:
            # food features
            radius = (cell.radius - (self.server.config.foodMaxRadius + self.server.config.foodMinRadius) /
                      2) / (self.server.config.foodMaxRadius - self.server.config.foodMinRadius) * 2
            log_radius = np.log(
                cell.radius / ((self.server.config.foodMaxRadius + self.server.config.foodMinRadius) / 2))
            position_x = (cell.position.x) / \
                self.server.config.borderWidth * 2  # [-1, 1]
            position_y = (cell.position.y) / \
                self.server.config.borderHeight * 2  # [-1, 1]
            relative_position_x = (cell.position.x - player.centerPos.x) / \
                self.server.config.serverViewBaseX * 2  # [-1, 1]
            relative_position_y = (cell.position.y - player.centerPos.y) / \
                self.server.config.serverViewBaseY * 2  # [-1, 1]
            features_food = [relative_position_x, relative_position_y, position_x, position_y,
                             radius, log_radius, relative_position_x ** 2 + relative_position_y ** 2]
            return cell.cellType, features_food

        elif cell.cellType == 2:
            # virus features
            boost_x = (cell.boostDistance * cell.boostDirection.x) / \
                self.server.config.splitVelocity  # [-1, 1]
            boost_y = cell.boostDistance * cell.boostDirection.y / \
                self.server.config.splitVelocity  # [-1, 1]
            radius = (cell.radius - (self.server.config.virusMaxRadius + self.server.config.virusMinRadius) /
                      2) / (self.server.config.virusMaxRadius - self.server.config.virusMinRadius) * 2
            log_radius = np.log(
                cell.radius / ((self.server.config.virusMaxRadius + self.server.config.virusMinRadius) / 2))
            position_x = (cell.position.x) / \
                self.server.config.borderWidth * 2  # [-1, 1]
            position_y = (cell.position.y) / \
                self.server.config.borderHeight * 2  # [-1, 1]
            relative_position_x = (cell.position.x - player.centerPos.x) / \
                self.server.config.serverViewBaseX * 2  # [-1, 1]
            relative_position_y = (cell.position.y - player.centerPos.y) / \
                self.server.config.serverViewBaseY * 2  # [-1, 1]
            features_virus = [relative_position_x, relative_position_y, position_x,
                              position_y, relative_position_x ** 2 + relative_position_y ** 2]
            return cell.cellType, features_virus

        elif cell.cellType == 3:
            # I didn't consider action and observation of ejection also I still implement it.
            return None
            # ejected mass features
            boost_x = (cell.boostDistance * cell.boostDirection.x) / \
                self.server.config.splitVelocity  # [-1, 1]
            boost_y = cell.boostDistance * cell.boostDirection.y / \
                self.server.config.splitVelocity  # [-1, 1]
            position_x = (cell.position.x - self.server.config.borderWidth /
                          2) / self.server.config.borderWidth * 2  # [-1, 1]
            position_y = (cell.position.y - self.server.config.borderHeight /
                          2) / self.server.config.borderHeight * 2  # [-1, 1]
            relative_position_x = (cell.position.x - player.centerPos.x + 0 *
                                   self.server.config.serverViewBaseX / 2) / self.server.config.serverViewBaseX * 2  # [-1, 1]
            relative_position_y = (cell.position.y - player.centerPos.y + 0 *
                                   self.server.config.serverViewBaseY / 2) / self.server.config.serverViewBaseY * 2  # [-1, 1]
            features_ejected = [boost_x, boost_y, position_x,
                                position_y, relative_position_x, relative_position_y]
            return cell.cellType, features_ejected

    def parse_reward(self, player, id):
        mass_reward, kill_reward, killed_reward = self.calc_reward(player, id)
        if mass_reward < 0 and self.reward_settings == "agg":
            mass_reward = 0  # no death penalty.
        if mass_reward > 0:
            if kill_reward:
                self.hit[id][2] += 1
            elif self.split[id]:
                self.hit[id][0] += 1
            elif self.near:
                self.hit[id][3] += 1
            else:
                self.hit[id][1] += 1  # numbers of 4 events

        if len(player.cells) == 0:
            self.t_killed[id] += 1
        # reward for being big, not dead, eating part of others, killing all of others, not be eaten by someone
        # when the agent eats another outside agent, kill_reward will be mass_reward * (-1)
        reward = mass_reward * self.mass_reward_eps + \
            kill_reward * self.kill_reward_eps[id] + \
            killed_reward * self.killed_reward_eps
        reward2 = mass_reward * self.mass_reward_eps + \
            killed_reward * self.killed_reward_eps
        return reward, reward2

    def calc_reward(self, player, id):
        mass_reward = sum([c.mass for c in player.cells])
        if self.last_mass[id] is None:
            mass_reward = 0
        else:
            mass_reward -= self.last_mass[id]
        self.last_mass[id] = sum([c.mass for c in player.cells])
        kill_reward = player.killreward
        killedreward = player.killedreward
        return mass_reward, kill_reward, killedreward

    # can be used to add multiple directions of the agent to render, add_dir should be used before render. a = [direction_x_0, direction_y_0, direction_x_1, direction_y_1, ... , direction_x_m-1, direction_y_m-1], m = number of different directions.
    def add_dir(self, a):
        self.dir = deepcopy(a)

    def render(self, playeridx, mode='human', name=""):
        from . import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(
                self.server.config.serverViewBaseX, self.server.config.serverViewBaseY)
            self.render_border()
            self.render_grid()

        bound = self.players[playeridx].get_view_box()
        self.viewer.set_bounds(*bound)

        self.geoms_to_render = []
        self.render_dir(self.players[playeridx].centerPos)
        for node in self.players[playeridx].viewNodes:
            self.add_cell_geom(node)

        self.geoms_to_render = sorted(
            self.geoms_to_render, key=lambda x: x.order)
        for geom in self.geoms_to_render:
            self.viewer.add_onetime(geom)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array', name=name)

    def render_border(self):
        from . import rendering
        map_left = - self.server.config.borderWidth / 2
        map_right = self.server.config.borderWidth / 2
        map_top = - self.server.config.borderHeight / 2
        map_bottom = self.server.config.borderHeight / 2
        line_top = rendering.Line((map_left, map_top), (map_right, map_top))
        line_top.set_color(0, 0, 0)
        self.viewer.add_geom(line_top)
        line_bottom = rendering.Line(
            (map_left, map_bottom), (map_right, map_bottom))
        line_bottom.set_color(0, 0, 0)
        self.viewer.add_geom(line_bottom)
        line_left = rendering.Line((map_left, map_top), (map_left, map_bottom))
        line_left.set_color(0, 0, 0)
        self.viewer.add_geom(line_left)
        map_right = rendering.Line(
            (map_right, map_top), (map_right, map_bottom))
        map_right.set_color(0, 0, 0)
        self.viewer.add_geom(map_right)
        cellwall = rendering.make_circle(
            radius=self.server.config.r, res=50, filled=False)
        cellwall.set_color(0, 0, 0)
        self.viewer.add_geom(cellwall)

    def render_grid(self):
        from . import rendering
        map_left = - self.server.config.borderWidth / 2
        map_right = self.server.config.borderWidth / 2
        map_top = - self.server.config.borderHeight / 2
        map_bottom = self.server.config.borderHeight / 2
        for i in range(0, int(map_right), 100):
            line = rendering.Line((i, map_top), (i, map_bottom))
            line.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(line)
            line = rendering.Line((-i, map_top), (-i, map_bottom))
            line.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(line)

        for i in range(0, int(map_bottom), 100):
            line = rendering.Line((map_left, i), (map_right, i))
            line.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(line)
            line = rendering.Line((map_left, -i), (map_right, -i))
            line.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(line)

    def render_dir(self, center):
        from . import rendering
        for i in range(len(self.dir)):
            line = rendering.Line(
                (0, 0), (self.dir[i][0] * 500, self.dir[i][1] * 500))
            line.set_color((len(self.dir) - i) /
                           len(self.dir), i / len(self.dir), 0)
            line.order = i
            xform = rendering.Transform()
            line.add_attr(xform)
            xform.set_translation(center.x, center.y)
            self.geoms_to_render.append(line)

    def add_cell_geom(self, cell):
        from . import rendering
        if cell.cellType == 0:
            cellwall = rendering.make_circle(radius=cell.radius)
            cellwall.set_color(cell.color.r * 0.75 / 255.0,
                               cell.color.g * 0.75 / 255.0, cell.color.b * 0.75 / 255.0)
            xform = rendering.Transform()
            cellwall.add_attr(xform)
            xform.set_translation(cell.position.x, cell.position.y)
            cellwall.order = cell.radius
            self.geoms_to_render.append(cellwall)

            geom = rendering.make_circle(
                radius=cell.radius - max(10, cell.radius * 0.1))
            geom.set_color(cell.color.r / 255.0, cell.color.g /
                           255.0, cell.color.b / 255.0)
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(cell.position.x, cell.position.y)
            if cell.owner.maxradius < self.server.config.virusMinRadius:
                geom.order = cell.owner.maxradius + 0.0001
            elif cell.radius < self.server.config.virusMinRadius:
                geom.order = self.server.config.virusMinRadius - 0.0001
            else:
                geom.order = cell.owner.maxradius + 0.0001
            self.geoms_to_render.append(geom)

        elif cell.cellType == 2:
            geom = rendering.make_circle(radius=cell.radius)
            geom.set_color(cell.color.r / 255.0, cell.color.g /
                           255.0, cell.color.b / 255.0, 0.6)
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(cell.position.x, cell.position.y)
            geom.order = cell.radius
            self.geoms_to_render.append(geom)

        else:
            geom = rendering.make_circle(radius=cell.radius)
            geom.set_color(cell.color.r / 255.0, cell.color.g /
                           255.0, cell.color.b / 255.0)
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(cell.position.x, cell.position.y)
            geom.order = cell.radius
            self.geoms_to_render.append(geom)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
