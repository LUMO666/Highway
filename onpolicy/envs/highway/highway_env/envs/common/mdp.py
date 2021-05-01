import numpy as np

import sys
sys.path.extend("../")
sys.path.append("../")

class MDP(object):
    def __init__(self, state=0):
        self.state = state
        self.transition = None
        self.reward = None
        self.terminal = None

    def step(self, action, np_random=np.random):
        raise NotImplementedError()

    def reset(self):
        self.state = 0
        return self.state

    @staticmethod
    def from_config(config, np_random=np.random):
        mode = config.get("mode", None)
        transition = np.array(config.get("transition", []))
        reward = np.array(config.get("reward", []))
        cost = np.array(config.get("cost", None))
        terminal = np.array(config.get("terminal", []))
        if mode == "deterministic":
            mdp = DeterministicMDP(transition, reward, terminal)
        elif mode == "stochastic":
            if cost is not None:
                mdp = ConstrainedMDP(transition, reward, cost, terminal)
            else:
                mdp = StochasticMDP(transition, reward, terminal)
        elif mode == "garnet":
            mdp = SparseMDP.make_garnet(config.get("num_states"),
                                        config.get("num_actions"),
                                        config.get("num_transitions"),
                                        config.get("reward_sparsity"),
                                        np_random=np_random)
        elif mode == "uniform":
            mdp = StochasticMDP.make_uniform(config.get("num_states"),
                                             config.get("num_actions"),
                                             np_random=np_random)
        else:
            raise ValueError("Unknown MDP mode in configuration")
        return mdp

    def to_config(self):
        raise NotImplementedError()


class DeterministicMDP(MDP):
    mode = "deterministic"

    def __init__(self, transition, reward, terminal=None, state=0):
        """
        :param transition: array of shape S x A
        :param reward: array of shape S x A
        :param terminal: array of shape S
        :param int state: initial state
        """
        super(DeterministicMDP, self).__init__(state)
        self.transition = transition
        self.reward = reward
        self.terminal = terminal
        if terminal is None or not np.size(terminal):
            self.terminal = np.zeros(np.shape(transition)[0])
        self.terminal = self.terminal.astype(bool)

    def step(self, action, np_random=np.random):
        reward = self.reward[self.state, action]
        done = self.terminal[self.state]
        self.state = self.next_state(self.state, action)
        return self.state, reward, done, {}

    def next_state(self, state, action):
        return self.transition[state, action]

    def randomize(self, np_random=np.random):
        self.transition = np_random.choice(range(np.shape(self.transition)[0]), size=np.shape(self.transition))
        self.reward = np_random.rand(*np.shape(self.reward))

    def to_config(self):
        return dict(
            mode=self.mode,
            transition=self.transition.tolist(),
            reward=self.reward.tolist(),
            terminal=self.terminal.tolist()
        )

    def update(self, config):
        if "transition" in config:
            self.transition = np.array(config["transition"])
        if "reward" in config:
            self.reward = np.array(config["reward"])
        if "terminal" in config:
            self.terminal = np.array(config["terminal"])


class StochasticMDP(DeterministicMDP):
    mode = "stochastic"

    def __init__(self, transition, reward, terminal=None, state=0):
        """
        :param transition: array of size S x A x S'
        :param reward:  array of shape S x A
        :param terminal:  array of shape S
        :param int state: initial state
        """
        super(StochasticMDP, self).__init__(transition, reward, terminal, state)

    def step(self, action, np_random=np.random):
        reward = self.reward[self.state, action]
        done = self.terminal[self.state]
        self.state = self.next_state(self.state, action)
        return self.state, reward, done, {}

    def next_state(self, state, action, np_random=np.random):
        probs = self.transition[state, action, :]
        nexts = np.arange(np.shape(self.transition)[2])
        return np_random.choice(nexts, p=probs)

    @staticmethod
    def from_deterministic(mdp: DeterministicMDP):
        shape = np.shape(mdp.transition)
        new_transition = np.zeros((shape[0], shape[1], shape[0]))
        for s in range(shape[0]):
            for a in range(shape[1]):
                new_transition[s, a, int(mdp.transition[s, a])] = 1
        return StochasticMDP(new_transition, mdp.reward, mdp.terminal)

    @staticmethod
    def make_garnet(num_states, num_actions, num_transitions, reward_sparsity, np_random=np.random):
        """
            Make a GARNET: an MDP with a given number of transitions at each (state,action) pair, and sparse reward
        :param num_states: number of states
        :param num_actions: number of rewards
        :param num_transitions: number of transitions from each (state, action) pair
        :param reward_sparsity: proportion of non-zero rewards
        :param np_random: randomness source
        :return: a GARNET Stochastic MDP
        """
        # Sample N random transitions for each state-action pair
        transition = np.zeros((num_states, num_actions, num_states))
        for s, a in np.ndindex((num_states, num_actions)):
            next = np_random.choice(range(num_states), num_transitions)
            cumulative = np.concatenate(([0], np.sort(np_random.rand(num_transitions - 1)), [1]), axis=0)
            for k in range(num_transitions):
                transition[s, a, next[k]] += cumulative[k + 1] - cumulative[k]

        # Sample N rewards uniformly distributed
        num_sparsity = int(num_actions * num_states * reward_sparsity)
        reward = np.zeros((num_states * num_actions))
        reward[:num_sparsity] = np_random.rand(num_sparsity)
        np_random.shuffle(reward)
        reward = reward.reshape((num_states, num_actions))
        return StochasticMDP(transition, reward)

    @staticmethod
    def make_uniform(num_states, num_actions, np_random=np.random):
        transition = np_random.rand(num_states, num_actions, num_states)
        transition /= np.sum(transition, axis=2, keepdims=True)
        reward = np_random.rand(num_states, num_actions)
        return StochasticMDP(transition, reward)


class SparseMDP(StochasticMDP):
    mode = "sparse"

    def __init__(self, transition, next, reward, terminal=None, state=0):
        """
        :param transition: probabilities, array of size S x A x B
        :param next: next state indexes, array of size S x A x B
        :param reward:  array of shape S x A
        :param terminal:  array of shape S
        :param int state: initial state
        """
        super().__init__(transition, reward, terminal=terminal, state=state)
        self.next = next

    def next_state(self, state, action, np_random=np.random):
        next_state_index = super().next_state(state, action, np_random=np_random)
        return self.next[state, action, next_state_index]

    @staticmethod
    def make_garnet(num_states, num_actions, num_transitions, reward_sparsity, np_random=np.random):
        """
            Make a GARNET: an MDP with a given number of transitions at each (state,action) pair, and sparse reward
        :param num_states: number of states
        :param num_actions: number of rewards
        :param num_transitions: number of transitions from each (state, action) pair
        :param reward_sparsity: proportion of non-zero rewards
        :param np_random: randomness source
        :return: a GARNET Stochastic MDP
        """
        # Sample N random transitions for each state-action pair
        transition = np.zeros((num_states, num_actions, num_transitions))
        next_states = np_random.randint(num_states, size=(num_states, num_actions, num_transitions))
        cumulative = np.sort(np_random.uniform(size=(num_states, num_actions, num_transitions - 1)), axis=-1)
        transition[:, :, 0] = cumulative[:, :, 0]
        for k in range(num_transitions - 2):
            transition[:, :, k + 1] += cumulative[:, :, k + 1] - cumulative[:, :, k]
        transition[:, :, num_transitions - 1] = 1 - cumulative[:, :, -1]

        # Sample N rewards uniformly distributed
        num_sparsity = int(num_actions * num_states * reward_sparsity)
        reward = np.zeros((num_states * num_actions))
        reward[:num_sparsity] = np_random.uniform(size=num_sparsity)
        np_random.shuffle(reward)
        reward = reward.reshape((num_states, num_actions))
        return SparseMDP(transition, next_states, reward)


class ConstrainedMDP(StochasticMDP):
    def __init__(self, transition, reward, cost, terminal=None, state=0):
        super().__init__(transition, reward, terminal, state)
        self.cost = cost

    def step(self, action, np_random=np.random):
        reward = self.reward[self.state, action]
        cost = self.cost[self.state, action]
        done = self.terminal[self.state]
        self.state = self.next_state(self.state, action)
        return self.state, reward, done, {"c_": cost}
