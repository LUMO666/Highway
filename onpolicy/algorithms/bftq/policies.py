import abc
import copy
import torch
import numpy as np
from gym.utils import seeding

from onpolicy.algorithms.bftq.greedy_policy import optimal_mixture, pareto_frontier_at



class BudgetedPolicy:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def execute(self, state, beta):
        pass


class EpsilonGreedyBudgetedPolicy(BudgetedPolicy):
    def __init__(self, pi_greedy, pi_random, config, np_random=np.random):
        super().__init__()
        self.pi_greedy = pi_greedy
        self.pi_random = pi_random
        self.config = config
        self.np_random = np_random
        self.time = 0

    def execute(self, state, beta):
        epsilon = self.config['final_temperature'] + (self.config['temperature'] - self.config['final_temperature']) * \
                       np.exp(- self.time / self.config['tau'])
        self.time += 1

        if self.np_random.random_sample() > epsilon:
            return self.pi_greedy.execute(state, beta)
        else:
            return self.pi_random.execute(state, beta)

    def set_time(self, time):
        self.time = time


class RandomBudgetedPolicy(BudgetedPolicy):
    def __init__(self, n_actions, np_random=np.random):
        self.n_actions = n_actions
        self.np_random = np_random

    def execute(self, state, beta):
        action_probs = self.np_random.rand(self.n_actions)
        action_probs /= np.sum(action_probs)
        budget_probs = sample_simplex(coeff=action_probs, bias=beta, min_x=0, max_x=1, np_random=self.np_random)
        action = self.np_random.choice(a=range(self.n_actions), p=action_probs)
        beta = budget_probs[action]
        return action, beta


class PytorchBudgetedFittedPolicy(BudgetedPolicy):
    def __init__(self, network, betas_for_discretisation, device, hull_options, clamp_qc=None, np_random=np.random):
        self.betas_for_discretisation = betas_for_discretisation
        self.device = device
        self.network = None
        self.hull_options = hull_options
        self.clamp_qc = clamp_qc
        self.np_random = np_random
        self.network = network

    def load_network(self, network_path):
        self.network = torch.load(network_path, map_location=self.device)

    def set_network(self, network):
        self.network = copy.deepcopy(network)

    def execute(self, state, beta):
        mixture, _ = self.greedy_policy(state, beta)
        choice = mixture.sup if self.np_random.rand() < mixture.probability_sup else mixture.inf
        return choice.action, choice.budget

    def greedy_policy(self, state, beta):
        with torch.no_grad():
            hull = pareto_frontier_at(
                state=torch.tensor([state], device=self.device, dtype=torch.float32),
                value_network=self.network,
                betas=self.betas_for_discretisation,
                device=self.device,
                hull_options=self.hull_options,
                clamp_qc=self.clamp_qc)
        mixture = optimal_mixture(hull[0], beta)
        return mixture, hull


class DiscreteDistribution(Configurable, ABC):
    def __init__(self, config=None, **kwargs):
        super(DiscreteDistribution, self).__init__(config)
        self.np_random = None

    @abstractmethod
    def get_distribution(self):
        """
        :return: a distribution over actions {action:probability}
        """
        raise NotImplementedError()

    def sample(self):
        """
        :return: an action sampled from the distribution
        """
        distribution = self.get_distribution()
        return self.np_random.choice(list(distribution.keys()), 1, p=np.array(list(distribution.values())))[0]

    def seed(self, seed=None):
        """
            Seed the policy randomness source
        :param seed: the seed to be used
        :return: the used seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_time(self, time):
        """ Set the local time, allowing to schedule the distribution temperature. """
        pass

    def step_time(self):
        """ Step the local time, allowing to schedule the distribution temperature. """
        pass


def exploration_factory(exploration_config, action_space):
    """
        Handles creation of exploration policies
    :param exploration_config: configuration dictionary of the policy, must contain a "method" key
    :param action_space: the environment action space
    :return: a new exploration policy
    """
    from rl_agents.agents.common.exploration.boltzmann import Boltzmann
    from rl_agents.agents.common.exploration.epsilon_greedy import EpsilonGreedy
    from rl_agents.agents.common.exploration.greedy import Greedy

    if exploration_config['method'] == 'Greedy':
        return Greedy(action_space, exploration_config)
    elif exploration_config['method'] == 'EpsilonGreedy':
        return EpsilonGreedy(action_space, exploration_config)
    elif exploration_config['method'] == 'Boltzmann':
        return Boltzmann(action_space, exploration_config)
    else:
        raise ValueError("Unknown exploration method")

class EpsilonGreedy(DiscreteDistribution):
    """
        Uniform distribution with probability epsilon, and optimal action with probability 1-epsilon
    """

    def __init__(self, action_space, config=None):
        super(EpsilonGreedy, self).__init__(config)
        self.action_space = action_space
        if isinstance(self.action_space, spaces.Tuple):
            self.action_space = self.action_space.spaces[0]
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("The action space should be discrete")
        self.config['final_temperature'] = min(self.config['temperature'], self.config['final_temperature'])
        self.optimal_action = None
        self.epsilon = 0
        self.time = 0
        self.writer = None
        self.seed()

    @classmethod
    def default_config(cls):
        return dict(temperature=1.0,
                    final_temperature=0.1,
                    tau=5000)

    def get_distribution(self):
        distribution = {action: self.epsilon / self.action_space.n for action in range(self.action_space.n)}
        distribution[self.optimal_action] += 1 - self.epsilon
        return distribution

    def update(self, values):
        """
            Update the action distribution parameters
        :param values: the state-action values
        :param step_time: whether to update epsilon schedule
        """
        self.optimal_action = np.argmax(values)
        self.epsilon = self.config['final_temperature'] + \
            (self.config['temperature'] - self.config['final_temperature']) * \
            np.exp(- self.time / self.config['tau'])
        if self.writer:
            self.writer.add_scalar('exploration/epsilon', self.epsilon, self.time)

    def step_time(self):
        self.time += 1

    def set_time(self, time):
        self.time = time

    def set_writer(self, writer):
        self.writer = writer

def sample_simplex(coeff, bias, min_x, max_x, np_random=np.random):
    """
    Sample from a simplex.

    The simplex is defined by:
        w.x + b <= 0
        x_min <= x <= x_max

    Warning: this is not uniform sampling.

    :param coeff: coefficient w
    :param bias: bias b
    :param min_x: lower bound on x
    :param max_x: upper bound on x
    :param np_random: source of randomness
    :return: a sample from the simplex
    """
    x = np.zeros(len(coeff))
    indexes = np.asarray(range(0, len(coeff)))
    np_random.shuffle(indexes)
    remain_indexes = np.copy(indexes)
    for i_index, index in enumerate(indexes):
        remain_indexes = remain_indexes[1:]
        current_coeff = np.take(coeff, remain_indexes)
        full_min = np.full(len(remain_indexes), min_x)
        full_max = np.full(len(remain_indexes), max_x)
        dot_max = np.dot(current_coeff, full_max)
        dot_min = np.dot(current_coeff, full_min)
        min_xi = (bias - dot_max) / coeff[index]
        max_xi = (bias - dot_min) / coeff[index]
        min_xi = np.max([min_xi, min_x])
        max_xi = np.min([max_xi, max_x])
        xi = min_xi + np_random.random_sample() * (max_xi - min_xi)
        bias = bias - xi * coeff[index]
        x[index] = xi
        if len(remain_indexes) == 1:
            break
    last_index = remain_indexes[0]
    x[last_index] = bias / coeff[last_index]
    return x