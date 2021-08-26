from onpolicy.envs.highway.agents.common.models import model_factory
import numpy as np
import torch
from torch import nn
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.bftq.models import BudgetedMLP
from onpolicy.algorithms.bftq.bftq import BudgetedFittedQ
from onpolicy.algorithms.bftq.policies import PytorchBudgetedFittedPolicy, RandomBudgetedPolicy, \
    EpsilonGreedyBudgetedPolicy
from gym.utils import seeding

class actor(nn.Module):
    def __init__(self, args, obs_space, action_space, hidden_size, use_recurrent_policy=False):
        super(actor, self).__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        obs_shape = get_shape_from_obs_space(obs_space)
        self.previous_state = None
        self.previous_beta = self.beta = 0
        self.exploration_policy = None
        self.bftq = None

        self.config = {
            "beta": 0,
            "exploration": {
                "temperature": 0,
                "final_temperature": 0,
                "tau": 5000
            },
            "clamp_qc": None,
            "hull_options": {
                "decimals": None,
                "qhull_options": "",
                "remove_duplicates": False,
                "library": "scipy"
            },
            "network": {
                "beta_encoder_type": "LINEAR",
                "size_beta_encoder": 10,
                "activation_type": "RELU",
                "reset_type": "XAVIER",
                "layers": [
                    64,
                    64
                ]
            }
        }
        self.reset()
    
    def forward(self, obs, rnn_states=None, masks=None, available_actions=None, deterministic=False):
        self.beta = self.config["beta"]

        state = state.flatten()
        self.previous_state, self.previous_beta = state, self.beta
        action, self.beta = self.exploration_policy.execute(state, self.beta)
        return action.detach().numpy
   

    def act(self, state):
        """
            Run the exploration policy to pick actions and budgets
        """
        # TODO: Choose the initial budget for the next episode and not at each step
        self.beta = self.config["beta"]

        state = state.flatten()
        self.previous_state, self.previous_beta = state, self.beta
        action, self.beta = self.exploration_policy.execute(state, self.beta)
        return action.detach().numpy

    def load_state_dict(self, policy_state_dict):
        #self.value_net.load_state_dict(policy_state_dict['state_dict'])
        self.bftq._value_network = policy_state_dict
        network = self.bftq._value_network
        self.exploration.pi_greedy.set_network(network)

    def eval(self):
        self.training = False
        self.config['exploration']['temperature'] = 0
        self.config['exploration']['final_temperature'] = 0
        self.exploration_policy.config = self.config["exploration"]


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        torch.manual_seed(seed & ((1 << 63) - 1))  # torch seeds are int64
        return [seed]

    def reset(self):
        if not self.np_random:
            self.seed()
        network = BudgetedMLP(size_state=np.prod(obs_shape),
                              n_actions=self.action_space.n,
                              **self.config["network"])
        self.bftq = BudgetedFittedQ(value_network=network, config=self.config, writer=self.writer)
        self.exploration_policy = EpsilonGreedyBudgetedPolicy(
            pi_greedy=PytorchBudgetedFittedPolicy(
                network,
                self.bftq.betas_for_discretisation,
                self.bftq.device,
                self.config["hull_options"],
                self.config["clamp_qc"],
                np_random=self.np_random
            ),
            pi_random=RandomBudgetedPolicy(n_actions=self.env.action_space.n, np_random=self.np_random),
            config=self.config["exploration"],
            np_random=self.np_random
        )