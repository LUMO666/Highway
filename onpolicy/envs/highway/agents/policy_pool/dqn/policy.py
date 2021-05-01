from onpolicy.envs.highway.agents.common.models import model_factory
import numpy as np
import torch
from torch import nn
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_shape_from_obs_space

class actor(nn.Module):
    def __init__(self, args, obs_space, action_space, hidden_size, use_recurrent_policy=False):
        super(actor, self).__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        obs_shape = get_shape_from_obs_space(obs_space)
        self.tpdv = dict(dtype=torch.float32)

        self.config = {
            "model": {
                "type": "DuelingNetwork",
                "base_module": {
                    "layers": hidden_size
                },
                "value": {
                    "layers": [hidden_size[1]]
                },
                "advantage": {
                    "layers": [hidden_size[1]]
                },
                "in": int(np.prod(obs_shape)),
                "out": self.action_space.n
            },
        }
        self.value_net = model_factory(self.config["model"])

    def forward(self, obs, rnn_states=None, masks=None, available_actions=None, deterministic=False):   
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
        :return: [a0, a1, a2...], a sequence of actions to perform
        """
        obs = check(obs).to(**self.tpdv)
        values = self.value_net(obs)
        if deterministic:
            actions = torch.argmax(values, axis=-1, keepdim=True)
        else:
            print("only support greedy action while evaluating!")
            raise NotImplementedError

        return actions, rnn_states

    def act(self, obs):
        """
            Act according to the state-action value model and an exploration policy
        :param state: current state
        :param step_exploration_time: step the exploration schedule
        :return: an action
        """
        obs = check(obs).to(**self.tpdv)
        if len(obs.shape) < 2:
            obs = obs.unsqueeze(0) 
        values = self.value_net(obs)
        actions = torch.argmax(values, axis=-1, keepdim=True)

        return actions.detach().numpy()

    def load_state_dict(self, policy_state_dict):
        self.value_net.load_state_dict(policy_state_dict['state_dict'])