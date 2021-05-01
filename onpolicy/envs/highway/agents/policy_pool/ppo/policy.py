import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.utils.util import get_shape_from_obs_space

class actor(nn.Module):
    def __init__(self, args, obs_space, action_space, hidden_size=64, use_recurrent_policy=True):
        super(actor, self).__init__()
        self.hidden_size = hidden_size
        self._use_recurrent_policy = use_recurrent_policy

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._recurrent_N = args.recurrent_N

        self.tpdv = dict(dtype=torch.float32)

        obs_shape = get_shape_from_obs_space(obs_space)

        base = CNNBase if len(obs_shape)==3 else MLPBase
        self.base = base(args, obs_shape)
        
        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, _ = self.act(actor_features, available_actions, deterministic)
        
        return actions, rnn_states