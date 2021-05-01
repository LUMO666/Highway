import numpy as np
from onpolicy.envs.highway.agents.dynamic_programming.value_iteration import ValueIterationAgent


class RobustValueIterationAgent(ValueIterationAgent):
    def __init__(self, env, config=None, vehicle_id =0):
        super(ValueIterationAgent, self).__init__(config)
        self.vehicle_id = vehicle_id
        self.finite_mdp = self.is_finite_mdp(env)
        if self.finite_mdp:
            self.mdp = env.mdp
        elif not self.finite_mdp:
            try:
                self.mdp = env.unwrapped.to_finite_mdp(vehicle_id)
            except AttributeError:
                raise TypeError("Environment must be of type finite_mdp.envs.finite_mdp.FiniteMDPEnv or handle a "
                                "conversion method called 'to_finite_mdp' to such a type.")
        
        self.env = env
        self.mode = None
        self.transitions =  np.array([])  # Dimension: M x S x A (x S)
        self.rewards = np.array([])  # Dimension: M x S x A
        self.models_from_config()

    @classmethod
    def default_config(cls):
        config = super(RobustValueIterationAgent, cls).default_config()
        config.update(dict(models=[]))
        return config

    def models_from_config(self):
        if not self.config.get("models", None):
            self.mode = self.mdp.mode
            self.transitions = np.array([self.mdp.transition])
            self.rewards =  np.array([self.mdp.reward])
            # raise ValueError("No finite MDP model provided in agent configuration")
        else:
            self.mode = self.config["models"][0]["mode"]  # Assume all modes are the same
            self.transitions = np.array([mdp["transition"] for mdp in self.config["models"]])
            self.rewards = np.array([mdp["reward"] for mdp in self.config["models"]])

    def act(self, state):
        if not self.finite_mdp:
            self.mdp = self.env.unwrapped.to_finite_mdp(self.vehicle_id)
            state = self.mdp.state
            self.transitions = np.array([self.mdp.transition])
            self.rewards =  np.array([self.mdp.reward])
            # print(f"state = {state}")
            self.state_action_value = self.get_state_action_value()
            # print(f"state_action_value = {self.state_action_value[state, :]}")
        return np.argmax(self.state_action_value[state, :])

    def get_state_value(self):
        return self.fixed_point_iteration(
            lambda v: RobustValueIterationAgent.best_action_value(
                RobustValueIterationAgent.worst_case(
                    self.bellman_expectation(v))),
            np.zeros((self.transitions.shape[1],)))

    def get_state_action_value(self):
        return self.fixed_point_iteration(
            lambda q: RobustValueIterationAgent.worst_case(
                self.bellman_expectation(
                    RobustValueIterationAgent.best_action_value(q))),
            np.zeros(self.transitions.shape[1:3]))

    @staticmethod
    def worst_case(model_action_values):
        return np.min(model_action_values, axis=0)

    def bellman_expectation(self, value):
        if self.mode == "deterministic":
            next_v = value[self.transitions]
        elif self.mode == "stochastic":
            v_shaped = value.reshape((1, 1, 1, np.size(value)))
            next_v = (self.transitions * v_shaped).sum(axis=-1)
        else:
            raise ValueError("Unknown mode")
        return self.rewards + self.config["gamma"] * next_v
