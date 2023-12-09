import importlib

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import supersuit as ss
import gymnasium


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def atari_network(orth_init=False):
    init = layer_init if orth_init else lambda m: m
    return nn.Sequential(
        init(nn.Conv2d(4, 32, 8, stride=4)),
        nn.ReLU(),
        init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        init(nn.Linear(64 * 7 * 7, 512)),
        nn.ReLU(),
    )

def preprocess(x):
    x = x.clone()
    x[:, :, :, [0, 1, 2, 3]] /= 255.0
    return x.permute((0, 3, 1, 2))

class Agent(nn.Module):
    def __init__(self, envs, share_network=False):
        super().__init__()
        self.actor_network = atari_network(orth_init=True)
        self.share_network = share_network
        if share_network:
            self.critic_network = self.actor_network
        else:
            self.critic_network = atari_network(orth_init=True)
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)


    def get_value(self, x):
        x = preprocess(x)
        return self.critic(self.critic_network(x))

    def get_action(self, x):
        x = preprocess(x)
        logits = self.actor(self.actor_network(x))
        probs = Categorical(logits=logits)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        x = preprocess(x)
        logits = self.actor(self.actor_network(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(self.critic_network(x)),
        )

    def load(self, path):
        self.load_state_dict(torch.load(path))
        if self.share_network:
            self.critic_network = self.actor_network


def pong_obs_modification(obs, _space, player_id):
    obs[:9, :, :] = 0
    # Todo: check player_id
    if "second" in player_id:
        # Mirror the image
        obs = obs[:, ::-1, :]
    return obs


def get_env(args, run_name):
    env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env()
    env = ss.max_observation_v0(env, 2)
    # env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    # Remove the score from the observation
    if "pong" in args.env_id:
        env = ss.lambda_wrappers.observation_lambda_v0(
            env,
            pong_obs_modification,
        )
    # env = ss.agent_indicator_v0(env, type_only=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = gymnasium.wrappers.RecordEpisodeStatistics(env)
    envs = ss.concat_vec_envs_v1(
        env, args.num_envs // 2, num_cpus=0, base_class="stable_baselines3"
    )
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    # print(envs.observation_space)
    assert isinstance(
        envs.single_action_space, gymnasium.spaces.Discrete
    ), "only discrete action space is supported"
    return envs


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))
