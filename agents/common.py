import importlib
import numpy as np
import torch as th
from torch import nn
import supersuit as ss
import gymnasium as gym
from torch.distributions import Categorical
import warnings


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
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


def pong_obs_modification(obs, _space, player_id):
    # Remove the score
    obs[:9, :, :] = 0
    if "second" in player_id:
        # Mirror the image
        obs = obs[:, ::-1, :]
    return obs


def get_env(args, run_name):
    env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env()
    env = ss.max_observation_v0(env, 2)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    if "pong" in args.env_id:
        env = ss.lambda_wrappers.observation_lambda_v0(
            env,
            pong_obs_modification,
        )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(
        env, args.num_envs // 2, num_cpus=0, base_class="stable_baselines3"
    )
    if args.capture_video:
        warnings.warn("Capture video is currently not supported")
        # envs = VecVideoRecorder(envs, f"videos/{run_name}", capped_cubic_video_schedule)
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"
    return envs


def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


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

    def get_action(self, x, deterministic=False):
        x = preprocess(x)
        logits = self.actor(self.actor_network(x))
        if not deterministic:
            probs = Categorical(logits=logits)
            return probs.sample()
        else:
            return logits.argmax(dim=1)

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

    def load(self, path, device):
        self.load_state_dict(th.load(path, map_location=th.device(device)))
        if self.share_network:
            self.critic_network = self.actor_network

    def forward(self, x):
        """
        Dummy function to collect both actor and critic activations
        """
        _ = self.actor(self.actor_network(x))
        _ = self.critic(self.critic_network(x))
        return th.tensor(0).to(x.device)  # Dummy return
