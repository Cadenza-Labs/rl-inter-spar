import importlib
import time
import numpy as np
import torch as th
from torch import nn
import supersuit as ss
import gymnasium as gym
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from torch.distributions import Categorical
import warnings
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from tqdm import tqdm


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


@th.no_grad()
def playground(
    envs,
    agent1,
    agent2,
    metrics: dict,
    device,
    video_path,
    rounds_to_record=30,
    max_video_length=2000,
    sliding_window=100,
):
    # The video wrapper is not working with our env so we have to use our own. So we will save the first round of each match as a video:
    player1 = agent1
    player2 = agent2
    # TRY NOT TO MODIFY: start the game
    obs = th.Tensor(envs.reset()).to(device)
    total_length = 0
    wins = 0
    frames = []
    num_rounds = 0
    start_time = time.time()
    metric_values = {k: [] for k in metrics.keys()}
    while True:
        actions = np.zeros(len(obs), dtype=np.int64)
        action1 = player1.get_action(obs[::2])
        action2 = player2.get_action(obs[1::2])
        for name, fn in metrics.items():
            metric_values[name].append(fn(obs))
        actions[::2] = action1.cpu().numpy()
        actions[1::2] = action2.cpu().numpy()
        obs, rewards, dones, _ = envs.step(actions)

        frame = obs[0, :, :, 0]
        frames.append(np.stack([frame, frame, frame], axis=2))
        obs = th.Tensor(obs).to(device)
        num_rounds += np.logical_or(rewards > 0, dones).sum().item()
        total_length += 1 * obs.shape[0]
        wins += (rewards[::2] == 1).sum().item()
        if rewards[0] != 0:
            rounds_to_record -= 1
        if dones[0] or rounds_to_record == 0:
            print(f"SPS: {total_length  / (time.time() - start_time)}")
            print(f"{player1.name} vs {player2.name}: {num_rounds} rounds played")
            print(
                f"{player1.name} vs {player2.name}: {wins} wins /  {num_rounds - wins} losses"
            )
            print(f"Average episode length: {total_length / num_rounds}")
            frames = np.stack(frames, dtype=np.uint8)
            total_frames = min(len(frames), max_video_length)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            # Initialize the Pong frame
            pong_img = ax1.imshow(frames[0], cmap="gray")

            # Initialize the agent value plot
            time_values = list(range(total_frames))
            plots = []
            ymin = -1
            ymax = 1
            for name, values in metric_values.items():
                values = values[: min(sliding_window, total_frames)]
                (plot,) = ax2.plot(
                    time_values[: min(sliding_window, total_frames)],
                    values,
                    label=name,
                    alpha=0.5,
                )
                ymin = min(ymin, min(values))
                ymax = max(ymax, max(values))
                plots.append(plot)
            ax2.legend()
            # ax2.set_xlim(0, total_frames)
            ax2.set_ylim(ymin, ymax)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Value")
            ax2.set_title("Value over time")

            # Update function for animation
            pbar = tqdm(total=total_frames)

            def update(frame):
                pbar.n = frame
                pbar.refresh()
                pong_img.set_array(frames[frame])
                for plot, values in zip(plots, metric_values.values()):
                    plot.set_data(
                        time_values[max(0, frame - sliding_window) : frame + 1],
                        values[max(0, frame - sliding_window) : frame + 1],
                    )
                if frame > sliding_window:
                    ax2.set_xlim(frame - sliding_window, frame)
                return [pong_img] + plots

            # Create the animation
            animation = FuncAnimation(fig, update, frames=total_frames, blit=True)

            # Save the animation as a video
            video_path.mkdir(parents=True, exist_ok=True)
            pbar.set_description("Generating video")
            animation.save(
                str(video_path / f"{player1.name}_vs_{player2.name}.mp4"), fps=30
            )
            print(
                "Saved video to"
                + str(video_path / f"{player1.name}_vs_{player2.name}.mp4")
            )
            # Save value plot as a png
            fig.savefig(str(video_path / f"{player1.name}_vs_{player2.name}.png"))
            break
