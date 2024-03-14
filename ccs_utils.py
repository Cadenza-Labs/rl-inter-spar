from dataclasses import dataclass
import numpy as np
import torch as th
from torch.utils.data import random_split
from agents.common import Agent, preprocess
import pickle
import scipy.signal
from tqdm.auto import tqdm
import random
from nicehooks import nice_hooks


@dataclass
class Trajectory:
    obs: list[np.ndarray]
    actions: list[np.ndarray]
    rewards: list[np.ndarray]
    terminal: bool


class TrajectoriesCollector:
    def __init__(self, env):
        self.env = env

    def sample(self, policy, steps):
        obs = self.env.reset()
        trajs, i, i_episode = [], 0, 0
        traj = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "terminal": False,
        }
        with tqdm(total=steps, desc="Collecting trajectories...") as pbar:
            while True:
                with th.no_grad():
                    action = policy(obs)
                next_obs, reward, done, infos = self.env.step(action)
                traj["obs"].append(obs)
                traj["actions"].append(action)
                traj["rewards"].append(reward)
                i += 1
                i_episode += 1
                obs = next_obs
                if i % (steps // 10) == 0:
                    pbar.update(i)
                if done.any() or (reward != 0).any():
                    i_episode = 0
                    traj["terminal"] = True
                    trajs.append(Trajectory(**traj))
                    traj = {
                        "obs": [],
                        "actions": [],
                        "rewards": [],
                        "terminal": False,
                    }
                    if i > steps:
                        break
        print(f"Generated {len(trajs)} trajectories.")
        return trajs


def generate_dataset(env, model_path, num_episodes, max_episode_length, seed, device):
    """Generate trajectory data using the given environment and model."""
    model = load_model(model_path, env, device)
    get_action = get_action_fn(model, device)
    trajectory_collector = TrajectoriesCollector(env)
    trajectories = trajectory_collector.sample(
        get_action, num_episodes * max_episode_length
    )
    random.Random(seed).shuffle(trajectories)
    return trajectories


def load_trajectories(dataset_path, gamma):
    """Load trajectories"""
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)
    observation_pairs = sum([traj.obs for traj in dataset], [])
    reward_pairs = sum([traj.rewards for traj in dataset], [])
    return_pairs = th.cat(
        [calculate_returns(traj.rewards, gamma) for traj in dataset], dim=0
    )
    return observation_pairs, return_pairs, reward_pairs


def load_model(model_path, env, device):
    agent = Agent(env)
    agent.load(model_path, device)
    agent.to(device)
    return agent


def get_action_fn(agent, device):
    """
    Load the model and return a function that evaluate the model on a given numpy observation.
    """

    def model(obs: np.ndarray):
        obs = th.tensor(obs, dtype=th.float).to(device)
        return agent.get_action(obs).detach().cpu().numpy()

    return model


def get_hidden_activations_dataset(module, device, observation_pairs):
    """Calculate hidden layer activations for given pair dataset."""
    hidden_act_dataset = []
    for pair in tqdm(observation_pairs, desc="Computing hidden activations"):
        th_pair = preprocess(th.tensor(pair, dtype=th.float).to(device))
        with th.no_grad():
            _, activations = nice_hooks.run(module, th_pair, return_activations=True)
            hidden_act_dataset.append(activations.to("cpu"))
    return hidden_act_dataset


def is_ball_approaching(observation_pairs, device, ball_color=236):
    """Returns tensor of whether ball is approaching.
    True, False: ball approaching left player.
    False, True: ball approaching right player.
    False, False: ball not present/not moving.
    """
    ball_approaching = []
    for pair in observation_pairs:
        # pair (players, img_x, img_y, frames)
        ball_indices = (
            (pair == ball_color).sum(axis=1).argmax(axis=1)
        )  # (players, frames)
        # calculate index difference of where the ball is to get its direction
        # assumption: first frame is before last frame in frame stack
        ball_index_diff = ball_indices[:, 0] - ball_indices[:, -1]  # (players,)
        # ball flying to left is equivalent to ball approaching each player in their own perspective
        ball_approaching_player = ball_index_diff > 0
        ball_approaching.append(ball_approaching_player)
    return th.tensor(ball_approaching, dtype=th.float).to(device)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def calculate_returns(reward_pairs, gamma=0.99):
    """Calculate discounted returns for each step in given pair of reward histories."""
    reward_pairs = np.array(reward_pairs)
    return_pairs = np.stack(
        (
            discount_cumsum(reward_pairs[:, 0], gamma),
            discount_cumsum(reward_pairs[:, 1], gamma),
        )
    ).T
    return th.tensor(return_pairs)


def normalize(self, activations):
    """
    Mean-normalizes the data x (of shape (n, d))
    If self.var_normalize, also divides by the standard deviation
    """
    normalized_x = activations - activations.mean(axis=0, keepdims=True)
    if self.var_normalize:
        normalized_x /= normalized_x.std(axis=0, keepdims=True)

    return normalized_x


def normalize_wrt_ball_approaching_no_player(activation_pairs, ball_approaching):
    """
    Normalize activations with respect to the ball position.
    """
    indices_approaching = th.where(ball_approaching == 1)
    indices_not_approaching = th.where(ball_approaching == 0)
    activations_approaching = normalize(activation_pairs[indices_approaching])
    activations_not_approaching = normalize(activation_pairs[indices_not_approaching])

    # Place the normalized activations back into the combined arrays
    combined_activations = th.zeros_like(activation_pairs)
    combined_activations[indices_approaching] = activations_approaching
    combined_activations[indices_not_approaching] = activations_not_approaching
    return combined_activations


def normalize_wrt_ball_approaching(activation_pairs, ball_pos_pairs):
    """
    Normalize activations with respect to the ball position for each player separately.
    """
    indices_player1_left = th.where(ball_pos_pairs[:, 0] == 0)[0]
    indices_player1_right = th.where(ball_pos_pairs[:, 0] == 1)[0]
    indices_player2_left = th.where(ball_pos_pairs[:, 1] == 0)[0]
    indices_player2_right = th.where(ball_pos_pairs[:, 1] == 1)[0]
    activations_player1_left = normalize(activation_pairs[:, 0][indices_player1_left])
    activations_player1_right = normalize(activation_pairs[:, 0][indices_player1_right])
    activations_player2_left = normalize(activation_pairs[:, 1][indices_player2_left])
    activations_player2_right = normalize(activation_pairs[:, 1][indices_player2_right])

    # Place the normalized activations back into the combined arrays
    combined_activations_player1 = th.zeros_like(activation_pairs[:, 0])
    combined_activations_player2 = th.zeros_like(activation_pairs[:, 1])
    combined_activations_player1[indices_player1_left] = activations_player1_left
    combined_activations_player1[indices_player1_right] = activations_player1_right
    combined_activations_player2[indices_player2_left] = activations_player2_left
    combined_activations_player2[indices_player2_right] = activations_player2_right
    return th.stack((combined_activations_player1, combined_activations_player2), dim=1)


@th.no_grad()
def extract_activations(
    model,
    layer_name,
    dataset_path,
    verbose,
    device,
    test_fraction,
    gamma,
    seed,
    normalize=True,
):
    if verbose:
        print("get hidden activations")
    # TODO: load outside of CCS to avoid dupicate
    # TODO?: we could store the activations of each layer in different files
    activations_path = dataset_path.with_suffix("") / "activations.pt"
    ball_approaching_path = dataset_path.with_suffix("") / "ball_pos.pt"
    observation_pairs, return_pairs, reward_pairs = load_trajectories(
        dataset_path, gamma
    )
    if activations_path.exists() and ball_approaching_path.exists():
        if verbose:
            print(f"Found cached activations at: {activations_path}")
        activation_pairs = th.load(activations_path)
        ball_approaching_pairs = th.load(ball_approaching_path)
        if verbose:
            print(f"loaded return pairs of shape {return_pairs.shape}")
            print(f"loaded reward pairs of shape {reward_pairs.shape}")
            print(f"loaded activation pairs of shape {activation_pairs.shape}")
    else:
        if verbose:
            print(f"No cached activations. Computing them...")
        activations_path.parent.mkdir(parents=True, exist_ok=True)
        if not ball_approaching_path.exists():
            ball_approaching_pairs = is_ball_approaching(observation_pairs, device)
            th.save(ball_approaching_pairs, ball_approaching_path)
        else:
            ball_approaching_pairs = th.load(ball_approaching_path)
        if not activations_path.exists():
            activation_pairs = get_hidden_activations_dataset(
                model, device, observation_pairs
            )
        else:
            activation_pairs = th.load(activations_path)
            th.save(activation_pairs, activations_path)
    activation_pairs = (
        th.cat(
            [pair[layer_name].unsqueeze(0) for pair in activation_pairs],
            axis=0,
        )
        .detach()
        .to(device)
    )
    if normalize:
        activation_pairs = normalize_wrt_ball_approaching_no_player(
            activation_pairs, ball_approaching_pairs
        )
    train_activations, test_activations = random_split(
        activation_pairs,
        lengths=[1 - test_fraction, test_fraction],
        generator=th.Generator().manual_seed(seed),
    )
    train_returns, test_returns = random_split(
        return_pairs,
        lengths=[1 - test_fraction, test_fraction],
        generator=th.Generator().manual_seed(seed),
    )
    train_rewards, test_rewards = random_split(
        reward_pairs,
        lengths=[1 - test_fraction, test_fraction],
        generator=th.Generator().manual_seed(seed),
    )
    train_observations, test_observations = random_split(
        observation_pairs,
        lengths=[1 - test_fraction, test_fraction],
        generator=th.Generator().manual_seed(seed),
    )
    return (
        th.stack(list(train_activations)),
        th.stack(list(test_activations)),
        th.stack(list(train_returns)),
        th.stack(list(test_returns)),
        np.array(train_rewards),
        np.array(test_rewards),
        np.array(train_observations),
        np.array(test_observations),
    )
