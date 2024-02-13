from pathlib import Path
import pickle
import torch.nn.functional as F
from sklearn.linear_model import Ridge
import torch.nn as nn
from torch.utils import data as data_th
from torch.utils.data import DataLoader
import torch as th
import numpy as np
import scipy
import random
import time
from itertools import product
import csv
from einops import rearrange
from warnings import warn

import argparse
from agents.common import get_env, Agent, preprocess
from utils import strtobool
from huggingface_hub import hf_hub_download
from tqdm import tqdm, trange
from dataclasses import dataclass
from nicehooks import nice_hooks
from probe_visualization import ProbeMonitor

HF_PATH = Path("hf_models")
DATASET_PATH = Path("datasets")

WEIGHT_DECAY = 0.01
VAL_FRACTION = 0.2
GAMMA = 0.99
SEED = 44


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
def extract(
    layer_name, dataset_path, verbose, device, val_fraction, gamma, seed, normalize=True
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
    train_activations, test_activations = data_th.random_split(
        activation_pairs,
        lengths=[val_fraction, 1 - val_fraction],
        generator=th.Generator().manual_seed(seed),
    )
    train_returns, test_returns = data_th.random_split(
        return_pairs,
        lengths=[val_fraction, 1 - val_fraction],
        generator=th.Generator().manual_seed(seed),
    )
    train_rewards, test_rewards = data_th.random_split(
        reward_pairs,
        lengths=[val_fraction, 1 - val_fraction],
        generator=th.Generator().manual_seed(seed),
    )
    train_observations, test_observations = data_th.random_split(
        observation_pairs,
        lengths=[val_fraction, 1 - val_fraction],
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


@th.no_grad()
def supervised_prediction(lr, obs, module, layer_name):
    """Predict the value of an observation using a supervised model."""

    obs = preprocess(obs)
    _, activations = nice_hooks.run(module, obs, return_activations=True)
    return lr.predict(activations[layer_name].cpu())


def train_supervised(
    dataset_path, model, layer_name, verbose, device, val_fraction, gamma, seed
):
    """Linear classifier trained with supervised learning."""
    (
        train_activations,
        test_activations,
        train_returns,
        test_returns,
        train_rewards,
        test_rewards,
        train_observations,
        test_observations,
    ) = extract(
        layer_name,
        dataset_path,
        verbose,
        device,
        val_fraction,
        gamma,
        seed,
        normalize=False,
    )

    x_train = rearrange(train_activations, "n p d -> (n p) d")

    # TODO: Put an own function
    # Generate value predictions from the value network for each observation in the dataset
    observations = rearrange(
        th.tensor(np.array(train_observations, dtype=np.uint8), dtype=th.float),
        "n p h w f -> (n p) h w f",
    ).to(device)
    y_train = model.get_value(observations).squeeze().detach().cpu().numpy()

    # lr = LRProbe.train(x_train, values, device=device)
    lr = Ridge()
    lr.fit(x_train.cpu(), y_train)
    # print("Linear regression accuracy (test): {}".format(lr.score(x_test.cpu(), y_test.cpu())))
    # print("Linear regression accuracy (train): {}".format(lr.score(x_train.cpu(), y_train.cpu())))

    return lr

class Probe(nn.Module):
    @th.no_grad()
    def calibrate(self, hidden_activations, rewards):
        """Calibrate the probe
        Args:
            hidden_activations: hidden activations of shape (n, d)
            rewards: rewards of shape (n, 1)
        """
        # Compute the loss for forward and -forward and choose the sign that minimizes the loss
        loss = nn.MSELoss()
        positive_loss = loss(self.forward(hidden_activations), rewards)
        negative_loss = loss(-self.forward(hidden_activations), rewards)
        if positive_loss < negative_loss:
            self.sign = 1
        else:
            self.sign = -1

class LRProbe(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        # Define a linear layer followed by tanh activation
        self.net = nn.Sequential(
            nn.Linear(d_in, 1, bias=True),
            nn.Tanh()  # Add Tanh activation to ensure output is between -1 and 1
        )

    def forward(self, x, iid=None):
        # Forward pass will output values constrained between -1 and 1
        return self.net(x).squeeze(-1)

    def pred(self, x, iid=None):
        # For regression tasks, this could return raw predictions directly
        # Adjustments or post-processing can be applied based on specific needs
        return self(x)
    
    @staticmethod
    def train(self, probe):
        """Train a single probe on its layer."""
        batch_size = (
            len(self.train_activations) if self.batch_size == -1 else self.batch_size
        )
        dataloader = DataLoader(self.train_activations, batch_size, shuffle=True)

        # set up optimizer
        optimizer = th.optim.AdamW(
            probe.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Start training (full batch)
        for batch in dataloader:
            v1 = probe(batch)

            # get the corresponding loss
            loss = self.get_loss(v1)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return probe
    
    @property
    def direction(self):
        # Access the weights of the linear layer
        return self.net[0].weight.data


class MLPProbe(Probe):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x.flatten(start_dim=1)))
        o = self.linear2(h)
        return th.sigmoid(o)

    
class LinearProbe(Probe):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.sign = 1

    def forward(self, x):
        h = self.linear(x.flatten(start_dim=1))
        return th.tanh(h) * self.sign
    
    
class CCS:
    """Implementation of contrast consistent search for value functions."""

    def __init__(
        self,
        env,
        model,
        layer_name,
        dataset_path,
        num_epochs=1000,
        num_tries=10,
        learning_rate=1e-3,
        informative_loss_weight=1.0,
        batch_size=-1,
        verbose=False,
        device="cuda",
        weight_decay=WEIGHT_DECAY,
        var_normalize=False,
        val_fraction=VAL_FRACTION,
        gamma=GAMMA,
        seed=SEED,
        load=True,
        linear=True
    ):
        self.env = env
        self.model = model
        self.layer_name = layer_name
        # training
        self.var_normalize = var_normalize
        self.num_epochs = num_epochs
        self.num_tries = num_tries
        self.learning_rate = learning_rate
        self.informative_loss_weight = informative_loss_weight
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.gamma = gamma
        self.seed = seed
        self.dataset_path = dataset_path.with_suffix("")
        self.best_probe = None
        self.probe_path = (
            self.dataset_path
            / "probes"
            / f"{self.layer_name}_s{seed}_nt{num_tries}_ne{num_epochs}_wd{weight_decay}_inflossw{inf_loss_weight : g}.pt"
        )
        self.linear = linear

        if load and self.probe_path.exists():
            print(f"Loading probe from {self.probe_path}")
            # We evaluate the model on the environment to get the observation shape
            obs = preprocess(th.tensor(env.reset(), dtype=th.float, device=self.device))
            _, self.train_activations = nice_hooks.run(
                self.model, obs, return_activations=True
            )
            self.train_activations = self.train_activations[self.layer_name].unsqueeze(
                0
            )
            self.best_probe = self.initialize_probe()
            self.best_probe.load_state_dict(th.load(self.probe_path))
            self.best_probe.to(self.device)
            self.train_activations = None
        (
            self.train_activations,
            self.test_activations,
            self.train_returns,
            self.test_returns,
            self.train_rewards,
            self.test_rewards,
            self.train_observations,
            self.test_observations,
        ) = extract(
            layer_name,
            dataset_path,
            verbose,
            device,
            val_fraction,
            gamma,
            seed,
            normalize=False,
        )

    def initialize_probe(self):
        dim = self.train_activations[0][0].flatten().shape[0]
        
        
        if self.linear:
            print("Probe is linear")
            return LinearProbe(dim).to(self.device)
        else:
            print("Probe is MLP")
            return MLPProbe(dim).to(self.device)
        

    def get_loss(self, value_1, value_2):
        """Returns the CCS loss for two values each of shape (n,1) or (n,)."""
        # TODO add more loss options
        informative_loss = ((1 - value_1.abs()) ** 2 + (1 - value_2.abs()) ** 2).mean(0)
        consistent_loss = ((value_1 + value_2) ** 2).mean(0)
        return consistent_loss + informative_loss * self.informative_loss_weight

    def get_return_metrics(self):
        """Computes metrics of value probe against trajectory returns."""
        x0 = self.test_activations[:, 0].detach().to(self.device)
        x1 = self.test_activations[:, 1].detach().to(self.device)
        # compute returns from trajectory data assuming gamma=1
        return_1 = self.test_returns[:, 0].detach().to(self.device)
        return_2 = self.test_returns[:, 1].detach().to(self.device)
        with th.no_grad():
            value_1, value_2 = self.best_probe(x0), self.best_probe(x1)
        avg_value_sum = (value_1 + value_2).mean()
        avg_return_sum = (return_1 + return_2).mean()
        value_1_loss = ((value_1 - return_1) ** 2).mean()
        value_2_loss = ((value_2 - return_2) ** 2).mean()
        return (
            value_1_loss.cpu().item(),
            value_2_loss.cpu().item(),
            avg_value_sum.cpu().item(),
            avg_return_sum.cpu().item(),
        )

    @th.no_grad()
    def evaluate(self, probe, dataloader):
        """
        Evaluate a probe on a given dataset
        """
        train_loss = 0
        for batch in dataloader:
            x0_batch = batch[:, 0]
            x1_batch = batch[:, 1]
            v0, v1 = probe(x0_batch), probe(x1_batch)
            train_loss += self.get_loss(v0, v1)
        return train_loss

    @th.no_grad()
    def elicit(self, obs):
        """
        Elicit a value from the model using `self.best_probe` for a given observation
        """
        obs = preprocess(obs)
        _, activations = nice_hooks.run(self.model, obs, return_activations=True)
        return self.best_probe(activations[self.layer_name])

    @th.no_grad()
    def calibrate(self):
        """
        Calibrate the best probe
        """
        if (self.train_rewards == 0).all():
            warn(
                "All rewards are zero. The probe will not be calibrated. Consider using a different dataset."
            )
            return
        rewards = th.tensor(self.train_rewards, dtype=th.float).to(self.device)
        self.best_probe.calibrate(
            self.train_activations.reshape((-1, self.train_activations.shape[-1])),
            rewards.reshape(-1, 1),
        )

    def train(self, probe):
        """Train a single probe on its layer."""
        batch_size = (
            len(self.train_activations) if self.batch_size == -1 else self.batch_size
        )
        dataloader = DataLoader(self.train_activations, batch_size, shuffle=True)

        # set up optimizer
        optimizer = th.optim.AdamW(
            probe.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Start training (full batch)
        for epoch in trange(self.num_epochs):
            for batch in dataloader:
                x0_batch = batch[:, 0]
                x1_batch = batch[:, 1]
                # probe
                v0, v1 = probe(x0_batch), probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(v0, v1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.evaluate(probe, dataloader)

    def repeated_train(self, save=True):
        """Repeatedly train probes on given hidden layer."""
        best_loss = np.inf
        test_loss_best = np.inf
        batch_size = (
            len(self.test_activations) if self.batch_size == -1 else self.batch_size
        )
        test_set = DataLoader(self.test_activations, batch_size)
        for train_num in trange(self.num_tries):
            probe = self.initialize_probe()
            train_loss = self.train(probe)
            test_loss = self.evaluate(probe, test_set)
            print(
                f"Train repetition {train_num}, final train loss = {float(train_loss):.5f}, test loss {float(test_loss)}"
            )
            if train_loss < best_loss:
                print(f"New best loss!")
                self.best_probe = probe
                best_loss = train_loss
                test_loss_best = test_loss
        if save:
            self.probe_path.parent.mkdir(parents=True, exist_ok=True)
            th.save(self.best_probe.state_dict(), self.probe_path)
            # Save metadata
            metadata_path = self.probe_path.with_suffix(".csv")
            with open(metadata_path, "w") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "layer_name",
                        "train_loss",
                        "test_loss",
                        "num_epochs",
                        "num_tries",
                        "learning_rate",
                        "informative_loss_weight",
                        "batch_size",
                        "weight_decay",
                        "val_fraction",
                        "seed",
                        "var_normalize",
                    ]
                )
                writer.writerow(
                    [
                        self.layer_name,
                        best_loss.item(),
                        test_loss_best.item(),
                        self.num_epochs,
                        self.num_tries,
                        self.learning_rate,
                        self.informative_loss_weight,
                        self.batch_size,
                        self.weight_decay,
                        self.val_fraction,
                        self.seed,
                        self.var_normalize,
                    ]
                )
        return best_loss


def parse_args():
    parser = argparse.ArgumentParser("Run CCS on a given model and environment.")
    env_group = parser.add_argument_group("Environment and model")
    env_group.add_argument(
        "--env-id",
        type=str,
        default="pong_v3",
        help="Environment name",
    )
    env_group.add_argument(
        "--model-path", type=str, help="Path to model", required=True
    )
    env_group.add_argument(
        "--device",
        type=str,
        help="Device to use. [cuda, cpu, auto]",
        default="auto",
    )
    env_group.add_argument(
        "--from-hf",
        type=lambda x: bool(strtobool(x)),
        help="Whether to load from Huggingface.",
        nargs="?",
        const=True,
        default=True,
    )
    env_group.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        help="Whether to capture videos from the data collection",
        nargs="?",
        const=True,
        default=False,
    )

    ccs_group = parser.add_argument_group("CCS")
    ccs_group.add_argument(
        "--modules",
        help="The modules of the model to run ccs on (critic_network | actor_network)",
        nargs="*",
        default=[],
    )
    ccs_group.add_argument(
        "--layer-indicies",
        help="The indicies of the module layer we want to run ccs on",
        type=int,
        nargs="*",
        default=[],
    )
    ccs_group.add_argument(
        "--best-of-n",
        help="The number of probes to train and evaluate, keeping the best one",
        type=int,
        default=10,
    )
    ccs_group.add_argument(
        "--informative-loss-weights",
        help="The weights of the informative loss",
        type=float,
        nargs="*",
        default=[1.0],
    )
    ccs_group.add_argument(
        "--load-best-probe",
        help="Whether to load the best probe from the dataset if it exists",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
    )
    ccs_group.add_argument(
        "--save-probe",
        help="Whether to save the best probe",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=True,
    )
    ccs_group.add_argument(
        "--linear",
        help="Whether to use a linear probe (True) or a MLP probe (False)",
        type=lambda x: bool(strtobool(x)),
        default=True,
    )
    ccs_group.add_argument(
        "--skip-ccs-probe-training",
        help="Whether to skip the CCS probe training",
        default=False,
        action="store_true",
    )

    vis_group = parser.add_argument_group(
        "Visualization", "Parameters for the probe visualization across time"
    )
    vis_group.add_argument(
        "--rounds-to-record",
        help="The number of rounds to record",
        type=int,
        default=3,
    )
    vis_group.add_argument(
        "--max-num-steps",
        help="The maximum number of steps to record",
        type=int,
        default=10000,
    )
    vis_group.add_argument(
        "--max-video-length",
        help="The maximum length of the recorded videos",
        type=int,
        default=6000,
    )
    vis_group.add_argument(
        "--interactive",
        help="Whether to run in interactive mode",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
    )
    vis_group.add_argument(
        "--record-probe-videos",
        help="Whether to record a videos of each probe value across time",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
    )
    vis_group.add_argument(
        "--record-video-with-all-probes",
        help="Whether to record a video with all probes values across time",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
    )
    vis_group.add_argument(
        "--record-agent-value",
        help="Whether to record the agents values across time in the probe video",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
    )
    vis_group.add_argument(
        "--sliding-window",
        help="The size of the sliding window for the probe visualization",
        type=int,
        default=200,
    )
    return parser.parse_args()


def monitor_probes(
    args, env, agent1, agent2, layers, fn_grouped_by_probe, metrics, video_path
):
    if (
        args.interactive
        or args.record_probe_videos
        or args.record_video_with_all_probes
    ):
        monitor = ProbeMonitor(
            env,
            agent1,
            agent2,
            metrics,
            args.device,
        )
        print("\nMonitoring probe...")
        monitor.run(
            args.rounds_to_record,
            args.max_num_steps,
        )
        if args.interactive:
            print("Starting interactive visualization...")
            monitor.interactive_visualization(args.sliding_window)
            print("Interactive visualization finished.")
        if args.record_video_with_all_probes:
            print("Recording video with all probes...")
            monitor.save_video(
                metrics.keys(),
                video_path / "_".join(f"{m}.{l}" for m, l in layers),
                file_name=f"ccs_eval_{int(time.time())}",
                sliding_window=args.sliding_window,
                max_video_length=args.max_video_length,
            )
        if args.record_probe_videos:
            print("Recording a video for each probe...")
            # Create a video for each probe
            # Notes: A lot of computation is duplicated here. We could avoid this by refactoring playground
            for probe_name, probe_fn_dict in fn_grouped_by_probe.items():
                extra_metrics = (
                    ["Right player value", "Left player value"]
                    if args.record_agent_value
                    else []
                )
                monitor.save_video(
                    list(probe_fn_dict.keys()) + extra_metrics,
                    video_path / probe_name,
                    file_name=f"ccs_eval_{int(time.time())}"
                    + ("_pv" if args.record_agent_value else ""),
                    sliding_window=args.sliding_window,
                    max_video_length=args.max_video_length,
                )


if __name__ == "__main__":
    args = parse_args()
    # TODO? support multiple envs
    args.num_envs = 2
    env = get_env(args, "ccs")
    args.model_name = args.model_path
    if args.from_hf:
        hf_hub_download(
            repo_id="Butanium/selfplay_ppo_pong_v3_pettingzoo_cleanRL",
            filename=args.model_path,
            local_dir="hf_models",
        )
        args.model_path = HF_PATH / args.model_path

    data_save_path = DATASET_PATH / args.model_name / "selfplay.pkl"
    if args.device == "auto":
        args.device = "cuda" if th.cuda.is_available() else "cpu"
        print(f"Using device: {args.device}")
    if not data_save_path.exists():
        print("Generating dataset...")
        trajs = generate_dataset(
            env,
            args.model_path,
            num_episodes=1,
            max_episode_length=10000,
            # num_envs=4,
            seed=42,
            device=args.device,
        )
        # save(data_save_path, trajs)
        data_save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(data_save_path, "wb") as file:
            pickle.dump(trajs, file)

    model = load_model(args.model_path, env, args.device)

    # Train multiple CCS probes on specified layer
    if args.layer_indicies == []:
        args.layer_indicies = range(
            len(model.actor_network)
        )  # actor and critic network have same number of layers
    if args.modules == []:
        args.modules = ["actor_network", "critic_network"]
    layers = list(product(args.modules, args.layer_indicies))
    layer_names = [f"{m}.{l}" for m, l in layers]
    probes = []
    probes_fn_dict = {}
    fn_grouped_by_probe = {}
    if not args.skip_ccs_probe_training:
        for inf_loss_weight in args.informative_loss_weights:
            for layer_name in layer_names:
                print(
                    "\n\n"
                    "===================================\n"
                    f"Training CCS probe for {layer_name}\n"
                    f"informative loss = {inf_loss_weight}\n"
                    f"Probe is linear: {'True' if args.linear else 'False'}\n"
                    "==================================="
                )
                ccs = CCS(
                    env,
                    model,
                    layer_name,
                    data_save_path,
                    informative_loss_weight=inf_loss_weight,
                    device=args.device,
                    num_tries=args.best_of_n,
                    load=args.load_best_probe,
                    linear=args.linear,
                    # verbose=True,
                )
                if ccs.best_probe is None:
                    ccs.repeated_train(save=args.save_probe)
                ccs.calibrate()
                probes.append(ccs)
                inf_loss_string = f"with inf loss weight {inf_loss_weight :.2g}"
                probe_dict = {
                    f"Right player CCS probe on {layer_name} {inf_loss_string}": lambda obs, ccs=ccs: ccs.elicit(
                        obs[:1]
                    ).item(),
                    f"Left player CCS probe on {layer_name} {inf_loss_string}": lambda obs, ccs=ccs: ccs.elicit(
                        obs[1:2]
                    ).item(),
                }
                probes_fn_dict.update(probe_dict)
                fn_grouped_by_probe[
                    f"{layer_name}_inf_loss_weight_{inf_loss_weight: g}"
                ] = probe_dict
    for layer_name in layer_names:
        print(f"\n\n====== Training Supervised probe for {layer_name} ======")
        supervised_probe = train_supervised(
            dataset_path=data_save_path,
            model=model,
            layer_name=layer_name,
            verbose=False,
            device=args.device,
            val_fraction=WEIGHT_DECAY,
            gamma=GAMMA,
            seed=SEED,
        )
        probes.append(supervised_probe)
        probe_dict = {
            f"Right supervised probe on {layer_name}": lambda obs, supervised_probe=supervised_probe: supervised_prediction(
                supervised_probe, obs[:1], model, layer_name
            ),
            f"Left supervised probe on {layer_name}": lambda obs, supervised_probe=supervised_probe: supervised_prediction(
                supervised_probe, obs[1:2], model, layer_name
            ),
        }
        probes_fn_dict.update(probe_dict)
        fn_grouped_by_probe[f"{layer_name}_supervised_probe"] = probe_dict

    metrics = {
        "Right player value": lambda obs: model.get_value(obs[:1]).item(),
        "Left player value": lambda obs: model.get_value(obs[1:2]).item(),
    }
    metrics.update(probes_fn_dict)
    video_path = Path("videos") / "ccs_eval" / args.model_name
    model.name = args.model_name.replace("/", "_")
    monitor_probes(
        args, env, model, model, layers, fn_grouped_by_probe, metrics, video_path
    )
    if not args.skip_ccs_probe_training:
        # Evaluate probe against trajectory returns
        print(
            "Best probe CCS eval metrics: v1_loss={:.5f}, v2_loss={:.5f}, avg_value_sum={:.5f}, avg_return_sum={:.5f}".format(
                *ccs.get_return_metrics()
            )
        )
