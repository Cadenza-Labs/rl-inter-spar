import shutil
import os
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from sb3_extract import (
    obs_to_tensor,
    sb3_preprocess,
    get_activations,
    get_extractor_activation,
)
import torch
import torch.nn as nn
from torch.utils import data as data_th
import copy
import numpy as np
from imitation.data.rollout import generate_trajectories, make_sample_until
from imitation.data.serialize import save, load_with_rewards


def generate_dataset(
    env_name, model_name, num_episodes, max_episode_length, num_envs, seed
):
    """Generate trajectory data using the given environment and model."""
    model = load_model(model_name)
    env = make_atari_env(
        env_name,
        n_envs=num_envs,
        env_kwargs=dict(max_num_frames_per_episode=max_episode_length),
        wrapper_kwargs=dict(noop_max=0),
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    trajectories = generate_trajectories(
        model,
        env,
        make_sample_until(min_episodes=num_episodes),
        np.random.default_rng(seed),
    )
    return trajectories


def load_data(model_name, val_fraction, seed):
    """Load and split data of given model."""
    dataset = load_with_rewards(f"datasets/{model_name}")
    val_length = int(len(dataset) * val_fraction)
    assert val_length > 0, "Validation fraction is too low!"
    train_length = len(dataset) - val_length
    train_dataset, val_dataset = data_th.random_split(
        dataset,
        lengths=[train_length, val_length],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_dataset, val_dataset


def load_model(model_name):
    """Load sb3 model."""
    algo_name = model_name.split("-")[0]
    algo_map = {"ppo": PPO, "dqn": DQN}
    return algo_map[algo_name].load(f"agents/{model_name}", device="cpu")


def switch_player_perspective(env_name, obs):
    """Switch perspective of observation to other player."""
    if env_name == "PongNoFrameskip-v4":
        # TODO mirroring the observation shouldn't mirror the score!
        return np.flip(obs, axis=-1).copy()
    else:
        raise ValueError("Unsupported environment for switching perspectives!")


def get_hidden_activations_dataset(env_name, model, dataset):
    """Calculate hidden layer activations for given trajectory dataset."""
    hidden_act_dataset = []
    for traj in dataset:
        obs_1 = traj.obs
        th_obs_1 = obs_to_tensor(model, obs_1)
        obs_2 = switch_player_perspective(env_name, obs_1)
        th_obs_2 = obs_to_tensor(model, obs_2)
        _, activations_1 = get_activations(
            model.policy.features_extractor, sb3_preprocess(model, th_obs_1)
        )
        _, activations_2 = get_activations(
            model.policy.features_extractor, sb3_preprocess(model, th_obs_2)
        )
        hidden_act_dataset.append((activations_1, activations_2))
    return hidden_act_dataset


class ValueProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 1)

    def forward(self, x):
        h = self.linear1(x)
        return torch.tanh(h)


class CCS:
    """Implementation of contrast consistent search for value functions."""

    def __init__(
        self,
        env_name,
        model_name,
        num_epochs=1000,
        num_tries=10,
        learning_rate=1e-3,
        batch_size=-1,
        verbose=False,
        device="cuda",
        weight_decay=0.01,
        var_normalize=False,
        val_fraction=0.2,
        seed=42,
    ):
        self.env_name = env_name
        self.model_name = model_name
        # training
        self.var_normalize = var_normalize
        self.num_epochs = num_epochs
        self.num_tries = num_tries
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.seed = seed

        self.model = load_model(model_name)
        self.train_data, self.test_data = load_data(model_name, val_fraction, seed)

        self.train_activations = get_hidden_activations_dataset(
            env_name, self.model, self.train_data
        )
        self.test_activations = get_hidden_activations_dataset(
            env_name, self.model, self.test_data
        )

    def initialize_probe(self, layer_name):
        dim = self.train_activations[0][0][layer_name].shape[-1]
        self.probe = ValueProbe(dim)
        self.probe.to(self.device)

    def normalize(self, activations):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = activations - activations.mean(axis=0, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=0, keepdims=True)

        return normalized_x

    def get_loss(self, value_1, value_2):
        """Returns the CCS loss for two values each of shape (n,1) or (n,)."""
        # TODO add more loss options
        informative_loss = (torch.min((1 - value_1) ** 2, (1 - value_2) ** 2)).mean(0)
        consistent_loss = ((value_1 + value_2) ** 2).mean(0)
        return consistent_loss + informative_loss

    def get_return_metrics(self):
        """Computes metrics of value probe against trajectory returns."""
        num_trajs = len(self.test_activations)
        traj_lengths = [
            len(self.test_activations[i][0][layer_name]) for i in range(num_trajs)
        ]
        x0 = (
            torch.cat(
                [self.test_activations[i][0][layer_name] for i in range(num_trajs)],
                axis=0,
            )
            .detach()
            .to(self.device)
        )
        x1 = (
            torch.cat(
                [self.test_activations[i][1][layer_name] for i in range(num_trajs)],
                axis=0,
            )
            .detach()
            .to(self.device)
        )
        # compute returns from trajectory data assuming gamma=1
        return_1 = (
            torch.cat(
                [
                    torch.flip(
                        torch.cumsum(
                            torch.flip(torch.tensor(self.test_data[i].rews), dims=(0,)),
                            dim=0,
                        ),
                        dims=(0,),
                    )
                    for i in range(num_trajs)
                ],
                axis=0,
            )
            .detach()
            .to(self.device)
        )
        # TODO here we make the assumption that player 2 reward is the negative of player 1
        return_2 = (
            torch.cat(
                [
                    torch.flip(
                        torch.cumsum(
                            torch.flip(
                                -torch.tensor(self.test_data[i].rews), dims=(0,)
                            ),
                            dim=0,
                        ),
                        dims=(0,),
                    )
                    for i in range(num_trajs)
                ],
                axis=0,
            )
            .detach()
            .to(self.device)
        )
        with torch.no_grad():
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

    def train(self, layer_name):
        """Train a single probe on the given hidden layer."""
        num_trajs = len(self.train_activations)
        x0 = (
            torch.cat(
                [self.train_activations[i][0][layer_name] for i in range(num_trajs)],
                axis=0,
            )
            .detach()
            .to(self.device)
        )
        x1 = (
            torch.cat(
                [self.train_activations[i][1][layer_name] for i in range(num_trajs)],
                axis=0,
            )
            .detach()
            .to(self.device)
        )
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]

        # set up optimizer
        optimizer = torch.optim.AdamW(
            self.probe.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        num_batches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.num_epochs):
            for j in range(num_batches):
                x0_batch = x0[j * batch_size : (j + 1) * batch_size]
                x1_batch = x1[j * batch_size : (j + 1) * batch_size]

                # probe
                v0, v1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(v0, v1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()

    def repeated_train(self, layer_name):
        """Repeatedly train probes on given hidden layer."""
        best_loss = np.inf
        for train_num in range(self.num_tries):
            self.initialize_probe(layer_name)
            loss = self.train(layer_name)
            print(f"Train repetition {train_num}, final train loss = {loss:.5f}")
            if loss < best_loss:
                print(f"New best loss!")
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss
        return best_loss


if __name__ == "__main__":
    env_name = "PongNoFrameskip-v4"
    model_name = f"ppo-{env_name}"
    layer_name = "cnn"

    # Generate and save dataset if not exists
    data_save_path = f"datasets/{model_name}"
    if not os.path.isdir(data_save_path):
        # shutil.rmtree(data_save_path)
        trajs = generate_dataset(
            env_name,
            model_name,
            num_episodes=10,
            max_episode_length=500,
            num_envs=4,
            seed=42,
        )
        save(data_save_path, trajs)

    # Train multiple CCS probes on specified layer
    ccs = CCS(env_name, model_name)
    ccs.repeated_train(layer_name)

    # Evaluate probe against trajectory returns
    print(
        "Best probe CCS eval metrics: v1_loss={:.5f}, v2_loss={:.5f}, avg_value_sum={:.5f}, avg_return_sum={:.5f}".format(
            *ccs.get_return_metrics()
        )
    )
