import shutil
import pathlib
from pathlib import Path
import pickle
import torch.nn as nn
from torch.utils import data as data_th
from torch.utils.data import DataLoader
import torch as th
import copy
import numpy as np
import random

# from imitation.data.rollout import generate_trajectories, make_sample_until
# from imitation.data.serialize import save, load_with_rewards
import argparse
from agents.common import get_env, Agent, preprocess
from utils import strtobool
from huggingface_hub import hf_hub_download
from tqdm import tqdm, trange
from dataclasses import dataclass
from nicehooks import nice_hooks

HF_PATH = Path("hf_models")
DATASET_PATH = Path("datasets")


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
        with tqdm(total=steps) as pbar:
            while True:
                with th.no_grad():
                    action = policy(obs)
                next_obs, reward, done, infos = self.env.step(action)
                traj["obs"].append(obs)
                traj["actions"].append(action)
                traj["rewards"].append(np.array(reward))
                i += 1
                i_episode += 1
                obs = next_obs
                if i % (steps // 10) == 0:
                    pbar.update(i)
                if done.any():  # TODO: change this for multi envs
                    i_episode = 0
                    traj["terminal"] = True
                    trajs.append(Trajectory(**traj))
                    traj = {
                        "obs": [],
                        "actions": [],
                        "rewards": [],
                        "terminal": False,
                    }
                    obs = self.env.reset()
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
    # trajectories = generate_trajectories(
    #    model,
    #    env,
    #    make_sample_until(min_episodes=num_episodes),
    #    np.random.default_rng(seed),
    # )
    return trajectories


def load_data(dataset_path):
    """Load trajectories"""
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)
    observation_pairs = sum([traj.obs for traj in dataset], [])
    return observation_pairs


def load_model(model_path, env, device):
    agent = Agent(env)
    agent.load(model_path)
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


def get_hidden_activations_dataset(module, device, dataset):
    """Calculate hidden layer activations for given pair dataset."""
    hidden_act_dataset = []
    for pair in tqdm(dataset):
        th_pair = preprocess(th.tensor(pair, dtype=th.float).to(device))
        with th.no_grad():
            _, activations = nice_hooks.run(module, th_pair, return_activations=True)
            hidden_act_dataset.append(activations.to("cpu"))
    return hidden_act_dataset


class ValueProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 1)

    def forward(self, x):
        h = self.linear1(x)
        return th.tanh(h)


class CCS:
    """Implementation of contrast consistent search for value functions."""

    def __init__(
        self,
        env,
        module,
        dataset_path,
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
        self.env = env
        self.module = module
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

        if verbose:
            print("get hidden activations")
        activations_path: Path = dataset_path.with_suffix('') / "activations.pt"
        if activations_path.exists():
            if verbose:
                print(f"Found cached activations at: {activations_path}")
            activation_pairs = th.load(activations_path)
        else:
            if verbose:
                print(f"No cached activations. Computing them...")
            observation_pairs = load_data(dataset_path)
            activation_pairs = get_hidden_activations_dataset(
                module, device, observation_pairs
            )
            activations_path.parent.mkdir(parents=True, exist_ok=True)
            th.save(activation_pairs, activations_path)
        self.train_activations, self.test_activations = data_th.random_split(
            activation_pairs,
            lengths=[val_fraction, 1 - val_fraction],
            generator=th.Generator().manual_seed(seed),
        )

    def initialize_probe(self, layer_name):
        dim = self.train_activations[0][layer_name].shape[-1]
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
        informative_loss = ((1 - value_1.abs()) ** 2 + (1 - value_2.abs()) ** 2).mean(0)
        consistent_loss = ((value_1 + value_2) ** 2).mean(0)
        return consistent_loss + informative_loss

    def get_return_metrics(self):
        """Computes metrics of value probe against trajectory returns."""
        num_trajs = len(self.test_activations)
        traj_lengths = [
            len(self.test_activations[i][0][layer_name]) for i in range(num_trajs)
        ]
        x0 = (
            th.cat(
                [self.test_activations[i][0][layer_name] for i in range(num_trajs)],
                axis=0,
            )
            .detach()
            .to(self.device)
        )
        x1 = (
            th.cat(
                [self.test_activations[i][1][layer_name] for i in range(num_trajs)],
                axis=0,
            )
            .detach()
            .to(self.device)
        )
        # compute returns from trajectory data assuming gamma=1
        return_1 = (
            th.cat(
                [
                    th.flip(
                        th.cumsum(
                            th.flip(th.tensor(self.test_data[i].rews), dims=(0,)),
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
            th.cat(
                [
                    th.flip(
                        th.cumsum(
                            th.flip(
                                -th.tensor(self.test_data[i].rews), dims=(0,)
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

    def train(self, layer_name):
        """Train a single probe on the given hidden layer."""
        x = (
            th.cat(
                [pair[layer_name].unsqueeze(0) for pair in self.train_activations],
                axis=0,
            )
            .detach()
            .to(self.device)
        )
        batch_size = len(x) if self.batch_size == -1 else self.batch_size
        dataloader = DataLoader(x, batch_size, shuffle=True)

        # set up optimizer
        optimizer = th.optim.AdamW(
            self.probe.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Start training (full batch)
        for epoch in trange(self.num_epochs):
            for batch in dataloader:
                x0_batch = batch[:, 0]
                x1_batch = batch[:, 1]
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
        for train_num in trange(self.num_tries):
            self.initialize_probe(layer_name)
            loss = self.train(layer_name)
            print(f"Train repetition {train_num}, final train loss = {loss:.5f}")
            if loss < best_loss:
                print(f"New best loss!")
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss
        return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run CCS on a given model and environment.")
    parser.add_argument(
        "--env_id",
        type=str,
        default="pong_v3",
        help="Environment name",
    )
    parser.add_argument("--model_path", type=str, help="Path to model", required=True)
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use. [cuda, cpu]",
        default="cuda",
    )
    parser.add_argument(
        "--from_hf",
        type=lambda x: bool(strtobool(x)),
        help="Whether to load from Huggingface.",
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        help="Whether to capture videos from the data collection",
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--layer_name", type=str, help="The layer to run ccs on", default=""
    )
    args = parser.parse_args()
    # TODO? support multiple envs
    args.num_envs = 2
    env = get_env(args, "ccs")
    # f = env.reset
    # env.reset = lambda : f()[0]
    # print(f"type: {env.reset()[0].shape}")
    # exit()
    args.model_name = args.model_path
    if args.from_hf:
        hf_hub_download(
            repo_id="Butanium/selfplay_ppo_pong_v3_pettingzoo_cleanRL",
            filename=args.model_path,
            local_dir="hf_models",
        )
        args.model_path = HF_PATH / args.model_path

    data_save_path = DATASET_PATH / args.model_name / "selfplay.pkl"
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
    
    # Train multiple CCS probes on specified layer
    model = load_model(args.model_path, env, args.device)
    ccs = CCS(
        env, model.critic_network, data_save_path, device=args.device, verbose=True
    )
    ccs.repeated_train(args.layer_name)

    # Evaluate probe against trajectory returns
    # print(
    #     "Best probe CCS eval metrics: v1_loss={:.5f}, v2_loss={:.5f}, avg_value_sum={:.5f}, avg_return_sum={:.5f}".format(
    #         *ccs.get_return_metrics()
    #     )
    # )
