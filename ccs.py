from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torch as th
import numpy as np
import csv
from warnings import warn

from agents.common import preprocess
from tqdm import trange
from nicehooks import nice_hooks
from ccs_utils import extract_activations

HF_PATH = Path("hf_models")
DATASET_PATH = Path("datasets")

WEIGHT_DECAY = 0.01
VAL_FRACTION = 0.2
GAMMA = 0.99
SEED = 44


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.sign = 1

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


class MLPProbe(Probe):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x.flatten(start_dim=1)))
        o = self.linear2(h)
        return th.tanh(o) * self.sign


class LinearProbe(Probe):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

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
        linear=True,
        normalize=False,
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
            / f"{self.layer_name}_s{seed}_nt{num_tries}_ne{num_epochs}_wd{weight_decay}_inflossw{informative_loss_weight : g}.pt"
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
        ) = extract_activations(
            model,
            layer_name,
            dataset_path,
            verbose,
            device,
            val_fraction,
            gamma,
            seed,
            normalize=normalize,
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
            self.train_activations.reshape((-1, *self.train_activations.shape[2:])),
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
