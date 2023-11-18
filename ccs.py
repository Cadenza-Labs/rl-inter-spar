from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import nicehooks.nice_hooks as nice_hooks
from sb3_extract import obs_to_tensor, sb3_preprocess, get_activations, get_extractor_activation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as data_th
import copy
import numpy as np
from typing import Sequence
from imitation.data.rollout import generate_trajectories, make_sample_until



def generate_dataset(env_name, model_name, num_episodes, num_envs, seed):
    model = load_model(model_name)
    env = make_atari_env(env_name, n_envs=num_envs)
    env = VecFrameStack(env, n_stack=4)
    # TODO add model or env wrapper to align obs shapes
    trajectories = generate_trajectories(model, env, make_sample_until(min_episodes=num_episodes), np.random.default_rng(seed))
    return trajectories


def load_data(model_name, val_fraction, seed):
    dataset = load_dataset(model_name)
    val_length = int(len(dataset) * val_fraction)
    train_length = len(dataset) - val_length
    train_dataset, val_dataset = data_th.random_split(
        dataset,
        lengths=[train_length, val_length],
        generator=torch.Generator().manual_seed(42),
    )
    return train_dataset, val_dataset

def load_model(model_name):
    return PPO.load(f'agents/{model_name}', device='cpu')

def get_preprocessed_value_estimate(model, obs):
    pass

def get_hidden_activations(model, obs_a1, obs_a2):
    """Given a pair of observations return activations for all layers."""
    result, activation_a1 = get_extractor_activation(model, obs_to_tensor(model, obs_a1))
    result2, activation_a2 = get_extractor_activation(model, obs_to_tensor(model, obs_a2))

    return activation_a1, activation_a2


class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 1)

    def forward(self, x):
        h = self.linear1(x)
        # TODO discuss suitable activation functions
        return torch.tanh(h)

class SupervisedTraining:
    def __init__(self, model_name):
        model = load_model(model_name)
        train_data, test_data = load_data(model_name)


class CCS:
    def __init__(self, model_name, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="cuda", weight_decay=0.01, var_normalize=False):

        # training
        self.var_normalize = var_normalize
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        self.model = load_model(model_name)
        self.train_data, self.test_data = load_data(model_name)

        # TODO vectorize this
        self.train_activations1, self.train_activations2 = get_hidden_activations(train_data)
        self.test_activations1, self.test_activations2 = get_hidden_activations(test_data)

        self.best_probe = copy.deepcopy(self.probe)

    def initialize_probe(self):
        self.probe = MLPProbe(self.d)
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
        """
        Returns the CCS loss for two values each of shape (n,1) or (n,)
        """
        # TODO add more loss options
        informative_loss = (torch.min((1 - value_1)**2, (1 - value_2)**2)).mean(0)
        consistent_loss = ((value_1 + value_2)**2).mean(0)
        return consistent_loss + informative_loss

    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        # TODO normalize data in one call
        x0 = torch.tensor(self.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            value_1, value_2 = self.best_probe(x0), self.best_probe(x1)
        # TODO discuss whether we need this / can compute it
        # - compare with value network if applicable
        # - compare with expected returns from trajectories

        # avg_confidence = 0.5*(p0 + (1-p1))
        #predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        #acc = (predictions == y_test).mean()
        #acc = max(acc, 1 - acc)
        # return acc


    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
            
                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()
    
    def repeated_train(self):
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return best_loss


if __name__ == "__main__":
    env_name = "PongNoFrameskip-v4"
    model_name = "ppo-PongNoFrameskip-v4"
    # Generate dataset
    trajs = generate_dataset(env_name, model_name, num_episodes=10, num_envs=4, seed=42)
    # Train CCS without any labels
    ccs = CCS(model_name="ppo-PongNoFrameskip-v4")
    ccs.repeated_train()

    # Evaluate
    # ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
    # print("CCS accuracy: {}".format(ccs_acc))