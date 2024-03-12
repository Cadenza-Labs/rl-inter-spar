from nicehooks import nice_hooks
import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.preprocessing import preprocess_obs


def obs_to_tensor(model, obs: np.ndarray):
    """
    Convert an obs to a tensor using model internal obs_to_tensor
    """
    return model.policy.obs_to_tensor(obs)[0]


def sb3_preprocess(model, obs: th.Tensor):
    """
    Preprocess an observation according to the sb3 model preprocessing.
    Tested for:
        - PPO
    """
    return preprocess_obs(
        obs,
        model.policy.observation_space,
        normalize_images=model.policy.normalize_images,
    )


def get_activations(module: nn.Module, obs, *args):
    """
    Get activations from module on obs
    Returns: A tuple containing (module(obs), cache)
    where cache is a dictionary where keys are layer and item activations
    """
    return nice_hooks.run(module, obs, *args, return_activations=True)


def get_extractor_activation(model, obs: th.Tensor):
    """
    Returns activations hooked from the sb3 model's feature_extractor
    """
    result, cache = get_activations(
        model.policy.features_extractor, sb3_preprocess(model, obs)
    )
    return result, cache
