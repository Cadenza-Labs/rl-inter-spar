import torch as th 
import numpy as np 
from sklearn.linear_model import Ridge 
from torch import nn 
from einops import rearrange
from nicehooks import nice_hooks
from agents.common import preprocess
from ccs_utils import extract_activations

@th.no_grad()
def supervised_prediction(lr, obs, module, layer_name):
    """Predict the value of an observation using a supervised model."""

    obs = preprocess(obs)
    _, activations = nice_hooks.run(module, obs, return_activations=True)
    act = activations[layer_name].flatten(start_dim=1)
    return lr(act)


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
    ) = extract_activations(
        model,
        layer_name,
        dataset_path,
        verbose,
        device,
        val_fraction,
        gamma,
        seed,
        normalize=False,
    )
    train_activations = train_activations.flatten(start_dim=2)
    test_activations = test_activations.flatten(start_dim=2)
    x_train = rearrange(train_activations, "n p d -> (n p) d").cpu().numpy()
    x_test = rearrange(test_activations, "n p d -> (n p) d").cpu().numpy()

    # TODO: Put an own function
    # Generate value predictions from the value network for each observation in the dataset
    train_obs = rearrange(
        th.tensor(np.array(train_observations, dtype=np.uint8), dtype=th.float),
        "n p h w f -> (n p) h w f",
    ).to(device)
    test_obs = rearrange(
        th.tensor(np.array(test_observations, dtype=np.uint8), dtype=th.float),
        "n p h w f -> (n p) h w f",
    ).to(device)
    with th.no_grad():
        y_train = model.get_value(train_obs).squeeze().detach().cpu().numpy()
        y_test = model.get_value(test_obs).squeeze().detach().cpu().numpy()
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]

    lr = Ridge()
    print(
        f"Fitting a linear regression with {len(x_train)} samples and {x_train.shape[1]} features"
    )
    lr.fit(x_train, y_train)
    print("Linear regression accuracy (test): {}".format(lr.score(x_test, y_test)))
    print("Linear regression accuracy (train): {}".format(lr.score(x_train, y_train)))
    # Convert lr to nn.Linear
    module = nn.Linear(x_train.shape[1], 1)
    module.weight.data.copy_(th.tensor(lr.coef_, device=device))
    module.bias.data.copy_(th.tensor(lr.intercept_, device=device))
    return module.to(device)
