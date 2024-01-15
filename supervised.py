
import torch as th
from sklearn.linear_model import LogisticRegression


def train_supervised(dataset_path, layer):
    """Linear classifier trained with supervised learning."""
    
    activations_path = dataset_path.with_suffix("") / "activations.pt"
    returns_path = dataset_path.with_suffix("") / "returns.pt"
    breakpoint()
    # TODO: Check what's wrong with the path
    if activations_path.exists():
        activation_pairs = th.load(activations_path)
        return_pairs = th.load(returns_path)
    
    
    x_train = neg_hs_train - pos_hs_train  
    x_test = neg_hs_test - pos_hs_test
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(x_train, y_train)
    print("Logistic regression accuracy: {}".format(lr.score(x_test, y_test)))


