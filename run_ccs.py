import pickle
from itertools import product

import argparse
from agents.common import get_env
from utils import strtobool
from huggingface_hub import hf_hub_download
from probe_visualization import monitor_probes
# Do not remove Trajectory import, it is used in the pickle.load
from ccs_utils import generate_dataset, load_model, Trajectory
from supervised import supervised_prediction, train_supervised
from ccs import *


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
    ccs_group.add_argument(
        "--skip-supervised-probe",
        help="Whether to skip the supervised probe training",
        default=False,
        action="store_true",
    )
    ccs_group.add_argument(
        "--normalize",
        help="Whether to normalize the activations wrt the ball approaching",
        action="store_true",
        default=False,
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
                    normalize=args.normalize,
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
    if not args.skip_supervised_probe:
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
                f"Right supervised probe on {layer_name}": lambda obs, supervised_probe=supervised_probe, layer_name=layer_name: supervised_prediction(
                    supervised_probe, obs[:1], model, layer_name
                ).item(),
                f"Left supervised probe on {layer_name}": lambda obs, supervised_probe=supervised_probe, layer_name=layer_name: supervised_prediction(
                    supervised_probe, obs[1:2], model, layer_name
                ).item(),
            }
            probes_fn_dict.update(probe_dict)
            fn_grouped_by_probe[f"{layer_name}_supervised_probes"] = probe_dict

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
