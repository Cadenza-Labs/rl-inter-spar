# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
import argparse
import os
from pathlib import Path
import random
import time
from distutils.util import strtobool

from huggingface_hub import hf_hub_download

import numpy as np
import torch as th

from common import get_env, Agent, playground


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    # Playground specific arguments: 
    parser.add_argument("--env-id", type=str, default="pong_v3",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=16,
    help="the number of parallel game environments")
    parser.add_argument("--max-video-length", type=int, default=2000)
    parser.add_argument("--num-rounds", type=int, default=5,
        help="the number of rounds to record")
    parser.add_argument("--sliding-window", type=int, default=200,
        help="the sliding window for the value plot")
    parser.add_argument("--agent1-path", type=str, required=True,
        help="the path to the first agent to be loaded")
    parser.add_argument("--agent1-name", type=str, default="",
        help="the name of the first agent to be loaded")
    parser.add_argument("--agent2-path", type=str, default="",
        help="the path to the second agent to be loaded, defaults to the first agent")
    parser.add_argument("--agent2-name", type=str, default="",
        help="the name of the second agent to be loaded")
    parser.add_argument("--interactive", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, an interactive plot will be shown")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, video will be recorded")
    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.backends.cudnn.deterministic = args.torch_deterministic

    device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")
    hf_path = Path(__file__).resolve().parent.parent / "hf_models"
    # env setup
    args.capture_video = True
    envs = get_env(args, run_name)

    # agent loading
    if args.agent1_name == "":
        args.agent1_name = args.agent1_path.split("/")[-1]
    print(f"Loading agent 1 from {args.agent1_path}")
    hf_hub_download(
        repo_id="Butanium/selfplay_ppo_pong_v3_pettingzoo_cleanRL",
        filename=args.agent1_path,
        local_dir=hf_path,
    )
    agent1 = Agent(envs).to(device)
    agent1.load(hf_path / args.agent1_path, device)
    agent1.requires_grad_(False)
    agent1.name = args.agent1_name
    if args.agent2_path == "":
        args.agent2_path = args.agent1_path
        args.agent2_name = args.agent1_name
    print(f"Loading agent 2 from {args.agent2_path}")
    hf_hub_download(
        repo_id="Butanium/selfplay_ppo_pong_v3_pettingzoo_cleanRL",
        filename=args.agent2_path,
        local_dir=hf_path,
    )
    agent2 = Agent(envs).to(device)
    agent2.load(hf_path / args.agent2_path, device)
    agent2.requires_grad_(False)
    agent2.name = args.agent2_name
    if args.capture_video:
        video_path = Path("videos") / run_name
    else:
        video_path = None
    playground(
        envs,
        agent1,
        agent2,
        {
            "Right Player Value": lambda obs: agent1.get_value(obs[:1]).item(),
            "Left Player Value": lambda obs: agent2.get_value(obs[1:2]).item(),
        },
        device,
        video_path,
        interactive=args.interactive,
        rounds_to_record=args.num_rounds,
        max_video_length=args.max_video_length,
        sliding_window=args.sliding_window,
    )
    envs.close()
