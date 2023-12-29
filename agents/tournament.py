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
from torch.utils.tensorboard import SummaryWriter

from common import get_env, Agent
from itertools import product
from torchvision.io import write_video
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="SPAR_RL_ELK",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    # Tournament specific arguments: 
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, one round per match will be recorded")
    parser.add_argument("--env-id", type=str, default="pong_v3",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--agent-paths", type=str, default=["self"], nargs="+",
        help="the path to the agents to be loaded")
    parser.add_argument("--agent-names", type=str, default=["self"], nargs="+",
        help="the name of the agents to be loaded")
    parser.add_argument("--num-rounds", type=int, default=10,
        help="the number of rounds to play for each match")
    args = parser.parse_args()
    # fmt: on
    return args


@th.no_grad()
def tournament(envs, agents, total_rounds, device, video_path=None):
    winrates = np.zeros((len(agents), len(agents)))
    lengths = np.zeros((len(agents), len(agents)))
    matchs = product(range(len(agents)), range(len(agents)))
    for p1, p2 in matchs:
        # The video wrapper is not working with our env so we have to use our own. So we will save the first round of each match as a video:
        player1 = agents[p1]
        player2 = agents[p2]
        num_rounds = 0
        # TRY NOT TO MODIFY: start the game
        obs = th.Tensor(envs.reset()).to(device)
        total_length = 0
        current_lengths = np.zeros(obs.shape[0], dtype=np.int64)
        wins = 0
        draws = 0
        recording = video_path is not None
        if video_path is not None:
            frames = []
        start_time = time.time()
        while num_rounds < total_rounds or recording:
            actions = np.zeros(args.num_envs, dtype=np.int64)
            action1 = player1.get_action(obs[::2]).cpu().numpy()
            action2 = player2.get_action(obs[1::2]).cpu().numpy()
            actions[::2] = action1
            actions[1::2] = action2
            obs, rewards, dones, _ = envs.step(actions)
            if video_path is not None and recording:
                frame = obs[0, :, :, 0]
                frames.append(np.stack([frame, frame, frame], axis=2))
                if dones[0]:
                    recording = False
                    frames = np.stack(frames, dtype=np.uint8)
                    write_video(
                        str(video_path / f"{player1.name}_vs_{player2.name}.mp4"),
                        frames,
                        30.0,
                    )
                    print(
                        "Saved video to"
                        + str(video_path / f"{player1.name}_vs_{player2.name}.mp4")
                    )
            obs = th.Tensor(obs).to(device)
            num_rounds += np.logical_or(rewards > 0, dones).sum().item()
            total_length += (
                (current_lengths * np.logical_or(rewards > 0, dones)).sum().item()
            )
            current_lengths += 1
            current_lengths *= 1 - np.logical_or(rewards != 0, dones)
            wins += (rewards[::2] == 1).sum().item()
            draws += np.logical_and(rewards == 0, dones).sum().item()
        lengths[p1, p2] = total_length / num_rounds
        winrates[p1, p2] = wins / num_rounds
        print(
            f"{player1.name} vs {player2.name}: {wins} wins / {draws} draws / {num_rounds - wins - draws} losses"
        )
        print(f"Average episode length: {total_length / num_rounds}")
        print(
            f"SPS: {total_length + current_lengths.sum() / (time.time() - start_time)}"
        )

    return winrates, lengths


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.backends.cudnn.deterministic = args.torch_deterministic

    device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")
    hf_path = Path(__file__).resolve().parent.parent / "hf_models"
    # env setup
    envs = get_env(args, run_name)

    # agent loading
    agents = []
    for agent_path, name in zip(args.agent_paths, args.agent_names):
        print(f"Loading agent from {agent_path}")
        hf_hub_download(
            repo_id="Butanium/selfplay_ppo_pong_v3_pettingzoo_cleanRL",
            filename=agent_path,
            local_dir=hf_path,
        )
        agent = Agent(envs).to(device)
        agent.load(hf_path / agent_path)
        agent.requires_grad_(False)
        agent.name = name
        agents.append(agent)

    if args.capture_video:
        video_path = Path("videos") / run_name
        video_path.mkdir(parents=True, exist_ok=True)
    else:
        video_path = None
    winrates, mean_lengths = tournament(
        envs, agents, args.num_rounds, device, video_path
    )
    # Sort agents by winrate on both axis:
    indicies = np.argsort(np.sum(winrates, axis=1))[::-1]
    winrates = winrates[indicies]
    winrates = winrates[:, indicies]
    print(f"Winrates:\n{winrates}")
    # Save the results in a wandb table:
    if args.track:
        table = pd.DataFrame(winrates, columns=[agents[i].name for i in indicies])
        wandb.log({"Winrates table": wandb.Table(dataframe=table)})

    cNorm = colors.Normalize(vmin=0, vmax=1)
    fig, ax = plt.subplots()
    ax.imshow(winrates, cmap="bwr", norm=cNorm)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(len(agents)), labels=[agents[i].name for i in indicies])
    ax.set_yticks(np.arange(len(agents)), labels=[agents[i].name for i in indicies])
    for i in range(len(agents)):
        for j in range(len(agents)):
            ax.text(
                j, i, f"{winrates[i,j]:.2f}", ha="center", va="center", color="black"
            )
    ax.set_title("Winrates")
    fig.tight_layout()

    writer.add_figure("Winrates", fig)
    if args.track:
        wandb.log({"Winrates": wandb.Image(fig)})
    fig.savefig(f"runs/{run_name}/winrates.png")
    plt.close(fig)

    # Same for mean episode lengths:
    mean_lengths = mean_lengths[indicies]
    mean_lengths = mean_lengths[:, indicies]
    print(f"Mean episode lengths:\n{mean_lengths}")
    if args.track:
        table = pd.DataFrame(mean_lengths, columns=[agents[i].name for i in indicies])
        wandb.log({"Mean episode lengths table": wandb.Table(dataframe=table)})
    fig, ax = plt.subplots()
    ax.imshow(mean_lengths, cmap="bwr")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(len(agents)), labels=[agents[i].name for i in indicies])
    ax.set_yticks(np.arange(len(agents)), labels=[agents[i].name for i in indicies])
    for i in range(len(agents)):
        for j in range(len(agents)):
            ax.text(
                j,
                i,
                f"{mean_lengths[i,j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    ax.set_title("Mean episode lengths")
    fig.tight_layout()

    writer.add_figure("Mean episode lengths", fig)

    if args.track:
        wandb.log({"Mean episode lengths": wandb.Image(fig)})
    fig.savefig(f"runs/{run_name}/mean_lengths.png")
    plt.close(fig)
    envs.close()
    writer.close()
