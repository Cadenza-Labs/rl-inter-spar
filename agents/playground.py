# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
import argparse
import os
from pathlib import Path
import random
import time
from distutils.util import strtobool

from huggingface_hub import hf_hub_download
from matplotlib.animation import FuncAnimation
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

from common import get_env, Agent
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


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
    parser.add_argument("--num-rounds", type=int, default=30,
        help="the number of rounds to record")
    parser.add_argument("--sliding-window", type=int, default=1000,
        help="the sliding window for the value plot")
    parser.add_argument("--agent1-path", type=str, required=True,
        help="the path to the first agent to be loaded")
    parser.add_argument("--agent1-name", type=str, default="",
        help="the name of the first agent to be loaded")
    parser.add_argument("--agent2-path", type=str, default="",
        help="the path to the second agent to be loaded, defaults to the first agent")
    parser.add_argument("--agent2-name", type=str, default="",
        help="the name of the second agent to be loaded")
    args = parser.parse_args()
    # fmt: on
    return args


@th.no_grad()
def playground(
    envs,
    agent1,
    agent2,
    metrics: dict,
    device,
    video_path,
    rounds_to_record=30,
    max_video_length=2000,
    sliding_window=1000,
):
    # The video wrapper is not working with our env so we have to use our own. So we will save the first round of each match as a video:
    player1 = agent1
    player2 = agent2
    # TRY NOT TO MODIFY: start the game
    obs = th.Tensor(envs.reset()).to(device)
    total_length = 0
    wins = 0
    frames = []
    num_rounds = 0
    start_time = time.time()
    metric_values = {k: [] for k in metrics.keys()}
    while True:
        actions = np.zeros(args.num_envs, dtype=np.int64)
        action1 = player1.get_action(obs[::2])
        action2 = player2.get_action(obs[1::2])
        for name, fn in metrics.items():
            metric_values[name].append(fn(obs))
        actions[::2] = action1.cpu().numpy()
        actions[1::2] = action2.cpu().numpy()
        obs, rewards, dones, _ = envs.step(actions)

        frame = obs[0, :, :, 0]
        frames.append(np.stack([frame, frame, frame], axis=2))
        obs = th.Tensor(obs).to(device)
        num_rounds += np.logical_or(rewards > 0, dones).sum().item()
        total_length += 1 * obs.shape[0]
        wins += (rewards[::2] == 1).sum().item()
        if rewards[0] != 0:
            rounds_to_record -= 1
        if dones[0] or rounds_to_record == 0:
            print(f"SPS: {total_length  / (time.time() - start_time)}")
            print(f"{player1.name} vs {player2.name}: {num_rounds} rounds played")
            print(
                f"{player1.name} vs {player2.name}: {wins} wins /  {num_rounds - wins} losses"
            )
            print(f"Average episode length: {total_length / num_rounds}")
            frames = np.stack(frames, dtype=np.uint8)
            total_frames = min(len(frames), max_video_length)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            # Initialize the Pong frame
            pong_img = ax1.imshow(frames[0], cmap="gray")

            # Initialize the agent value plot
            time_values = list(range(total_frames))
            plots = []
            ymin = -1
            ymax = 1
            for name, values in metric_values.items():
                values = values[: min(sliding_window, total_frames)]
                (plot,) = ax2.plot(
                    time_values[: min(sliding_window, total_frames)],
                    values,
                    label=name,
                )
                ymin = min(ymin, min(values))
                ymax = max(ymax, max(values))
                plots.append(plot)
            ax2.legend()
            # ax2.set_xlim(0, total_frames)
            ax2.set_ylim(ymin, ymax)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Value")
            ax2.set_title("Value over time")

            # Update function for animation
            pbar = tqdm(total=total_frames)

            def update(frame):
                pbar.n = frame
                pbar.refresh()
                pong_img.set_array(frames[frame])
                for plot, values in zip(plots, metric_values.values()):
                    plot.set_data(
                        time_values[max(0, frame - sliding_window) : frame + 1],
                        values[max(0, frame - sliding_window) : frame + 1],
                    )
                if frame > sliding_window:
                    ax2.set_xlim(frame - sliding_window, frame)
                return pong_img, *plots

            # Create the animation
            animation = FuncAnimation(fig, update, frames=total_frames, blit=True)

            # Save the animation as a video
            video_path.mkdir(parents=True, exist_ok=True)
            pbar.set_description("Generating video")
            animation.save(
                str(video_path / f"{player1.name}_vs_{player2.name}.mp4"), fps=30
            )
            print(
                "Saved video to"
                + str(video_path / f"{player1.name}_vs_{player2.name}.mp4")
            )
            # Save value plot as a png
            fig.savefig(str(video_path / f"{player1.name}_vs_{player2.name}.png"))
            break


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
    agent1.load(hf_path / args.agent1_path)
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
    agent2.load(hf_path / args.agent2_path)
    agent2.requires_grad_(False)
    agent2.name = args.agent2_name

    video_path = Path("videos") / run_name
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
        rounds_to_record=args.num_rounds,
        max_video_length=args.max_video_length,
        sliding_window=args.sliding_window,
    )
    envs.close()
