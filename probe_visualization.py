from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torch as th
import numpy as np
import matplotlib.style as mplstyle
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.ticker import AutoLocator
from pathlib import Path


class FrameSelector:

    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.frame = 0
        self.last_frame = -1

    def set_current_frame(self, frame):
        self.frame = frame

    def __call__(self, frame):
        if frame != self.last_frame:
            self.frame = (self.frame + 1) % self.num_frames
        self.last_frame = frame
        return self.frame


class ProbeMonitor:
    def __init__(self, envs, agent1, agent2, metrics, device):
        self.envs = envs
        self.agent1 = agent1
        self.agent2 = agent2
        self.metrics = metrics
        self.device = device
        self.metrics_values = {k: [] for k in metrics.keys()}
        self.frames = None

    @th.no_grad
    def run(self, rounds_to_record=30, max_num_steps=2000):
        """
        Run a game between the two agents and record the frames and metrics

        Args:
            rounds_to_record: number of rounds to record
            max_num_steps: maximum number of steps to record
        """
        player1 = self.agent1
        player2 = self.agent2
        obs = th.Tensor(self.envs.reset()).to(self.device)
        total_length = 0
        wins = 0
        frames = []
        num_rounds = 0
        start_time = time.time()
        while True:
            actions = np.zeros(len(obs), dtype=np.int64)
            action1 = player1.get_action(obs[::2])
            action2 = player2.get_action(obs[1::2])
            for name, fn in self.metrics.items():
                self.metrics_values[name].append(fn(obs))
            actions[::2] = action1.cpu().numpy()
            actions[1::2] = action2.cpu().numpy()
            obs, rewards, dones, _ = self.envs.step(actions)
            frame = obs[0, :, :, 0]
            frames.append(np.stack([frame, frame, frame], axis=2))
            obs = th.Tensor(obs).to(self.device)
            num_rounds += np.logical_or(rewards > 0, dones).sum().item()
            total_length += obs.shape[0] // 2
            wins += (rewards[::2] == 1).sum().item()
            if rewards[0] != 0:
                rounds_to_record -= 1
            if dones[0] or rounds_to_record == 0 or len(frames) > max_num_steps:
                print(f"SPS: {total_length  / (time.time() - start_time)}")
                if num_rounds > 0:
                    print(
                        f"{player1.name} vs {player2.name}: {num_rounds} rounds played"
                    )
                    print(
                        f"{player1.name} vs {player2.name}: {wins} wins /  {num_rounds - wins} losses"
                    )
                    print(f"Average episode length: {total_length / num_rounds}")
                else:
                    print("No rounds finished")
                self.frames = np.stack(frames, dtype=np.uint8)
                return

    def init_plot(self, ax1, ax2, metrics_name, sliding_window, total_frames):
        """
        Initialize ax1 as the game render and ax2 as the value plot

        Args:
            ax1: the axis to render the game
            ax2: the axis to plot the metrics
            metrics_name: the name of the metrics to plot
            sliding_window: the size of the sliding window to plot
            total_frames: the total number of frames to plot
        """
        # Initialize the game frame
        ax1.axis("off")
        game_render = ax1.imshow(self.frames[0], cmap="gray")

        # Initialize the agent value plot
        plots = []
        ymin = -1
        ymax = 1
        for name in metrics_name:
            values = self.metrics_values[name]
            values = values[: min(sliding_window, total_frames)]
            (plot,) = ax2.plot(
                list(range(min(sliding_window, total_frames))),
                values,
                label=name,
                alpha=0.5,
            )
            ymin = min(ymin, min(values) - 0.05)
            ymax = max(ymax, max(values) + 0.05)
            plots.append(plot)
        ax2.set_ylim(ymin, ymax)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Value")
        ax2.set_title("Value over time")
        return game_render, plots

    def interactive_visualization(self, sliding_window=200):
        """
        Create an interactive visualization of the game and the metrics. The
        UI contains a slider to control the time, a play/pause button and checkboxes
        to control the visibility of the metrics.

        Args:
            sliding_window: the size of the sliding window to plot
        """
        mplstyle.use("fast")
        fig, (ax1, ax2, rax) = plt.subplots(
            1, 3, figsize=(12, 6), width_ratios=[3, 3, 1], layout="constrained"
        )
        game_render, plots = self.init_plot(
            ax1, ax2, self.metrics_values.keys(), sliding_window, len(self.frames)
        )
        frame_selector = FrameSelector(len(self.frames))
        lock_slider = False

        # Create the slider and the button to control time below the game render
        ax_time = fig.add_axes([0.25, 0.01, 0.65, 0.03])
        ax_media = fig.add_axes([0.1, 0.01, 0.1, 0.03])
        time_slider = Slider(
            ax_time,
            "Time",
            0,
            len(self.frames) - 1,
            valinit=0,
            valstep=1,
            valfmt="%d",
        )

        def on_changed(val):
            nonlocal lock_slider
            if lock_slider:
                lock_slider = False
                return
            frame = int(val)
            frame_selector.set_current_frame(frame - 1)
            if val >= 0:
                update(None)

        time_slider.on_changed(on_changed)

        # Play/pause button
        media_button = Button(ax_media, "||", hovercolor="0.975")

        def play_pause(_event):
            if media_button.label.get_text() == "||":
                media_button.label.set_text("▶")
                interactive_animation.event_source.stop()
            else:
                media_button.label.set_text("||")
                interactive_animation.event_source.start()

        media_button.on_clicked(play_pause)

        # Make checkbuttons with all plotted lines with correct visibility
        rax.set_title("Metrics")
        rax.set_frame_on(False)
        labels = [str(plot.get_label()) for plot in plots]
        visibility = [False] * len(labels)
        plot_colors = [plot.get_color() for plot in plots]
        check = CheckButtons(
            rax, labels, visibility, label_props={"color": plot_colors}
        )
        label_to_plot = {plot.get_label(): plot for plot in plots}
        label_to_index = {label: i for i, label in enumerate(labels)}
        for plot in plots:
            plot.set_visible(False)

        def checkbox_callback(label):
            plot = label_to_plot[label]
            plot.set_visible(check.get_status()[label_to_index[label]])
            plot.figure.canvas.draw_idle()

        check.on_clicked(checkbox_callback)

        # Create and show the animation
        def update(_frame):
            nonlocal lock_slider
            frame = frame_selector(_frame)
            if _frame is not None and time_slider.val != frame:
                lock_slider = True
                time_slider.set_val(frame)
            game_render.set_array(self.frames[frame])
            for plot, values in zip(plots, self.metrics_values.values()):
                plot.set_data(
                    list(range(max(0, frame - sliding_window), frame + 1)),
                    values[max(0, frame - sliding_window) : frame + 1],
                )
            if frame > sliding_window:
                ax2.set_xlim(frame - sliding_window, frame)
            else:
                ax2.set_xlim(0, sliding_window)
            return [game_render] + plots

        interactive_animation = FuncAnimation(
            fig, update, frames=len(self.frames), blit=False, interval=1000 / 60
        )
        plt.tight_layout()
        plt.show()

    def save_video(
        self,
        metrics_names,
        video_path: Path,
        file_name=None,
        max_video_length=4000,
        sliding_window=200,
    ):
        """
        Save a video of the game and the metrics

        Args:
            metrics_name: the name of the metrics to plot
            video_path: the path to save the video
            file_name: the name of the video file. If None, the name is
                {agent1.name}_vs_{agent2.name}
            max_video_length: the maximum number of frames to save
            sliding_window: the size of the sliding window to plot
        """
        fig, (ax1, ax2, lax) = plt.subplots(
            1, 3, figsize=(19, 10), layout="constrained"
        )
        game_render, plots = self.init_plot(
            ax1,
            ax2,
            metrics_names,
            sliding_window,
            min(max_video_length, len(self.frames)),
        )
        # Add the legend to the right subplot
        h, l = ax2.get_legend_handles_labels()
        lax.legend(h, l, borderaxespad=0)
        lax.axis("off")

        # Update function for animation
        pbar = tqdm(total=min(max_video_length, len(self.frames)))

        def update(frame):
            pbar.n = frame
            pbar.refresh()
            game_render.set_array(self.frames[frame])
            for plot, metric in zip(plots, metrics_names):
                values = self.metrics_values[metric]
                plot.set_data(
                    list(range(max(0, frame - sliding_window), frame + 1)),
                    values[max(0, frame - sliding_window) : frame + 1],
                )
            # Some magic to make the x axis of the plot look good
            ax2.xaxis.set_major_locator(AutoLocator())
            if frame > sliding_window:
                ax2.set_xlim(frame - sliding_window, frame)
                ax2.xaxis.set_major_locator(AutoLocator())
            xinf, xlim = ax2.get_xlim()
            xticks = ax2.get_xticks()
            if xticks[-1] > xlim:
                xticks[-2] = xlim
                xticks = xticks[:-1]
            elif xticks[-1] < xlim:
                xticks[-1] = xlim
            if xticks[0] < xinf:
                xticks[1] = xinf
                xticks = xticks[1:]
            elif xticks[0] > xinf:
                xticks[0] = xinf
            ax2.set_xticks(xticks)
            return [game_render] + plots

        # Create the animation
        animation = FuncAnimation(
            fig, update, frames=min(max_video_length, len(self.frames)), blit=True
        )

        video_path.mkdir(parents=True, exist_ok=True)
        f_name = file_name or f"{self.agent1.name}_vs_{self.agent2.name}"
        pbar.set_description("Generating video")
        animation.save(str(video_path / f"{f_name}.mp4"), fps=30)
        pbar.close()
        print("Saved video to " + str(video_path / f"{f_name}.mp4"))
