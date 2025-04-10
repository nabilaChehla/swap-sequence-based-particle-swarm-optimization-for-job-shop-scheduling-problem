import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


class ScheduleVisualizer:
    """Handles visualization of schedules."""

    @staticmethod
    def plot_gantt_chart(
        jssp,
        figsize=(15, 8),
        save_folder=None,
        title="Job Shop Schedule Gantt Chart",
    ):
        """Plots a Gantt chart for the JSSP schedule without job color legend."""
        fig, ax = plt.subplots(figsize=figsize)
        machines = sorted({op.machine for job in jssp.jobs for op in job.operations})
        y_ticks = np.arange(len(machines))
        y_labels = [f"Machine {m}" for m in machines]

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time")
        ax.set_title(title)
        ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.7)

        colors = plt.cm.get_cmap("tab20", jssp.num_jobs)

        for job in jssp.jobs:
            for op in job.operations:
                if op.start_time is not None and op.end_time is not None:
                    y_pos = machines.index(op.machine)
                    rect = patches.Rectangle(
                        (op.start_time, y_pos - 0.4),
                        op.end_time - op.start_time,
                        0.8,
                        facecolor=colors(job.job_id),
                        edgecolor="black",
                        alpha=0.7,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        op.start_time + (op.end_time - op.start_time) / 2,
                        y_pos,
                        f"J{job.job_id}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

        max_time = max(
            op.end_time
            for job in jssp.jobs
            for op in job.operations
            if op.end_time is not None
        )
        ax.set_xlim(0, max_time * 1.05)

        plt.tight_layout()
        if save_folder is not None:
            # Create directory if it doesn't exist
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, "gantt_chart.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        plt.close()

    @staticmethod
    def plot_convergence(
        iteration_history,
        makespan_history,
        upper_bound=None,
        save_folder=None,
        title="Makespan Improvement Over Iterations",
    ):
        """Plots the makespan improvement over iterations and optionally saves to a folder.

        Args:
          iteration_history: List of iteration numbers
          makespan_history: List of best makespan values at each iteration
          upper_bound: A constant value representing the upper bound to show on the plot
          save_folder: Path to folder where plot should be saved (None to not save)
          title: Title for the plot
        """
        plt.figure(figsize=(10, 5))
        plt.plot(
            iteration_history,
            makespan_history,
            "b-",
            linewidth=1.5,
            label="Best Makespan",
        )

        # Add upper bound line
        if upper_bound is not None:
            plt.axhline(
                y=upper_bound,
                color="r",
                linestyle="--",
                linewidth=1.5,
                label=f"Upper Bound = {upper_bound}",
            )

        plt.xlabel("Iteration")
        plt.ylabel("Best Makespan")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        if save_folder is not None:
            # Create directory if it doesn't exist
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, "convergence_plot.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        plt.close()
