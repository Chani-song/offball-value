from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

from offball_value.bundesliga import (  # noqa: E402
    FIELD_LENGTH,
    FIELD_WIDTH,
    FPS,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.pass_dynamics import (  # noqa: E402
    ArrivalModelConfig,
    ReceptionRegionConfig,
    detect_kick_frame,
    estimate_frame_velocities,
    point_reception_estimate,
    reception_region_estimate,
)


DEFAULT_RESULTS = ROOT / "data" / "processed" / "offball_top120.csv"
DEFAULT_SUMMARY = (
    ROOT / "data" / "processed" / "visualizations" / "offball_half" / "summary.csv"
)
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "bundesliga-integrated"


def draw_pitch(ax, attacking_direction: int) -> None:
    half_length = FIELD_LENGTH / 2.0
    half_width = FIELD_WIDTH / 2.0
    line_color = "#526157"
    ax.set_facecolor("#eef4ec")
    ax.plot(
        [-half_length, half_length, half_length, -half_length, -half_length],
        [-half_width, -half_width, half_width, half_width, -half_width],
        color=line_color,
        lw=1.0,
    )
    ax.plot([0.0, 0.0], [-half_width, half_width], color=line_color, lw=1.0)
    ax.add_patch(plt.Circle((0.0, 0.0), 9.15, fill=False, color=line_color, lw=1.0))
    for side in (-1, 1):
        goal_x = side * half_length
        box_x = goal_x - 16.5 if side > 0 else goal_x
        ax.add_patch(
            plt.Rectangle(
                (box_x, -20.16),
                16.5,
                40.32,
                fill=False,
                color=line_color,
                lw=1.0,
            )
        )
    if attacking_direction > 0:
        ax.set_xlim(0.0, half_length)
    else:
        ax.set_xlim(-half_length, 0.0)
    ax.set_ylim(-half_width, half_width)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def plot_regions(
    out_file: Path,
    end_frame,
    attacking_team_id: str,
    attacking_direction: int,
    passer_id: str,
    ball_xy: tuple[float, float],
    regions: list[tuple[str, str, object]],
) -> None:
    max_weight = max(
        (float(region.weighted_value.max()) for _, _, region in regions),
        default=1.0,
    )
    fig, axes = plt.subplots(1, len(regions), figsize=(12, 8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    if len(regions) == 1:
        axes = [axes]
    for ax, (label, receiver_name, region) in zip(axes, regions):
        draw_pitch(ax, attacking_direction)
        ax.scatter(
            region.points[:, 0],
            region.points[:, 1],
            c=region.weighted_value,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=max(max_weight, 1e-12),
            marker="s",
            s=42,
            alpha=0.82,
            linewidths=0.0,
            zorder=2,
        )
        for player in end_frame.players.values():
            attacking = player.team_id == attacking_team_id
            ax.scatter(
                [player.x],
                [player.y],
                s=70,
                c="#2474e5" if attacking else "#111827",
                edgecolors="white",
                linewidths=1.3,
                zorder=5,
            )
        receiver = end_frame.players[region.receiver_id]
        passer = end_frame.players[passer_id]
        ax.scatter(
            [receiver.x],
            [receiver.y],
            s=180,
            facecolors="none",
            edgecolors="#00a896",
            linewidths=2.5,
            zorder=7,
        )
        ax.scatter(
            [passer.x],
            [passer.y],
            s=150,
            facecolors="none",
            edgecolors="#ff8c1a",
            linewidths=2.5,
            marker="s",
            zorder=7,
        )
        ax.scatter(
            [ball_xy[0]],
            [ball_xy[1]],
            s=45,
            c="#f5c518",
            edgecolors="#7a5200",
            linewidths=1.0,
            zorder=8,
        )
        ax.scatter(
            [region.peak_x],
            [region.peak_y],
            s=130,
            marker="*",
            c="#f5c518",
            edgecolors="#7a2d00",
            linewidths=1.2,
            zorder=8,
        )
        ax.plot(
            [ball_xy[0], region.peak_x],
            [ball_xy[1], region.peak_y],
            color="#f5c518",
            lw=1.4,
            ls="--",
            alpha=0.85,
            zorder=4,
        )
        ax.set_title(
            f"{receiver_name}\nDynamic option {region.option_value:.4f}",
            fontsize=11,
        )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare single-point dynamic reception races for one ranked scene."
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--history-seconds", type=float, default=0.4)
    parser.add_argument("--pass-speed-mps", type=float, default=14.0)
    parser.add_argument("--kick-search-seconds", type=float, default=3.0)
    parser.add_argument("--use-event-frame-as-kick", action="store_true")
    parser.add_argument("--region-resolution-m", type=float, default=1.5)
    parser.add_argument("--plot-out", type=Path, default=None)
    args = parser.parse_args()

    summary_row = pd.read_csv(args.summary).query("rank == @args.rank").iloc[0]
    rows = pd.read_csv(args.results, low_memory=False)
    matches = rows[
        (rows["match_id"] == summary_row["match_id"])
        & (rows["event_frame"] == summary_row["event_frame"])
        & (rows["runner_name"] == summary_row["runner_name"])
    ]
    if matches.empty:
        raise ValueError(f"Rank {args.rank} scene is not present in {args.results}")
    row = matches.iloc[0]

    config = ArrivalModelConfig(
        history_seconds=args.history_seconds,
        pass_speed_mps=args.pass_speed_mps,
    )
    region_config = ReceptionRegionConfig(
        resolution_m=args.region_resolution_m,
    )
    event_frame_id = int(row["event_frame"])
    history_frames = max(1, int(round(config.history_seconds * FPS)))
    kick_search_frames = max(1, int(round(args.kick_search_seconds * FPS)))
    target_frames = range(
        event_frame_id - kick_search_frames - history_frames,
        event_frame_id + kick_search_frames + 1,
    )
    files = find_bundesliga_files(args.data_dir, str(row["match_id"]))
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    frames = load_bundesliga_frames(files["positions"], target_frames=target_frames)
    kick = detect_kick_frame(
        frames.values(),
        str(row["passer_id"]),
        event_frame_id,
        target_xy=(
            (float(row["pass_end_x"]), float(row["pass_end_y"]))
            if pd.notna(row.get("pass_end_x")) and pd.notna(row.get("pass_end_y"))
            else None
        ),
        target_player_id=(
            str(row["recipient_id"])
            if pd.notna(row.get("recipient_id"))
            else None
        ),
    )
    end_frame_id = (
        event_frame_id if args.use_event_frame_as_kick else kick.kick_frame_id
    )
    end_frame = frames[end_frame_id]
    velocities = estimate_frame_velocities(frames.values(), end_frame_id, config)
    ball_xy = (
        (end_frame.ball.x, end_frame.ball.y)
        if end_frame.ball is not None
        else (float(row["ball_x"]), float(row["ball_y"]))
    )

    target_receiver_id = str(row["defender_target_receiver_id"])
    runner_id = str(row["runner_id"])
    target_receiver = end_frame.players[target_receiver_id]
    candidate_points = [
        (
            "target_receiver_feet",
            target_receiver_id,
            (target_receiver.x, target_receiver.y),
        ),
        (
            "runner_threat_point",
            runner_id,
            (
                float(row["runner_threat_point_x"]),
                float(row["runner_threat_point_y"]),
            ),
        ),
    ]

    outputs = []
    region_outputs = []
    plotted_regions = []
    for label, receiver_id, target_xy in candidate_points:
        estimate = point_reception_estimate(
            end_frame,
            velocities,
            receiver_id,
            str(row["team_id"]),
            ball_xy,
            target_xy,
            config,
            passer_id=str(row["passer_id"]),
        )
        defender_name = (
            metadata.players[estimate.nearest_defender_id].short_name
            if estimate.nearest_defender_id in metadata.players
            else estimate.nearest_defender_id
        )
        receiver_name = (
            metadata.players[receiver_id].short_name
            if receiver_id in metadata.players
            else receiver_id
        )
        path_defender_name = (
            metadata.players[estimate.path_suppression_defender_id].short_name
            if estimate.path_suppression_defender_id in metadata.players
            else estimate.path_suppression_defender_id
        )
        pressure_defender_name = (
            metadata.players[estimate.passer_pressure_defender_id].short_name
            if estimate.passer_pressure_defender_id in metadata.players
            else estimate.passer_pressure_defender_id
        )
        velocity = velocities[receiver_id]
        outputs.append(
            {
                "candidate": label,
                "receiver": receiver_name,
                "receiver_speed_mps": velocity.speed,
                "target_x": estimate.target_x,
                "target_y": estimate.target_y,
                "pass_distance_m": estimate.pass_distance_m,
                "ball_eta_s": estimate.ball_arrival_time_s,
                "receiver_eta_s": estimate.receiver_arrival_time_s,
                "first_defender": defender_name,
                "defender_eta_s": estimate.nearest_defender_arrival_time_s,
                "defender_margin_s": estimate.defender_time_margin_s,
                "p_receiver_first": estimate.receiver_first_probability,
                "path_defender": path_defender_name,
                "path_fraction": estimate.path_suppression_fraction,
                "path_margin_s": estimate.path_time_margin_s,
                "p_path_survives": estimate.path_survival_probability,
                "p_secure_given_first": estimate.secure_possession_probability,
                "pressure_defender": pressure_defender_name,
                "pressure_eta_s": estimate.passer_pressure_arrival_time_s,
                "p_execute_pressure": estimate.pressure_execution_probability,
                "p_execute_distance": estimate.distance_execution_probability,
                "p_execute": estimate.pass_execution_probability,
                "p_receive": estimate.receive_probability,
            }
        )
        region = reception_region_estimate(
            end_frame,
            velocities,
            receiver_id,
            str(row["passer_id"]),
            str(row["team_id"]),
            int(row["attacking_direction"]),
            ball_xy,
            config,
            region_config,
        )
        region_outputs.append(
            {
                "candidate": label,
                "receiver": receiver_name,
                "points": region.point_count,
                "expected_p_receive": region.expected_receive_probability,
                "expected_receive_value": region.expected_receive_value,
                "dynamic_option_value": region.option_value,
                "peak_x": region.peak_x,
                "peak_y": region.peak_y,
                "peak_p_receive": region.peak_receive_probability,
                "peak_receive_value": region.peak_receive_value,
                "peak_weighted_value": region.peak_weighted_value,
            }
        )
        plotted_regions.append((label, receiver_name, region))

    print(
        f"Rank {args.rank}: {row['passer_name']} | runner={row['runner_name']} | "
        f"event_frame={event_frame_id} | kick_frame={end_frame_id} "
        f"({end_frame_id - event_frame_id:+d}) | "
        f"kick_detected={kick.detected} | "
        f"target_alignment={kick.target_alignment:.3f} | "
        f"detected_release_speed={kick.release_ball_speed_mps:.2f}m/s | "
        f"history={config.history_seconds:.2f}s | "
        f"model_pass_speed={config.pass_speed_mps:.1f}m/s"
    )
    print(pd.DataFrame(outputs).to_string(index=False))
    print("\nReceiving-region integration")
    print(pd.DataFrame(region_outputs).to_string(index=False))
    plot_out = args.plot_out or (
        ROOT
        / "data"
        / "processed"
        / "visualizations"
        / "dynamic_reception"
        / f"rank_{args.rank:02d}.png"
    )
    plot_regions(
        plot_out,
        end_frame,
        str(row["team_id"]),
        int(row["attacking_direction"]),
        str(row["passer_id"]),
        ball_xy,
        plotted_regions,
    )
    print(f"\nSaved receiving-region plot to {plot_out}")


if __name__ == "__main__":
    main()
