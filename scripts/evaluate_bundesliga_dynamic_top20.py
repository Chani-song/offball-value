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
    BundesligaFrame,
    BundesligaMatchMeta,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.pass_dynamics import (  # noqa: E402
    ArrivalModelConfig,
    ReceptionRegionConfig,
    detect_kick_frame,
    estimate_frame_velocities,
    reception_region_estimate,
)


DEFAULT_RESULTS = ROOT / "data" / "processed" / "offball_top120.csv"
DEFAULT_SUMMARY = (
    ROOT / "data" / "processed" / "visualizations" / "offball_half" / "summary.csv"
)
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "bundesliga-integrated"
DEFAULT_OUT = ROOT / "data" / "processed" / "dynamic_top20.csv"
DEFAULT_VIS_DIR = ROOT / "data" / "processed" / "visualizations" / "dynamic_top20"


def player_name(metadata: BundesligaMatchMeta, player_id: str | None) -> str:
    if not player_id:
        return ""
    player = metadata.players.get(player_id)
    return player.short_name if player is not None else player_id


def is_goalkeeper(metadata: BundesligaMatchMeta, player_id: str) -> bool:
    player = metadata.players.get(player_id)
    return player is not None and player.position == "TW"


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


def plot_scene(
    out_file: Path,
    rank: int,
    frame: BundesligaFrame,
    metadata: BundesligaMatchMeta,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    passer_id: str,
    recipient_id: str | None,
    runner_id: str,
    existing_top_id: str | None,
    dynamic_top_id: str,
    dynamic_top_region,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    draw_pitch(ax, attacking_direction)
    ax.scatter(
        dynamic_top_region.points[:, 0],
        dynamic_top_region.points[:, 1],
        c=dynamic_top_region.weighted_value,
        cmap="YlOrRd",
        marker="s",
        s=48,
        alpha=0.78,
        linewidths=0.0,
        zorder=2,
    )
    for player in frame.players.values():
        attacking = player.team_id == attacking_team_id
        ax.scatter(
            [player.x],
            [player.y],
            s=75,
            c="#2474e5" if attacking else "#111827",
            edgecolors="white",
            linewidths=1.3,
            zorder=5,
        )

    passer = frame.players.get(passer_id)
    runner = frame.players.get(runner_id)
    recipient = frame.players.get(recipient_id) if recipient_id else None
    existing_top = frame.players.get(existing_top_id) if existing_top_id else None
    dynamic_top = frame.players[dynamic_top_id]
    if passer is not None:
        ax.scatter(
            [passer.x],
            [passer.y],
            s=160,
            facecolors="none",
            edgecolors="#ff8c1a",
            linewidths=2.6,
            marker="s",
            zorder=8,
        )
    if runner is not None:
        ax.scatter(
            [runner.x],
            [runner.y],
            s=175,
            facecolors="none",
            edgecolors="#ff6b00",
            linewidths=2.6,
            marker="o",
            zorder=8,
        )
    if recipient is not None:
        ax.scatter(
            [recipient.x],
            [recipient.y],
            s=170,
            facecolors="none",
            edgecolors="#00a6c7",
            linewidths=2.5,
            marker="^",
            zorder=9,
        )
        ax.plot(
            [ball_xy[0], recipient.x],
            [ball_xy[1], recipient.y],
            color="#f5c518",
            lw=2.0,
            zorder=4,
        )
    if existing_top is not None:
        ax.scatter(
            [existing_top.x],
            [existing_top.y],
            s=170,
            facecolors="none",
            edgecolors="#7e57c2",
            linewidths=2.5,
            marker="D",
            zorder=9,
        )
    ax.scatter(
        [dynamic_top.x],
        [dynamic_top.y],
        s=190,
        facecolors="none",
        edgecolors="#00a651",
        linewidths=2.8,
        marker="D",
        zorder=10,
    )
    ax.scatter(
        [dynamic_top_region.peak_x],
        [dynamic_top_region.peak_y],
        s=145,
        c="#f5c518",
        edgecolors="#7a2d00",
        linewidths=1.2,
        marker="*",
        zorder=10,
    )
    ax.plot(
        [ball_xy[0], dynamic_top_region.peak_x],
        [ball_xy[1], dynamic_top_region.peak_y],
        color="#00a651",
        lw=1.7,
        ls="--",
        zorder=4,
    )
    ax.scatter(
        [ball_xy[0]],
        [ball_xy[1]],
        s=45,
        c="#f5c518",
        edgecolors="#7a5200",
        linewidths=1.0,
        zorder=10,
    )
    ax.set_title(
        f"Rank {rank:02d} | Dynamic top: {player_name(metadata, dynamic_top_id)} "
        f"({dynamic_top_region.option_value:.4f})",
        fontsize=11,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def matched_result_row(results: pd.DataFrame, summary_row: pd.Series) -> pd.Series:
    matches = results[
        (results["match_id"] == summary_row["match_id"])
        & (results["event_frame"] == summary_row["event_frame"])
        & (results["runner_name"] == summary_row["runner_name"])
    ]
    if matches.empty:
        raise ValueError(
            f"No result row for rank {int(summary_row['rank'])}: "
            f"{summary_row['match_id']} frame {int(summary_row['event_frame'])}"
        )
    return matches.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate dynamic receiving-region options for the current top scenes."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vis-dir", type=Path, default=DEFAULT_VIS_DIR)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--history-seconds", type=float, default=0.4)
    parser.add_argument("--kick-search-seconds", type=float, default=3.0)
    parser.add_argument("--pass-speed-mps", type=float, default=14.0)
    parser.add_argument("--region-resolution-m", type=float, default=1.5)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary).sort_values("rank").head(args.top_k)
    results = pd.read_csv(args.results, low_memory=False)
    ranked_rows = [
        (summary_row, matched_result_row(results, summary_row))
        for _, summary_row in summary.iterrows()
    ]
    config = ArrivalModelConfig(
        history_seconds=args.history_seconds,
        pass_speed_mps=args.pass_speed_mps,
    )
    region_config = ReceptionRegionConfig(
        resolution_m=args.region_resolution_m,
    )
    history_frames = max(1, int(round(config.history_seconds * FPS)))
    kick_search_frames = max(1, int(round(args.kick_search_seconds * FPS)))

    metadata_by_match: dict[str, BundesligaMatchMeta] = {}
    frames_by_key: dict[tuple[str, int], BundesligaFrame] = {}
    for match_id in sorted({str(row["match_id"]) for _, row in ranked_rows}):
        match_rows = [row for _, row in ranked_rows if str(row["match_id"]) == match_id]
        target_frames: set[int] = set()
        for row in match_rows:
            event_frame = int(row["event_frame"])
            target_frames.update(
                range(
                    event_frame - kick_search_frames - history_frames,
                    event_frame + kick_search_frames + 1,
                )
            )
        files = find_bundesliga_files(args.data_dir, match_id)
        metadata_by_match[match_id] = load_bundesliga_match_metadata(files["matchinfo"])
        frames = load_bundesliga_frames(files["positions"], target_frames=target_frames)
        frames_by_key.update(
            {
                (match_id, frame_id): frame
                for frame_id, frame in frames.items()
            }
        )

    args.vis_dir.mkdir(parents=True, exist_ok=True)
    for old_image in args.vis_dir.glob("rank_*.png"):
        old_image.unlink()

    outputs = []
    for summary_row, row in ranked_rows:
        rank = int(summary_row["rank"])
        match_id = str(row["match_id"])
        event_frame = int(row["event_frame"])
        scene_frames = [
            frame
            for (frame_match, _), frame in frames_by_key.items()
            if frame_match == match_id
            and event_frame - kick_search_frames - history_frames
            <= frame.frame_id
            <= event_frame + kick_search_frames
        ]
        kick = detect_kick_frame(
            scene_frames,
            str(row["passer_id"]),
            event_frame,
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
        frame = frames_by_key[(match_id, kick.kick_frame_id)]
        velocities = estimate_frame_velocities(
            scene_frames,
            kick.kick_frame_id,
            config,
        )
        ball_xy = (
            (frame.ball.x, frame.ball.y)
            if frame.ball is not None
            else (float(row["ball_x"]), float(row["ball_y"]))
        )
        metadata = metadata_by_match[match_id]
        team_id = str(row["team_id"])
        passer_id = str(row["passer_id"])
        runner_id = str(row["runner_id"])
        recipient_id = (
            str(row["recipient_id"])
            if "recipient_id" in row and pd.notna(row["recipient_id"])
            else None
        )
        existing_top_id = (
            str(row["defender_target_receiver_id"])
            if pd.notna(row.get("defender_target_receiver_id"))
            else None
        )

        candidate_regions = []
        for player_id, player in frame.players.items():
            if player.team_id != team_id or player_id == passer_id:
                continue
            if is_goalkeeper(metadata, player_id):
                continue
            region = reception_region_estimate(
                frame,
                velocities,
                player_id,
                passer_id,
                team_id,
                int(row["attacking_direction"]),
                ball_xy,
                config,
                region_config,
            )
            candidate_regions.append((player_id, region))
        candidate_regions.sort(key=lambda item: item[1].option_value, reverse=True)
        dynamic_top_id, dynamic_top_region = candidate_regions[0]
        rank_by_id = {
            player_id: index
            for index, (player_id, _region) in enumerate(candidate_regions, start=1)
        }
        region_by_id = dict(candidate_regions)
        recipient_region = region_by_id.get(recipient_id)
        runner_region = region_by_id.get(runner_id)
        existing_region = region_by_id.get(existing_top_id)
        outputs.append(
            {
                "rank": rank,
                "match_id": match_id,
                "event_frame": event_frame,
                "kick_frame": kick.kick_frame_id,
                "kick_release_frame": kick.release_frame_id,
                "kick_offset_frames": kick.offset_frames,
                "kick_detected": kick.detected,
                "kick_target_alignment": kick.target_alignment,
                "detected_release_speed_mps": kick.release_ball_speed_mps,
                "passer_name": player_name(metadata, passer_id),
                "actual_recipient_name": player_name(metadata, recipient_id),
                "actual_recipient_dynamic_rank": rank_by_id.get(recipient_id),
                "actual_recipient_dynamic_value": (
                    recipient_region.option_value if recipient_region is not None else None
                ),
                "existing_top_option_name": player_name(metadata, existing_top_id),
                "existing_top_dynamic_rank": rank_by_id.get(existing_top_id),
                "existing_top_dynamic_value": (
                    existing_region.option_value if existing_region is not None else None
                ),
                "dynamic_top_option_name": player_name(metadata, dynamic_top_id),
                "dynamic_top_option_value": dynamic_top_region.option_value,
                "dynamic_top_peak_x": dynamic_top_region.peak_x,
                "dynamic_top_peak_y": dynamic_top_region.peak_y,
                "runner_name": player_name(metadata, runner_id),
                "runner_dynamic_rank": rank_by_id.get(runner_id),
                "runner_dynamic_value": (
                    runner_region.option_value if runner_region is not None else None
                ),
                "existing_top_is_dynamic_top": existing_top_id == dynamic_top_id,
                "actual_recipient_is_dynamic_top": recipient_id == dynamic_top_id,
                "runner_is_dynamic_top": runner_id == dynamic_top_id,
                "dynamic_candidates": ";".join(
                    f"{player_name(metadata, player_id)}:{region.option_value:.6f}"
                    for player_id, region in candidate_regions
                ),
                "image_file": f"rank_{rank:02d}.png",
            }
        )
        plot_scene(
            args.vis_dir / f"rank_{rank:02d}.png",
            rank,
            frame,
            metadata,
            team_id,
            int(row["attacking_direction"]),
            ball_xy,
            passer_id,
            recipient_id,
            runner_id,
            existing_top_id,
            dynamic_top_id,
            dynamic_top_region,
        )
        print(
            f"Rank {rank:02d}: dynamic={player_name(metadata, dynamic_top_id)} "
            f"actual_rank={rank_by_id.get(recipient_id)} "
            f"runner_rank={rank_by_id.get(runner_id)}"
        )

    out = pd.DataFrame(outputs).sort_values("rank").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    out.to_csv(args.vis_dir / "summary.csv", index=False)
    print(f"Saved {len(out)} rows to {args.out}")
    print(f"Saved visualizations to {args.vis_dir}")


if __name__ == "__main__":
    main()
