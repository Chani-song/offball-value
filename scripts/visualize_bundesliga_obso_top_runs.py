from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from offball_value.bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    BundesligaFrame,
    BundesligaMatchMeta,
    BundesligaObjectState,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.obso import counterfactual_frame_for_runner


DEFAULT_RESULTS = ROOT / "data" / "processed" / "bundesliga_obso_counterfactual_all.csv"
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "bundesliga-integrated"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "visualizations" / "bundesliga_obso_top_runs"


def player_name(metadata: BundesligaMatchMeta, player_id: str | None) -> str:
    if not player_id:
        return ""
    player = metadata.players.get(player_id)
    return player.short_name if player else player_id


def player_label(metadata: BundesligaMatchMeta, player_id: str) -> str:
    player = metadata.players.get(player_id)
    if player is None:
        return player_id[-4:]
    return player.shirt_number or player.short_name[:6]


def parse_top_options(value: str | float | None) -> list[tuple[str, str, float]]:
    if value is None or pd.isna(value):
        return []
    out = []
    for item in str(value).split(";"):
        parts = item.split(":")
        if len(parts) != 3:
            continue
        player_id, name, score = parts
        try:
            out.append((player_id, name, float(score)))
        except ValueError:
            continue
    return out


def closest_defender_for_runner(
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    runner_id: str,
) -> BundesligaObjectState | None:
    runner = end_frame.players.get(runner_id)
    if runner is None:
        return None
    defenders = [
        player for player in end_frame.players.values()
        if player.team_id != runner.team_id and player.object_id in start_frame.players
    ]
    if not defenders:
        return None
    return min(defenders, key=lambda player: (player.x - runner.x) ** 2 + (player.y - runner.y) ** 2)


def affected_defender_from_row(
    row: pd.Series,
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    runner_id: str,
) -> BundesligaObjectState | None:
    if "affected_defender_id" in row and pd.notna(row["affected_defender_id"]):
        defender_id = str(row["affected_defender_id"])
        defender = end_frame.players.get(defender_id)
        if defender is not None and defender.object_id in start_frame.players:
            return defender
    return closest_defender_for_runner(start_frame, end_frame, runner_id)


def draw_pitch(ax) -> None:
    half_l = FIELD_LENGTH / 2.0
    half_w = FIELD_WIDTH / 2.0
    ax.set_xlim(-half_l, half_l)
    ax.set_ylim(-half_w, half_w)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#eef4ec")
    ax.axis("off")

    line = "#526157"
    lw = 1.0
    ax.plot([-half_l, half_l, half_l, -half_l, -half_l], [-half_w, -half_w, half_w, half_w, -half_w], color=line, lw=lw)
    ax.plot([0, 0], [-half_w, half_w], color=line, lw=lw)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color=line, lw=lw))

    box_depth = 16.5
    box_width = 40.32
    six_depth = 5.5
    six_width = 18.32
    for side in [-1, 1]:
        goal_x = side * half_l
        rect_x = goal_x - side * box_depth if side > 0 else goal_x
        six_x = goal_x - side * six_depth if side > 0 else goal_x
        ax.add_patch(plt.Rectangle((rect_x, -box_width / 2), box_depth, box_width, fill=False, color=line, lw=lw))
        ax.add_patch(plt.Rectangle((six_x, -six_width / 2), six_depth, six_width, fill=False, color=line, lw=lw))


def scatter_team(
    ax,
    frame: BundesligaFrame,
    metadata: BundesligaMatchMeta,
    attacking_team_id: str,
    top_ids: set[str],
    muted: bool = False,
) -> None:
    for player in frame.players.values():
        is_attacker = player.team_id == attacking_team_id
        is_top = player.object_id in top_ids
        if is_attacker:
            color = "#2f80ed" if not muted else "#93c5fd"
            edge = "#114f9d"
            size = 58
        else:
            color = "#9aa3ad" if not muted else "#d1d5db"
            edge = "#4b5563"
            size = 52
        if is_top:
            color = "#14b8a6"
            edge = "#0f766e"
            size = 105
        alpha = 0.35 if muted else 0.95
        ax.scatter([player.x], [player.y], s=size, c=color, edgecolors=edge, linewidths=0.8, alpha=alpha, zorder=4 if is_top else 3)
        if is_top or not muted:
            ax.text(
                player.x,
                player.y - 1.3,
                player_label(metadata, player.object_id),
                ha="center",
                va="top",
                fontsize=6.5,
                color="#17212b",
                alpha=0.95 if not muted else 0.4,
                zorder=5,
            )


def draw_special_marker(ax, player: BundesligaObjectState | None, label: str, color: str, marker: str = "o") -> None:
    if player is None:
        return
    kwargs = {"s": 150, "c": color, "linewidths": 1.1, "marker": marker, "zorder": 8}
    if marker != "x":
        kwargs["edgecolors"] = "#111827"
    ax.scatter([player.x], [player.y], **kwargs)
    ax.text(player.x, player.y + 1.6, label, ha="center", va="bottom", fontsize=7.2, color="#111827", zorder=9)


def draw_runner_arrow(ax, start: BundesligaObjectState | None, end: BundesligaObjectState | None, color: str = "#f97316") -> None:
    if start is None or end is None:
        return
    ax.annotate(
        "",
        xy=(end.x, end.y),
        xytext=(start.x, start.y),
        arrowprops={"arrowstyle": "->", "lw": 2.6, "color": color, "shrinkA": 0, "shrinkB": 0},
        zorder=7,
    )


def draw_ball(ax, x: float, y: float) -> None:
    ax.scatter([x], [y], s=72, c="#facc15", edgecolors="#713f12", linewidths=0.9, zorder=9)
    ax.text(x, y + 1.4, "ball", ha="center", va="bottom", fontsize=6.5, color="#713f12", zorder=9)


def describe_options(options: list[tuple[str, str, float]]) -> str:
    return " | ".join(f"{name} {score:.3f}" for _, name, score in options[:3])


def plot_scene(
    row: pd.Series,
    rank: int,
    metadata: BundesligaMatchMeta,
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    out_file: Path,
    runner_alpha: float,
    defender_alpha: float,
) -> dict[str, object]:
    runner_id = str(row["runner_id"])
    passer_id = str(row["passer_id"])
    recipient_id = str(row["recipient_id"]) if pd.notna(row["recipient_id"]) else None
    attacking_team_id = str(row["team_id"])
    ball_x = float(row["ball_x"])
    ball_y = float(row["ball_y"])
    actual_options = parse_top_options(row.get("actual_top_options"))
    cf_options = parse_top_options(row.get("counterfactual_top_options"))
    actual_top_ids = {player_id for player_id, _, _ in actual_options}
    cf_top_ids = {player_id for player_id, _, _ in cf_options}
    defender = affected_defender_from_row(row, start_frame, end_frame, runner_id)
    target_receiver_id = row.get("defender_target_receiver_id")
    if target_receiver_id is not None and pd.isna(target_receiver_id):
        target_receiver_id = None
    defender_selection = row.get("defender_selection", "closest-runner")
    if defender_selection is None or pd.isna(defender_selection):
        defender_selection = "closest-runner"

    cf_frame = counterfactual_frame_for_runner(
        start_frame,
        end_frame,
        runner_id,
        runner_alpha=runner_alpha,
        defender_alpha=defender_alpha,
        affected_defender_id=defender.object_id if defender is not None else None,
        target_receiver_id=str(target_receiver_id) if target_receiver_id is not None else None,
        attacking_team_id=attacking_team_id,
        ball_xy=(ball_x, ball_y),
        attacking_direction=int(row["attacking_direction"]) if "attacking_direction" in row else 1,
        defender_selection=str(defender_selection),
    )
    cf_defender = cf_frame.players.get(defender.object_id) if defender is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 8.5), dpi=160)
    for ax, frame, title, top_ids in [
        (axes[0], end_frame, "actual", actual_top_ids),
        (axes[1], cf_frame, "counterfactual", cf_top_ids),
    ]:
        draw_pitch(ax)
        scatter_team(ax, start_frame, metadata, attacking_team_id, set(), muted=True)
        scatter_team(ax, frame, metadata, attacking_team_id, top_ids, muted=False)
        draw_ball(ax, ball_x, ball_y)
        draw_runner_arrow(ax, start_frame.players.get(runner_id), frame.players.get(runner_id))
        draw_special_marker(ax, frame.players.get(runner_id), "runner", "#f97316")
        draw_special_marker(ax, frame.players.get(passer_id), "passer", "#f59e0b", marker="s")
        draw_special_marker(ax, frame.players.get(recipient_id), "recipient", "#14b8a6", marker="^")
        if title == "actual":
            draw_special_marker(ax, defender, "defender", "#d946ef", marker="D")
        else:
            draw_special_marker(ax, cf_defender, "def cf", "#d946ef", marker="x")
        ax.set_title(title, fontsize=12, color="#111827")

    match_title = f"{metadata.home_team_name} vs {metadata.away_team_name}"
    delta = float(row["draft_obso_offball_value"])
    actual = float(row["actual_team_obso_topk"])
    counterfactual = float(row["counterfactual_team_obso_topk"])
    space_delta = row.get("draft_post_reception_space_offball_value_m2")
    space_text = ""
    if space_delta is not None and pd.notna(space_delta):
        space_text = f" | post-space {float(space_delta):+.1f}m2"
    title = (
        f"Rank {rank} | {match_title} | frame {int(row['event_frame'])} | "
        f"OBSO off-ball value {delta:+.4f}{space_text}"
    )
    subtitle = (
        f"passer {player_name(metadata, passer_id)} -> recipient {player_name(metadata, recipient_id)} | "
        f"non-receiving runner {player_name(metadata, runner_id)} | "
        f"actual top-k {actual:.4f}, counterfactual {counterfactual:.4f}"
    )
    fig.suptitle(title, fontsize=13, y=0.98, color="#111827")
    fig.text(0.5, 0.945, subtitle, ha="center", va="center", fontsize=9.5, color="#111827")
    fig.text(0.25, 0.035, f"actual top options: {describe_options(actual_options)}", ha="center", fontsize=8, color="#0f766e")
    fig.text(0.75, 0.035, f"counterfactual top options: {describe_options(cf_options)}", ha="center", fontsize=8, color="#0f766e")
    fig.text(0.5, 0.012, "faded dots show start frame; orange arrow shows runner movement; magenta marks the defender affected by the rollback", ha="center", fontsize=8, color="#374151")
    fig.tight_layout(rect=[0, 0.055, 1, 0.92])
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    return {
        "rank": rank,
        "match_id": row["match_id"],
        "event_frame": int(row["event_frame"]),
        "runner_name": player_name(metadata, runner_id),
        "passer_name": player_name(metadata, passer_id),
        "recipient_name": player_name(metadata, recipient_id),
        "affected_defender_id": defender.object_id if defender else None,
        "affected_defender": player_name(metadata, defender.object_id if defender else None),
        "affected_defender_distance_to_runner_m": (
            float(row["affected_defender_distance_to_runner_m"])
            if "affected_defender_distance_to_runner_m" in row and pd.notna(row["affected_defender_distance_to_runner_m"])
            else (
                ((defender.x - end_frame.players[runner_id].x) ** 2 + (defender.y - end_frame.players[runner_id].y) ** 2) ** 0.5
                if defender is not None and runner_id in end_frame.players
                else None
            )
        ),
        "defender_selection": row.get("defender_selection") if "defender_selection" in row else None,
        "defender_target_receiver_name": (
            row.get("defender_target_receiver_name") if "defender_target_receiver_name" in row else None
        ),
        "actual_team_obso_topk": actual,
        "counterfactual_team_obso_topk": counterfactual,
        "draft_obso_offball_value": delta,
        "draft_post_reception_space_offball_value_m2": (
            float(space_delta) if space_delta is not None and pd.notna(space_delta) else None
        ),
        "image_file": out_file.name,
    }


def load_frames_for_top_rows(data_dir: Path, rows: pd.DataFrame) -> tuple[dict[str, BundesligaMatchMeta], dict[tuple[str, int], BundesligaFrame]]:
    metadata_by_match: dict[str, BundesligaMatchMeta] = {}
    frames_by_key: dict[tuple[str, int], BundesligaFrame] = {}

    for match_id, group in rows.groupby("match_id"):
        files = find_bundesliga_files(data_dir, match_id)
        metadata = load_bundesliga_match_metadata(files["matchinfo"])
        metadata_by_match[match_id] = metadata
        target_frames = set(group["start_frame"].astype(int)) | set(group["event_frame"].astype(int))
        frames = load_bundesliga_frames(files["positions"], target_frames=target_frames)
        for frame_id, frame in frames.items():
            frames_by_key[(match_id, frame_id)] = frame
    return metadata_by_match, frames_by_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize top Bundesliga OBSO off-ball counterfactual candidates.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--include-nonpositive", action="store_true")
    parser.add_argument("--runner-alpha", type=float, default=0.0)
    parser.add_argument("--defender-alpha", type=float, default=0.3)
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"{args.results} does not exist. Run run_bundesliga_obso_demo.py first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.results).sort_values("draft_obso_offball_value", ascending=False)
    if not args.include_nonpositive:
        rows = rows[rows["draft_obso_offball_value"] > 0]
    rows = rows.head(args.top_k).reset_index(drop=True)
    if rows.empty:
        print("No rows to visualize.")
        return

    metadata_by_match, frames_by_key = load_frames_for_top_rows(args.data_dir, rows)

    summaries = []
    for idx, row in rows.iterrows():
        rank = idx + 1
        match_id = row["match_id"]
        start_frame = frames_by_key[(match_id, int(row["start_frame"]))]
        end_frame = frames_by_key[(match_id, int(row["event_frame"]))]
        out_file = args.out_dir / f"rank_{rank:02d}_{match_id}_frame_{int(row['event_frame'])}.png"
        summaries.append(
            plot_scene(
                row=row,
                rank=rank,
                metadata=metadata_by_match[match_id],
                start_frame=start_frame,
                end_frame=end_frame,
                out_file=out_file,
                runner_alpha=args.runner_alpha,
                defender_alpha=args.defender_alpha,
            )
        )

    summary = pd.DataFrame(summaries)
    summary_file = args.out_dir / "summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved {len(summary)} visualizations to {args.out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
