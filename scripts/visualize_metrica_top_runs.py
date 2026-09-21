from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from offball_value.baseline import PlayerState, compute_toy_option_score, euclidean
from offball_value.loaders import load_metrica_tracking


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "data" / "processed" / "demo_offball_counterfactual_game1.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "visualizations" / "metrica_top_runs"
METRICA_DIR = ROOT / "data" / "raw" / "metrica-sample-data" / "data"


def merge_metrica_tracking(game: int) -> pd.DataFrame:
    home = load_metrica_tracking(METRICA_DIR, game=game, team="Home")
    away = load_metrica_tracking(METRICA_DIR, game=game, team="Away")

    if len(home) != len(away):
        raise ValueError(f"Home/Away tracking lengths differ: {len(home)} vs {len(away)}")

    key_cols = [c for c in ["Frame", "Time [s]", "Period"] if c in home.columns]
    away_extra = away.drop(
        columns=[c for c in away.columns if c in key_cols or c.startswith("ball_")],
        errors="ignore",
    )
    return pd.concat([home.reset_index(drop=True), away_extra.reset_index(drop=True)], axis=1)


def row_players(row: pd.Series, team: str) -> list[PlayerState]:
    prefix = f"{team}_"
    players: list[PlayerState] = []
    for col in row.index:
        if not col.startswith(prefix) or not col.endswith("_x"):
            continue
        player_id = col[:-2]
        x = row[col]
        y = row.get(f"{player_id}_y")
        if pd.isna(x) or pd.isna(y):
            continue
        players.append(PlayerState(player_id=player_id, x=float(x), y=float(y)))
    return players


def split_teams(row: pd.Series, attacking_team: str) -> tuple[list[PlayerState], list[PlayerState]]:
    defending_team = "Away" if attacking_team == "Home" else "Home"
    return row_players(row, attacking_team), row_players(row, defending_team)


def ball_xy(row: pd.Series) -> tuple[float, float] | None:
    x = row.get("ball_x")
    y = row.get("ball_y")
    if pd.isna(x) or pd.isna(y):
        return None
    return float(x), float(y)


def player_map(players: list[PlayerState]) -> dict[str, PlayerState]:
    return {p.player_id: p for p in players}


def closest_defender_to_runner(
    defenders_start: dict[str, PlayerState],
    defenders_end: list[PlayerState],
    runner_end: PlayerState,
) -> tuple[PlayerState, PlayerState | None]:
    closest = min(
        defenders_end,
        key=lambda d: euclidean(d.x, d.y, runner_end.x, runner_end.y),
    )
    start = defenders_start.get(closest.player_id)
    if start is None:
        return closest, None

    counterfactual = PlayerState(
        player_id=closest.player_id,
        x=0.7 * start.x + 0.3 * closest.x,
        y=0.7 * start.y + 0.3 * closest.y,
    )
    return closest, counterfactual


def nearest_defender(player: PlayerState, defenders: list[PlayerState]) -> tuple[PlayerState, float]:
    defender = min(defenders, key=lambda d: euclidean(player.x, player.y, d.x, d.y))
    return defender, euclidean(player.x, player.y, defender.x, defender.y)


def draw_pitch(ax) -> None:
    ax.set_facecolor("#f7f5ee")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    line_color = "#555555"
    lw = 1.1
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=line_color, lw=lw)
    ax.plot([0.5, 0.5], [0, 1], color=line_color, lw=lw)
    ax.add_patch(plt.Circle((0.5, 0.5), 0.0915, fill=False, color=line_color, lw=lw))

    box_w = 0.44
    six_w = 0.20
    ax.add_patch(plt.Rectangle((0, 0.5 - box_w / 2), 0.165, box_w, fill=False, color=line_color, lw=lw))
    ax.add_patch(plt.Rectangle((0, 0.5 - six_w / 2), 0.055, six_w, fill=False, color=line_color, lw=lw))
    ax.add_patch(plt.Rectangle((1 - 0.165, 0.5 - box_w / 2), 0.165, box_w, fill=False, color=line_color, lw=lw))
    ax.add_patch(plt.Rectangle((1 - 0.055, 0.5 - six_w / 2), 0.055, six_w, fill=False, color=line_color, lw=lw))


def scatter_players(ax, players: list[PlayerState], color: str, edgecolor: str, label_prefix: str) -> None:
    ax.scatter(
        [p.x for p in players],
        [p.y for p in players],
        s=70,
        c=color,
        edgecolors=edgecolor,
        linewidths=0.8,
        zorder=3,
    )
    for p in players:
        short_id = p.player_id.replace(label_prefix, "")
        ax.text(
            p.x,
            p.y - 0.018,
            short_id,
            ha="center",
            va="top",
            fontsize=6,
            color="#1f2933",
            zorder=4,
        )


def draw_arrow(ax, start: PlayerState, end: PlayerState, color: str, linewidth: float = 2.8) -> None:
    ax.annotate(
        "",
        xy=(end.x, end.y),
        xytext=(start.x, start.y),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": linewidth,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=6,
    )


def draw_scene(
    result_row: pd.Series,
    start_row: pd.Series,
    end_row: pd.Series,
    rank: int,
    out_file: Path,
    attacking_team: str,
) -> dict[str, float | str]:
    attackers0, defenders0 = split_teams(start_row, attacking_team=attacking_team)
    attackers1, defenders1 = split_teams(end_row, attacking_team=attacking_team)

    attackers0_by_id = player_map(attackers0)
    attackers1_by_id = player_map(attackers1)
    defenders0_by_id = player_map(defenders0)

    runner_id = str(result_row["runner_id"])
    receiver_id = str(result_row["best_actual_receiver"])
    runner_start = attackers0_by_id[runner_id]
    runner_end = attackers1_by_id[runner_id]
    receiver_end = attackers1_by_id[receiver_id]
    affected_defender, affected_counterfactual = closest_defender_to_runner(
        defenders_start=defenders0_by_id,
        defenders_end=defenders1,
        runner_end=runner_end,
    )

    nearest_actual, actual_receiver_sep = nearest_defender(receiver_end, defenders1)
    cf_defenders = [
        affected_counterfactual if affected_counterfactual is not None and d.player_id == affected_defender.player_id else d
        for d in defenders1
    ]
    nearest_cf, cf_receiver_sep = nearest_defender(receiver_end, cf_defenders)
    recomputed_actual_score = compute_toy_option_score(receiver_end, defenders1, attacking_direction=1)
    recomputed_cf_score = compute_toy_option_score(receiver_end, cf_defenders, attacking_direction=1)

    fig, ax = plt.subplots(figsize=(10, 7.8), dpi=160)
    draw_pitch(ax)

    scatter_players(ax, defenders1, color="#b8bec6", edgecolor="#606975", label_prefix="Away_")
    scatter_players(ax, attackers1, color="#4d8fd7", edgecolor="#15508e", label_prefix="Home_")

    ax.scatter([p.x for p in attackers0], [p.y for p in attackers0], s=30, c="#4d8fd7", alpha=0.18, zorder=2)
    ax.scatter([p.x for p in defenders0], [p.y for p in defenders0], s=30, c="#606975", alpha=0.18, zorder=2)

    draw_arrow(ax, runner_start, runner_end, color="#f97316", linewidth=3.5)

    ax.scatter([runner_end.x], [runner_end.y], s=160, c="#f97316", edgecolors="#7c2d12", linewidths=1.2, zorder=7)
    ax.scatter([receiver_end.x], [receiver_end.y], s=170, c="#14b8a6", edgecolors="#0f766e", linewidths=1.2, zorder=7)
    ax.scatter([affected_defender.x], [affected_defender.y], s=150, c="#d946ef", edgecolors="#86198f", linewidths=1.1, zorder=7)

    ball = ball_xy(end_row)
    if ball is not None:
        ax.scatter([ball[0]], [ball[1]], s=90, c="#facc15", edgecolors="#854d0e", linewidths=0.9, zorder=8)
        ax.text(ball[0], ball[1] + 0.018, "ball", ha="center", va="bottom", fontsize=7, color="#713f12")

    if affected_counterfactual is not None:
        ax.scatter(
            [affected_counterfactual.x],
            [affected_counterfactual.y],
            s=115,
            marker="x",
            c="#86198f",
            linewidths=2.2,
            zorder=8,
        )
        ax.plot(
            [affected_defender.x, affected_counterfactual.x],
            [affected_defender.y, affected_counterfactual.y],
            color="#86198f",
            linestyle="--",
            lw=1.7,
            zorder=5,
        )

    ax.plot(
        [receiver_end.x, nearest_actual.x],
        [receiver_end.y, nearest_actual.y],
        color="#0f766e",
        linestyle="-",
        lw=1.8,
        alpha=0.8,
        zorder=5,
    )
    if nearest_cf.player_id != nearest_actual.player_id or affected_counterfactual is not None:
        ax.plot(
            [receiver_end.x, nearest_cf.x],
            [receiver_end.y, nearest_cf.y],
            color="#0f766e",
            linestyle=":",
            lw=1.8,
            alpha=0.8,
            zorder=5,
        )

    title = (
        f"Rank {rank} | frames {int(result_row['frame_start'])}-{int(result_row['frame_end'])} | "
        f"{float(result_row['time_start_s']):.2f}s-{float(result_row['time_end_s']):.2f}s | "
        f"draft value {float(result_row['draft_offball_value']):.4f}"
    )
    subtitle_1 = f"runner {runner_id} -> receiver {receiver_id}; affected defender {affected_defender.player_id}"
    subtitle_2 = (
        f"receiver sep actual {actual_receiver_sep:.3f}, cf {cf_receiver_sep:.3f}; "
        f"score actual {recomputed_actual_score:.4f}, cf {recomputed_cf_score:.4f}"
    )
    fig.suptitle(title, fontsize=10, color="#111827", y=0.985)
    fig.text(0.5, 0.948, subtitle_1, ha="center", va="center", fontsize=8.5, color="#111827")
    fig.text(0.5, 0.923, subtitle_2, ha="center", va="center", fontsize=8.5, color="#111827")

    fig.text(0.08, 0.035, "orange arrow: runner movement", fontsize=8, color="#9a3412")
    fig.text(0.33, 0.035, "teal: selected receiver", fontsize=8, color="#0f766e")
    fig.text(0.52, 0.035, "magenta: defender actual / x counterfactual", fontsize=8, color="#86198f")
    fig.text(0.82, 0.035, "faded: start frame", fontsize=8, color="#374151")

    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    return {
        "rank": rank,
        "frame_start": int(result_row["frame_start"]),
        "frame_end": int(result_row["frame_end"]),
        "runner_id": runner_id,
        "receiver_id": receiver_id,
        "affected_defender": affected_defender.player_id,
        "draft_offball_value": float(result_row["draft_offball_value"]),
        "receiver_sep_actual": actual_receiver_sep,
        "receiver_sep_counterfactual": cf_receiver_sep,
        "score_actual_recomputed": recomputed_actual_score,
        "score_counterfactual_recomputed": recomputed_cf_score,
        "image_file": out_file.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=1)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--attacking-team", type=str, default="Home")
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(
            f"{args.results} does not exist. Run scripts/demo_offball_counterfactual.py first."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results).head(args.top_k)

    needed_frame_ids = set(results["frame_start"].astype(int)) | set(results["frame_end"].astype(int))
    tracking = merge_metrica_tracking(game=args.game)
    needed = tracking[tracking["Frame"].astype(int).isin(needed_frame_ids)]
    rows_by_frame = {
        int(row["Frame"]): row
        for _, row in needed.iterrows()
    }

    summaries = []
    for idx, row in results.iterrows():
        rank = idx + 1
        frame_start = int(row["frame_start"])
        frame_end = int(row["frame_end"])
        if frame_start not in rows_by_frame or frame_end not in rows_by_frame:
            raise KeyError(f"Missing frame pair {frame_start}-{frame_end}")

        out_file = args.out_dir / f"rank_{rank:02d}_frames_{frame_start}_{frame_end}.png"
        summaries.append(
            draw_scene(
                result_row=row,
                start_row=rows_by_frame[frame_start],
                end_row=rows_by_frame[frame_end],
                rank=rank,
                out_file=out_file,
                attacking_team=args.attacking_team,
            )
        )

    summary = pd.DataFrame(summaries)
    summary_file = args.out_dir / "summary.csv"
    summary.to_csv(summary_file, index=False)

    print(f"Saved {len(summary)} images to {args.out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
