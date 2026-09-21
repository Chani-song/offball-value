#!/usr/bin/env python3
"""Benchmark causal background-player rollouts on held-out Bundesliga tracking."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from offball_value.background_rollout import (
    BackgroundPrediction,
    BackgroundRolloutConfig,
    EmpiricalBackgroundPredictor,
    fit_damping_lambda,
    predict_constant_velocity,
    predict_damped_constant_velocity,
    predict_hold,
)
from offball_value.bundesliga import (
    FPS,
    BundesligaFrame,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.empirical_action_space import (
    EmpiricalPrimitiveConfig,
    EmpiricalPrimitiveLibrary,
    causal_motion_state_from_frames,
)


MODEL_ORDER = (
    "hold",
    "constant_velocity",
    "damped_constant_velocity",
    "empirical_reference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bundesliga-integrated"),
    )
    parser.add_argument("--match-id", default="J03WMX")
    parser.add_argument(
        "--scene-csv",
        type=Path,
        default=Path(
            "data/processed/scene_extractor_v0_1/DFL-MAT-J03WMX/"
            "scene_candidates.csv"
        ),
    )
    parser.add_argument(
        "--scene-review-csv",
        type=Path,
        default=Path(
            "data/processed/scene_extractor_v0_1/DFL-MAT-J03WMX/"
            "scene_animation_reviews_human.csv"
        ),
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(
            "data/processed/empirical_movement_library_v0_1/"
            "leave_out_DFL-MAT-J03WMX/primitives.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/background_rollout_benchmark_v0_1"),
    )
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--empirical-neighbors", type=int, default=512)
    parser.add_argument("--maximum-speed", type=float, default=9.0)
    return parser.parse_args()


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _selected_scenes(
    scene_csv: Path,
    review_csv: Path | None,
    limit: int | None,
) -> tuple[pd.DataFrame, list[int], list[int]]:
    scenes = pd.read_csv(scene_csv)
    accepted_ids = set(scenes.loc[_truthy(scenes["accepted"]), "frame_id"].astype(int))
    added: list[int] = []
    removed: list[int] = []
    if review_csv is not None and review_csv.exists():
        reviews = pd.read_csv(review_csv)
        for row in reviews.itertuples(index=False):
            frame_id = int(row.frame_id)
            review = str(row.review).strip().lower()
            if review == "valid" and frame_id not in accepted_ids:
                accepted_ids.add(frame_id)
                added.append(frame_id)
            elif review == "invalid" and frame_id in accepted_ids:
                accepted_ids.remove(frame_id)
                removed.append(frame_id)
    selected = scenes[scenes["frame_id"].astype(int).isin(accepted_ids)].copy()
    selected = selected.sort_values("frame_id")
    if limit is not None:
        selected = selected.head(limit)
    return selected, sorted(added), sorted(removed)


def _frame_offsets(seconds: float) -> tuple[int, int]:
    frame_offset = seconds * FPS
    return int(math.floor(frame_offset + 1e-9)), int(math.ceil(frame_offset - 1e-9))


def _required_frame_ids(
    scenes: pd.DataFrame,
    config: BackgroundRolloutConfig,
    history_seconds: float,
) -> set[int]:
    ids: set[int] = set()
    history_frames = int(round(history_seconds * FPS))
    for frame_id in scenes["frame_id"].astype(int):
        ids.update(range(frame_id - history_frames, frame_id + 1))
        for horizon in config.horizons_seconds:
            for time_s in (horizon, max(0.0, horizon - config.velocity_interval_seconds)):
                lower, upper = _frame_offsets(time_s)
                ids.add(frame_id + lower)
                ids.add(frame_id + upper)
    return ids


def _interpolated_position(
    frames: dict[int, BundesligaFrame],
    base_frame_id: int,
    player_id: str,
    seconds: float,
) -> tuple[float, float] | None:
    lower_offset, upper_offset = _frame_offsets(seconds)
    lower_frame = frames.get(base_frame_id + lower_offset)
    upper_frame = frames.get(base_frame_id + upper_offset)
    if lower_frame is None or upper_frame is None:
        return None
    lower = lower_frame.players.get(player_id)
    upper = upper_frame.players.get(player_id)
    if lower is None or upper is None:
        return None
    if lower_offset == upper_offset:
        return float(lower.x), float(lower.y)
    raw_offset = seconds * FPS
    fraction = (raw_offset - lower_offset) / (upper_offset - lower_offset)
    return (
        float(lower.x + fraction * (upper.x - lower.x)),
        float(lower.y + fraction * (upper.y - lower.y)),
    )


def _actual_state(
    frames: dict[int, BundesligaFrame],
    base_frame_id: int,
    player_id: str,
    horizon_seconds: float,
    velocity_interval_seconds: float,
) -> tuple[float, float, float, float] | None:
    endpoint = _interpolated_position(
        frames,
        base_frame_id,
        player_id,
        horizon_seconds,
    )
    previous_time = max(0.0, horizon_seconds - velocity_interval_seconds)
    previous = _interpolated_position(
        frames,
        base_frame_id,
        player_id,
        previous_time,
    )
    if endpoint is None or previous is None:
        return None
    interval = max(horizon_seconds - previous_time, 1e-9)
    return (
        endpoint[0],
        endpoint[1],
        float((endpoint[0] - previous[0]) / interval),
        float((endpoint[1] - previous[1]) / interval),
    )


def _heading_error_degrees(
    actual_vx: float,
    actual_vy: float,
    predicted_vx: float,
    predicted_vy: float,
    minimum_speed_mps: float = 0.5,
) -> float:
    actual_speed = math.hypot(actual_vx, actual_vy)
    predicted_speed = math.hypot(predicted_vx, predicted_vy)
    if actual_speed < minimum_speed_mps or predicted_speed < minimum_speed_mps:
        return math.nan
    difference = math.atan2(predicted_vy, predicted_vx) - math.atan2(
        actual_vy,
        actual_vx,
    )
    wrapped = (difference + math.pi) % (2.0 * math.pi) - math.pi
    return float(abs(math.degrees(wrapped)))


def _prediction_record(
    prediction: BackgroundPrediction,
    *,
    match_id: str,
    frame_id: int,
    player_id: str,
    player_name: str,
    team_id: str,
    team_phase: str,
    playing_position: str | None,
    is_goalkeeper: bool,
    is_ball_carrier: bool,
    start_x: float,
    start_y: float,
    initial_speed_mps: float,
    initial_longitudinal_acceleration_mps2: float,
    actual: tuple[float, float, float, float],
) -> dict[str, object]:
    actual_x, actual_y, actual_vx, actual_vy = actual
    actual_speed = math.hypot(actual_vx, actual_vy)
    error_x = prediction.x - actual_x
    error_y = prediction.y - actual_y
    return {
        "match_id": match_id,
        "frame_id": frame_id,
        "horizon_seconds": prediction.horizon_seconds,
        "model": prediction.model,
        "player_id": player_id,
        "player_name": player_name,
        "team_id": team_id,
        "team_phase": team_phase,
        "playing_position": playing_position,
        "is_goalkeeper": is_goalkeeper,
        "is_ball_carrier": is_ball_carrier,
        "start_x_m": start_x,
        "start_y_m": start_y,
        "initial_speed_mps": initial_speed_mps,
        "initial_longitudinal_acceleration_mps2": (
            initial_longitudinal_acceleration_mps2
        ),
        "actual_x_m": actual_x,
        "actual_y_m": actual_y,
        "predicted_x_m": prediction.x,
        "predicted_y_m": prediction.y,
        "error_x_m": error_x,
        "error_y_m": error_y,
        "position_error_m": math.hypot(error_x, error_y),
        "actual_vx_mps": actual_vx,
        "actual_vy_mps": actual_vy,
        "actual_speed_mps": actual_speed,
        "predicted_vx_mps": prediction.vx_mps,
        "predicted_vy_mps": prediction.vy_mps,
        "predicted_speed_mps": prediction.speed_mps,
        "speed_error_mps": abs(prediction.speed_mps - actual_speed),
        "heading_error_degrees": _heading_error_degrees(
            actual_vx,
            actual_vy,
            prediction.vx_mps,
            prediction.vy_mps,
        ),
        "boundary_clipped": prediction.boundary_clipped,
        "empirical_neighbor_count": prediction.empirical_neighbor_count,
        "empirical_mean_feature_distance": (
            prediction.empirical_mean_feature_distance
        ),
        "fallback_model": prediction.fallback_model,
    }


def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    if len(coordinates) < 2:
        return np.asarray([], dtype=float)
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=2))
    return distances[np.triu_indices(len(coordinates), k=1)]


def _offside_line(x_values: np.ndarray, attacking_direction: int) -> float:
    if len(x_values) < 2:
        return math.nan
    ordered = np.sort(x_values)
    return float(ordered[-2] if attacking_direction > 0 else ordered[1])


def _formation_metrics(
    predictions: pd.DataFrame,
    scene_directions: dict[int, int],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = predictions.groupby(
        ["match_id", "frame_id", "horizon_seconds", "model", "team_id"],
        sort=False,
    )
    for (match_id, frame_id, horizon, model, team_id), group in grouped:
        actual = group[["actual_x_m", "actual_y_m"]].to_numpy(dtype=float)
        predicted = group[["predicted_x_m", "predicted_y_m"]].to_numpy(dtype=float)
        actual_pairwise = _pairwise_distances(actual)
        predicted_pairwise = _pairwise_distances(predicted)
        phase = str(group["team_phase"].iloc[0])
        direction = int(scene_directions[int(frame_id)])
        record = {
            "match_id": match_id,
            "frame_id": int(frame_id),
            "horizon_seconds": float(horizon),
            "model": model,
            "team_id": team_id,
            "team_phase": phase,
            "player_count": len(group),
            "centroid_error_m": float(
                np.linalg.norm(np.mean(predicted, axis=0) - np.mean(actual, axis=0))
            ),
            "team_depth_error_m": float(
                abs(np.ptp(predicted[:, 0]) - np.ptp(actual[:, 0]))
            ),
            "team_width_error_m": float(
                abs(np.ptp(predicted[:, 1]) - np.ptp(actual[:, 1]))
            ),
            "pairwise_distance_mae_m": (
                float(np.mean(np.abs(predicted_pairwise - actual_pairwise)))
                if len(actual_pairwise)
                else math.nan
            ),
            "actual_minimum_pairwise_distance_m": (
                float(np.min(actual_pairwise)) if len(actual_pairwise) else math.nan
            ),
            "predicted_minimum_pairwise_distance_m": (
                float(np.min(predicted_pairwise)) if len(predicted_pairwise) else math.nan
            ),
            "offside_line_error_m": math.nan,
        }
        if phase == "out_of_possession":
            actual_line = _offside_line(actual[:, 0], direction)
            predicted_line = _offside_line(predicted[:, 0], direction)
            record["offside_line_error_m"] = abs(predicted_line - actual_line)
        records.append(record)
    return pd.DataFrame(records)


def _prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    scope_masks = {
        "all_players": np.ones(len(predictions), dtype=bool),
        "all_outfield": ~predictions["is_goalkeeper"].astype(bool).to_numpy(),
        "goalkeepers": predictions["is_goalkeeper"].astype(bool).to_numpy(),
        "in_possession_outfield": (
            (predictions["team_phase"] == "in_possession")
            & ~predictions["is_goalkeeper"].astype(bool)
        ).to_numpy(),
        "defending_outfield": (
            (predictions["team_phase"] == "out_of_possession")
            & ~predictions["is_goalkeeper"].astype(bool)
        ).to_numpy(),
        "ball_carriers": predictions["is_ball_carrier"].astype(bool).to_numpy(),
    }
    records: list[dict[str, object]] = []
    for scope, mask in scope_masks.items():
        scoped = predictions.loc[mask]
        for (model, horizon), group in scoped.groupby(
            ["model", "horizon_seconds"],
            sort=False,
        ):
            errors = group["position_error_m"].to_numpy(dtype=float)
            heading = group["heading_error_degrees"].dropna().to_numpy(dtype=float)
            records.append(
                {
                    "scope": scope,
                    "model": model,
                    "horizon_seconds": float(horizon),
                    "observation_count": len(group),
                    "scene_count": int(group["frame_id"].nunique()),
                    "mean_position_error_m": float(np.mean(errors)),
                    "median_position_error_m": float(np.median(errors)),
                    "p90_position_error_m": float(np.percentile(errors, 90)),
                    "rmse_position_error_m": float(np.sqrt(np.mean(errors**2))),
                    "mean_speed_error_mps": float(group["speed_error_mps"].mean()),
                    "median_heading_error_degrees": (
                        float(np.median(heading)) if len(heading) else math.nan
                    ),
                    "heading_observation_count": len(heading),
                    "boundary_clip_rate": float(
                        group["boundary_clipped"].astype(bool).mean()
                    ),
                }
            )
    result = pd.DataFrame(records)
    result["model"] = pd.Categorical(result["model"], MODEL_ORDER, ordered=True)
    return result.sort_values(["scope", "horizon_seconds", "model"]).assign(
        model=lambda frame: frame["model"].astype(str)
    )


def _formation_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (model, horizon, phase), group in metrics.groupby(
        ["model", "horizon_seconds", "team_phase"],
        sort=False,
    ):
        records.append(
            {
                "model": model,
                "horizon_seconds": float(horizon),
                "team_phase": phase,
                "team_scene_count": len(group),
                "mean_centroid_error_m": float(group["centroid_error_m"].mean()),
                "mean_team_depth_error_m": float(group["team_depth_error_m"].mean()),
                "mean_team_width_error_m": float(group["team_width_error_m"].mean()),
                "mean_pairwise_distance_mae_m": float(
                    group["pairwise_distance_mae_m"].mean()
                ),
                "mean_offside_line_error_m": float(
                    group["offside_line_error_m"].mean()
                ),
            }
        )
    result = pd.DataFrame(records)
    result["model"] = pd.Categorical(result["model"], MODEL_ORDER, ordered=True)
    return result.sort_values(["team_phase", "horizon_seconds", "model"]).assign(
        model=lambda frame: frame["model"].astype(str)
    )


def _model_selection(
    prediction_summary: pd.DataFrame,
    formation_summary: pd.DataFrame,
) -> pd.DataFrame:
    primary = prediction_summary[
        (prediction_summary["scope"] == "all_outfield")
        & np.isclose(prediction_summary["horizon_seconds"], 2.0)
    ].copy()
    formation = (
        formation_summary[np.isclose(formation_summary["horizon_seconds"], 2.0)]
        .groupby("model", as_index=False)
        .agg(
            mean_centroid_error_m=("mean_centroid_error_m", "mean"),
            mean_pairwise_distance_mae_m=("mean_pairwise_distance_mae_m", "mean"),
            mean_offside_line_error_m=("mean_offside_line_error_m", "mean"),
        )
    )
    result = primary.merge(formation, on="model", how="left")
    result = result.sort_values("mean_position_error_m").reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result["provisional_selection"] = result["rank"] == 1
    return result


def _paired_scene_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    outfield = predictions[~predictions["is_goalkeeper"].astype(bool)]
    pivot = outfield.pivot(
        index=["match_id", "frame_id", "horizon_seconds", "player_id"],
        columns="model",
        values="position_error_m",
    ).reset_index()
    records = []
    reference = "empirical_reference"
    for horizon in sorted(pivot["horizon_seconds"].unique()):
        horizon_frame = pivot[np.isclose(pivot["horizon_seconds"], horizon)]
        for comparator in MODEL_ORDER:
            if comparator == reference:
                continue
            paired = horizon_frame.dropna(subset=[reference, comparator]).copy()
            paired["comparator_minus_empirical_m"] = (
                paired[comparator] - paired[reference]
            )
            scene_delta = paired.groupby("frame_id")[
                "comparator_minus_empirical_m"
            ].mean()
            count = len(scene_delta)
            mean_delta = float(scene_delta.mean())
            standard_error = float(scene_delta.std(ddof=1) / math.sqrt(count))
            critical = float(stats.t.ppf(0.975, df=count - 1))
            records.append(
                {
                    "horizon_seconds": float(horizon),
                    "reference_model": reference,
                    "comparator_model": comparator,
                    "scene_count": count,
                    "player_observation_count": len(paired),
                    "mean_comparator_minus_empirical_m": mean_delta,
                    "ci95_lower_m": mean_delta - critical * standard_error,
                    "ci95_upper_m": mean_delta + critical * standard_error,
                    "fraction_scenes_empirical_lower_error": float(
                        (scene_delta > 0).mean()
                    ),
                    "interpretation": (
                        "positive favors empirical_reference; scene-clustered paired CI"
                    ),
                }
            )
    return pd.DataFrame(records)


def _plot_summary(
    prediction_summary: pd.DataFrame,
    formation_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    colors = {
        "hold": "#6b7280",
        "constant_velocity": "#2563eb",
        "damped_constant_velocity": "#d97706",
        "empirical_reference": "#059669",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    primary = prediction_summary[prediction_summary["scope"] == "all_outfield"]
    for model in MODEL_ORDER:
        values = primary[primary["model"] == model].sort_values("horizon_seconds")
        axes[0].plot(
            values["horizon_seconds"],
            values["mean_position_error_m"],
            marker="o",
            color=colors[model],
            label=model.replace("_", " "),
        )
    axes[0].set_title("Held-out outfield position error")
    axes[0].set_xlabel("Horizon (s)")
    axes[0].set_ylabel("Mean endpoint error (m)")
    axes[0].grid(alpha=0.25)

    formation = (
        formation_summary.groupby(["model", "horizon_seconds"], as_index=False)
        .agg(pairwise_mae=("mean_pairwise_distance_mae_m", "mean"))
    )
    for model in MODEL_ORDER:
        values = formation[formation["model"] == model].sort_values(
            "horizon_seconds"
        )
        axes[1].plot(
            values["horizon_seconds"],
            values["pairwise_mae"],
            marker="o",
            color=colors[model],
            label=model.replace("_", " "),
        )
    axes[1].set_title("Within-team geometry distortion")
    axes[1].set_xlabel("Horizon (s)")
    axes[1].set_ylabel("Pairwise-distance MAE (m)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Background rollout benchmark — held-out target match", fontsize=13)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    scenes, human_added, human_removed = _selected_scenes(
        args.scene_csv,
        args.scene_review_csv,
        args.scene_limit,
    )
    if scenes.empty:
        raise ValueError("No benchmark scenes were selected")

    config = BackgroundRolloutConfig(
        empirical_neighbor_count=args.empirical_neighbors,
        maximum_speed_mps=args.maximum_speed,
    )
    config.validate()
    library = EmpiricalPrimitiveLibrary.load(args.library)
    library.assert_excludes_match(match_id)
    empirical_predictor = EmpiricalBackgroundPredictor(library, config)
    damping_lambda, damping_mse = fit_damping_lambda(library)

    files = find_bundesliga_files(args.data_dir, match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    required_ids = _required_frame_ids(
        scenes,
        config,
        EmpiricalPrimitiveConfig().history_seconds,
    )
    frames = load_bundesliga_frames(files["positions"], required_ids)

    rows: list[dict[str, object]] = []
    scene_directions = {
        int(row.frame_id): int(row.attacking_direction)
        for row in scenes.itertuples(index=False)
    }
    history_frame_count = int(round(EmpiricalPrimitiveConfig().history_seconds * FPS))
    for scene_index, scene in enumerate(scenes.itertuples(index=False), start=1):
        frame_id = int(scene.frame_id)
        frame = frames.get(frame_id)
        if frame is None:
            continue
        history = [
            frames[item]
            for item in range(frame_id - history_frame_count, frame_id + 1)
            if item in frames
        ]
        for player_id, player in frame.players.items():
            state = causal_motion_state_from_frames(
                history,
                player_id,
                frame_id,
            )
            player_meta = metadata.players.get(player_id)
            playing_position = player_meta.position if player_meta else None
            is_goalkeeper = playing_position == "TW"
            empirical_by_horizon: dict[float, BackgroundPrediction]
            if is_goalkeeper:
                empirical_by_horizon = {
                    horizon: predict_hold(
                        player.x,
                        player.y,
                        horizon,
                        config,
                        model="empirical_reference",
                        fallback_model="hold_for_goalkeeper",
                    )
                    for horizon in config.horizons_seconds
                }
            else:
                empirical_by_horizon = {
                    prediction.horizon_seconds: prediction
                    for prediction in empirical_predictor.predict_many(
                        player.x,
                        player.y,
                        state,
                    )
                }

            for horizon in config.horizons_seconds:
                actual = _actual_state(
                    frames,
                    frame_id,
                    player_id,
                    horizon,
                    config.velocity_interval_seconds,
                )
                if actual is None:
                    continue
                predictions = (
                    predict_hold(player.x, player.y, horizon, config),
                    predict_constant_velocity(
                        player.x,
                        player.y,
                        state,
                        horizon,
                        config,
                    ),
                    predict_damped_constant_velocity(
                        player.x,
                        player.y,
                        state,
                        horizon,
                        damping_lambda,
                        config,
                    ),
                    empirical_by_horizon[horizon],
                )
                for prediction in predictions:
                    rows.append(
                        _prediction_record(
                            prediction,
                            match_id=match_id,
                            frame_id=frame_id,
                            player_id=player_id,
                            player_name=(
                                player_meta.short_name if player_meta else player_id
                            ),
                            team_id=player.team_id,
                            team_phase=(
                                "in_possession"
                                if player.team_id == scene.possession_team_id
                                else "out_of_possession"
                            ),
                            playing_position=playing_position,
                            is_goalkeeper=is_goalkeeper,
                            is_ball_carrier=player_id == scene.ball_carrier_id,
                            start_x=player.x,
                            start_y=player.y,
                            initial_speed_mps=state.speed_mps,
                            initial_longitudinal_acceleration_mps2=(
                                state.longitudinal_acceleration_mps2
                            ),
                            actual=actual,
                        )
                    )
        if scene_index % 10 == 0 or scene_index == len(scenes):
            print(f"[{scene_index}/{len(scenes)}] benchmark scenes", flush=True)

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("No held-out predictions could be evaluated")
    formation_metrics = _formation_metrics(predictions, scene_directions)
    prediction_summary = _prediction_summary(predictions)
    formation_summary = _formation_summary(formation_metrics)
    selection = _model_selection(prediction_summary, formation_summary)
    paired_comparisons = _paired_scene_comparisons(predictions)

    output_dir = args.output_dir / match_id
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "player_predictions.csv", index=False)
    prediction_summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    formation_metrics.to_csv(output_dir / "formation_metrics.csv", index=False)
    formation_summary.to_csv(output_dir / "formation_summary.csv", index=False)
    selection.to_csv(output_dir / "model_selection.csv", index=False)
    paired_comparisons.to_csv(output_dir / "paired_scene_comparisons.csv", index=False)
    _plot_summary(
        prediction_summary,
        formation_summary,
        output_dir / "benchmark_summary.png",
    )

    selected_model = str(selection.iloc[0]["model"])
    manifest = {
        "target_match_id": match_id,
        "source_match_ids": list(library.source_match_ids),
        "target_match_excluded_from_library": match_id not in library.source_match_ids,
        "scene_count": int(scenes["frame_id"].nunique()),
        "human_valid_overrides_added": human_added,
        "human_invalid_overrides_removed": human_removed,
        "player_prediction_rows": len(predictions),
        "config": config.__dict__,
        "fitted_damping_lambda_per_second": damping_lambda,
        "fitted_damping_minimum_mse_m2": float(np.min(damping_mse)),
        "primary_selection_metric": "2.0 s mean position error, all outfield players",
        "provisional_selected_model": selected_model,
        "paired_uncertainty": (
            "95% t intervals over per-scene paired mean error differences"
        ),
        "empirical_reference_definition": (
            "Gaussian-kernel conditional mean over leave-one-match-out primitives, "
            "conditioned on causal speed and longitudinal acceleration"
        ),
        "goalkeeper_empirical_fallback": "hold",
        "observed_future_usage": "held-out benchmark target only",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n2.0 s all-outfield model selection")
    print(
        selection[
            [
                "rank",
                "model",
                "mean_position_error_m",
                "p90_position_error_m",
                "mean_pairwise_distance_mae_m",
                "mean_offside_line_error_m",
            ]
        ].to_string(index=False)
    )
    print(f"\nselected: {selected_model}")
    print(f"output:   {output_dir}")


if __name__ == "__main__":
    main()
