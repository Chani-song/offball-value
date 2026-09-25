#!/usr/bin/env python3
"""Do the two delivery models disagree about the defender's own responses?

The question is not which model scores better on StatsBomb passes — that was
settled by the stratified comparison — but whether swapping the model changes
anything the pipeline actually does.  The pipeline's only channel from a
defender's trajectory to P is the release-moment configuration, so the test is:
hold the release time, the target and every other player fixed, vary only the
candidate defender's counterfactual position, and ask how far P moves.

The hybrid's answer is already in the audit (`delivery` per cell).  The 360
model is recomputed here on the same cells, so the two see identical geometry
and any reconstruction error is common to both.

Carry options are excluded: their release point rides the carry path rather
than sitting on the observed ball, so reconstructing it from background frames
would give the 360 model a start the hybrid never saw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from offball_value.xpass import (
    lane_features,
    load_xpass_model,
    to_statsbomb_coordinates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/processed/xpass_360_nochoice/xpass_360_hist_gbdt.joblib"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def frame_lookup(scene):
    """relative_time_s -> {object_id: (team_id, x, y)} plus the ball."""
    frames = {}
    for frame in scene["background_frames"]:
        players = {
            str(row[0]): (str(row[1]), float(row[2]), float(row[3]))
            for row in frame["players"]
        }
        ball = frame.get("ball")
        frames[round(float(frame["relative_time_s"]), 2)] = (
            players,
            (float(ball[0]), float(ball[1])) if ball else None,
        )
    return frames


def nearest_time(frames, t):
    return min(frames, key=lambda u: abs(u - t))


def goalkeeper_id(players, attacking_team, attacking_direction):
    """The defending player deepest toward his own goal at this frame."""
    defenders = [
        (oid, xy) for oid, xy in players.items() if xy[0] != attacking_team
    ]
    defending = [
        (oid, value) for oid, value in players.items() if value[0] != attacking_team
    ]
    if not defending:
        return None
    # The attack runs toward +x when attacking_direction >= 0, so the
    # defending keeper sits at the largest x.
    sign = 1 if int(attacking_direction) >= 0 else -1
    return max(defending, key=lambda item: sign * item[1][1])[0]


def path_position(path_txy, t):
    best = min(path_txy, key=lambda row: abs(float(row[0]) - t))
    return float(best[1]), float(best[2])


def collect(scene, model):
    frames = frame_lookup(scene)
    attacking_team = str(scene["attacking_team_id"])
    direction = int(scene["attacking_direction"])
    zero_players, _ = frames[nearest_time(frames, 0.0)]
    keeper = goalkeeper_id(zero_players, attacking_team, direction)

    rows = []
    for defender in scene["candidate_defenders"]:
        defender_id = str(defender["defender_id"])
        carrier_option_ids = {
            str(option["option_id"])
            for option in defender["options"]
            if option.get("is_carrier")
        }
        for response in defender["responses"]:
            for option_id, cell in response.get("cells", {}).items():
                if option_id in carrier_option_ids:
                    continue
                if not cell.get("legal") or cell.get("delivery") is None:
                    continue
                release = float(cell["release_time_s"])
                target = (float(cell["event_xy"][0]), float(cell["event_xy"][1]))
                key = nearest_time(frames, release)
                players, ball = frames[key]
                if ball is None:
                    continue
                opponents = []
                for oid, (team, x, y) in players.items():
                    if team == attacking_team or oid == keeper:
                        continue
                    if oid == defender_id:
                        x, y = path_position(response["path_txy"], release)
                    opponents.append((x, y))
                rows.append(
                    {
                        "scene": f"{scene['match_id']}:{scene['onset_frame_id']}",
                        "defender": defender_id,
                        "option": option_id,
                        "response": str(response["response_id"]),
                        "hybrid": float(cell["delivery"]),
                        "start": ball,
                        "end": target,
                        "opponents": opponents,
                    }
                )
    if not rows:
        return rows
    features = np.stack(
        [
            np.concatenate(
                [
                    _pass_features(row, direction),
                    lane_features(
                        to_statsbomb_coordinates(row["start"], direction),
                        to_statsbomb_coordinates(row["end"], direction),
                        [
                            to_statsbomb_coordinates(xy, direction)
                            for xy in row["opponents"]
                        ],
                    ),
                ]
            )
            for row in rows
        ]
    )
    columns = getattr(model, "_offball_feature_columns", None)
    if columns is not None:
        features = features[:, columns]
    predictions = model.predict_proba(features)[:, 1]
    for row, value in zip(rows, predictions):
        row["x360"] = float(value)
        del row["opponents"]
    return rows


def _pass_features(row, direction):
    from offball_value.xpass import pass_features

    return pass_features(
        to_statsbomb_coordinates(row["start"], direction),
        to_statsbomb_coordinates(row["end"], direction),
        "Ground Pass",
        False,
    )


def report(rows):
    """Spread of P across the responses the defender could pick."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["scene"], row["defender"], row["option"]), []).append(row)

    summary = {"hybrid": [], "x360": []}
    flat = {"hybrid": [], "x360": []}
    for cells in groups.values():
        if len(cells) < 4:
            continue
        for name in ("hybrid", "x360"):
            values = np.array([cell[name] for cell in cells])
            summary[name].append(values.max() - values.min())
            flat[name].append(float((values.max() - values.min()) < 0.01))

    print(f"(장면, 수비수, 옵션) 조합 {len(summary['hybrid'])}개 · 셀 {len(rows):,}개\n")
    print("수비수가 대응을 바꿀 때 P가 움직이는 폭 (같은 옵션·같은 릴리스)\n")
    print(f"{'':<16}{'중앙값':>10}{'평균':>10}{'75%':>10}{'90%':>10}{'무반응(<1%p)':>14}")
    for name, label in (("hybrid", "하이브리드"), ("x360", "xpass360")):
        spread = np.array(summary[name])
        print(
            f"{label:<14}{np.median(spread):>10.3f}{spread.mean():>10.3f}"
            f"{np.percentile(spread, 75):>10.3f}{np.percentile(spread, 90):>10.3f}"
            f"{np.mean(flat[name]):>14.1%}"
        )

    print("\n두 모델이 매기는 P 수준 자체")
    for name, label in (("hybrid", "하이브리드"), ("x360", "xpass360")):
        values = np.array([row[name] for row in rows])
        print(
            f"  {label:<12} 중앙 {np.median(values):.3f} · "
            f"평균 {values.mean():.3f} · 0.9 초과 {np.mean(values > 0.9):.1%}"
        )
    pairs = np.corrcoef(
        [row["hybrid"] for row in rows], [row["x360"] for row in rows]
    )[0, 1]
    print(f"  상관 {pairs:.3f}")


def main() -> None:
    args = parse_args()
    model = load_xpass_model(args.model)
    rows = []
    for path in args.audits:
        scenes = json.loads(path.read_text(encoding="utf-8"))
        for scene in scenes:
            rows.extend(collect(scene, model))
        print(f"[{path.parent.name}] 누적 {len(rows):,} 셀", flush=True)
    report(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
