"""What the run itself added, by rebuilding the local game without it.

Every measure the dilemma work has produced so far - the knee, the floors,
the gap, the balance of the two commit costs - is computed from the scene as
it actually unfolded. Two scenes that end up at the same numbers are
indistinguishable to all of them, even when one was already like that before
the runner moved and the other was made that way by the run.

The reviewer's point: a dilemma is what the RUN created, so the baseline has
to be the same scene without the run. This rebuilds it. The runner's history
is untouched, so his velocity at onset is estimated exactly as before; from
onset forward he is replaced by constant-velocity extrapolation of that
pre-decision velocity - he keeps doing what he was doing instead of making
the run. Every other player, the ball, and the defender response search are
unchanged, so the difference is attributable to the run alone.

Two forms are reported because they fail in opposite ways and the reviewer
asked for both: the difference says how much the run added, and breaks down
when scenes differ in scale; the ratio says how many times over it added,
and breaks down when the baseline is near zero.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.bundesliga import FIELD_LENGTH, FIELD_WIDTH  # noqa: E402
from offball_value.local_game_payoff import (  # noqa: E402
    LocalGamePayoffConfig,
    build_local_game_payoff_audit,
)
from offball_value.local_game_structure import LocalGameStructureConfig  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "measure_commit_tradeoff", ROOT / "scripts" / "measure_commit_tradeoff.py"
)
_commit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_commit)

SCENES = ROOT / "data/processed/settled_possession_run_onset_v0_1/shot_context_onset_audit.json"
LABELS = ROOT / "examples/research_audit/human_reviews/dilemma/dilemma_relabel_2026_09_12.csv"
MANIFEST = ROOT / "data/processed/round2_payoff_geometric_chunk0/manifest.json"


def configs_from_manifest(path: Path):
    """Rebuild the exact configuration the audits under test were built with."""
    manifest = json.loads(path.read_text())["config"]
    payoff_fields = {f.name for f in dataclasses.fields(LocalGamePayoffConfig)}
    structure_fields = {f.name for f in dataclasses.fields(LocalGameStructureConfig)}

    def coerce(value):
        return tuple(value) if isinstance(value, list) else value

    payoff = LocalGamePayoffConfig(
        **{k: coerce(v) for k, v in manifest["payoff"].items() if k in payoff_fields}
    )
    structure = LocalGameStructureConfig(
        **{
            k: coerce(v)
            for k, v in manifest["structure"].items()
            if k in structure_fields
        }
    )
    return payoff, structure


def without_the_run(scene, velocity_history_seconds: float):
    """The same scene with the runner never accelerating into his run.

    Only frames at or after onset are rewritten, and the velocity driving the
    extrapolation is measured on frames strictly before it, so nothing about
    the run leaks into its own counterfactual.
    """
    runner_id = str(scene["runner_id"])
    frames = scene["frames"]

    def runner_at(frame):
        for row in frame["players"]:
            if str(row[0]) == runner_id:
                return row
        return None

    onset = next(
        (f for f in frames if float(f["relative_time_s"]) >= -1e-9), None
    )
    if onset is None or runner_at(onset) is None:
        return None
    reference = None
    for frame in frames:
        t = float(frame["relative_time_s"])
        if t < -velocity_history_seconds - 1e-9 or t >= -1e-9:
            continue
        if runner_at(frame) is not None:
            reference = frame
            break
    if reference is None:
        return None

    onset_row, reference_row = runner_at(onset), runner_at(reference)
    span = float(onset["relative_time_s"]) - float(reference["relative_time_s"])
    if span <= 1e-9:
        return None
    vx = (float(onset_row[2]) - float(reference_row[2])) / span
    vy = (float(onset_row[3]) - float(reference_row[3])) / span
    x0, y0 = float(onset_row[2]), float(onset_row[3])
    t0 = float(onset["relative_time_s"])

    rebuilt = []
    for frame in frames:
        t = float(frame["relative_time_s"])
        if t <= t0 + 1e-9:
            rebuilt.append(frame)
            continue
        players = []
        for row in frame["players"]:
            if str(row[0]) != runner_id:
                players.append(row)
                continue
            elapsed = t - t0
            row = list(row)
            row[2] = min(FIELD_LENGTH / 2.0, max(-FIELD_LENGTH / 2.0, x0 + vx * elapsed))
            row[3] = min(FIELD_WIDTH / 2.0, max(-FIELD_WIDTH / 2.0, y0 + vy * elapsed))
            players.append(row)
        rebuilt.append({**frame, "players": players})
    return {**scene, "frames": rebuilt}


def measures(game):
    """Best defender row per scene, keyed by defender, for pairing later."""
    out = {}
    for defender in game["candidate_defenders"]:
        row = _commit.defender_row(game, defender)
        if row is not None:
            out[str(defender["defender_id"])] = row
    return out


def response_table(game):
    """Per defender: the R9 beneficiary and every option's Q on every response.

    The first sweep stored only the summary row, which recomputes the
    beneficiary inside each world. The runner standing still can hand R9 a
    different beneficiary, and then the actual/counterfactual pair is not the
    same triangle at all. Storing the raw table lets any later question - the
    beneficiary held fixed, a different beneficiary, a different knee - be
    answered without rebuilding the audit, which is the expensive part.
    """
    from offball_value.assignment_rule import attacking_team_id, onset_state
    from offball_value.coupled_beneficiary import rule_r9, value_at_times

    state = onset_state(game)
    runner = str(game["runner_id"])
    out = {}
    for defender in game["candidate_defenders"]:
        defender_id = str(defender["defender_id"])
        if defender_id not in state:
            continue
        direct = next(
            (
                row
                for row in defender["responses"]
                if str(row["response_id"]) == str(defender.get("direct_best_response_id"))
            ),
            None,
        )
        if direct is None:
            continue
        defender_at, attacker_at = _commit._coupled.absolute_lookups(game, defender)
        if defender_at is None:
            continue
        curves = {
            option_id: value_at_times(cell.get("candidate_grid") or [])
            for option_id, cell in direct["cells"].items()
            if cell.get("legal") is not False
            and cell.get("q") is not None
            and option_id != runner
        }
        if not curves:
            continue
        beneficiary, _ = rule_r9(
            state,
            attacking_team_id(game),
            runner,
            str(game["carrier_id"]),
            defender_id,
            defender_at,
            attacker_at,
            curves,
            float(game["horizon_seconds"]),
        )
        option_ids = sorted(
            {
                option_id
                for response in defender["responses"]
                for option_id in (response.get("cells") or {})
            }
        )
        table = {option_id: [] for option_id in option_ids}
        for response in defender["responses"]:
            cells = response.get("cells") or {}
            for option_id in option_ids:
                cell = cells.get(option_id) or {}
                value = cell.get("q")
                legal = cell.get("legal") is not False
                table[option_id].append(
                    None if (value is None or not legal) else round(float(value), 5)
                )
        out[defender_id] = {
            "beneficiary": beneficiary,
            "runner": runner,
            "q": table,
            "n_responses": len(defender["responses"]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scenes", type=Path, default=SCENES)
    parser.add_argument(
        "--reuse-actual",
        nargs="*",
        default=[],
        help=(
            "Audit directories holding the already-built actual games. Every "
            "other analysis reads these, so reusing them keeps this measurement "
            "on exactly the same footing and halves the work."
        ),
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    options = parser.parse_args()

    wanted = {
        (row["match_id"], row["onset_frame_id"])
        for row in csv.DictReader(LABELS.open(encoding="utf-8"))
    }
    scenes = [
        scene
        for scene in json.loads(options.scenes.read_text())
        if (str(scene["match_id"]), str(scene["onset_frame_id"])) in wanted
    ]
    scenes.sort(key=lambda s: (str(s["match_id"]), int(s["onset_frame_id"])))
    if options.limit:
        scenes = scenes[: options.limit]
    if options.shards > 1:
        scenes = scenes[options.shard :: options.shards]

    prebuilt = {}
    for directory in options.reuse_actual:
        for game in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            prebuilt[(str(game["match_id"]), str(game["onset_frame_id"]))] = game

    payoff_config, structure_config = configs_from_manifest(MANIFEST)
    print(
        f"장면 {len(scenes)}개 · 반사실 = 온셋 이전 {structure_config.velocity_history_seconds}s "
        f"속도로 등속 연장\n",
        flush=True,
    )

    results, started = [], time.time()
    for index, scene in enumerate(scenes, 1):
        key = (str(scene["match_id"]), str(scene["onset_frame_id"]))
        variant = without_the_run(scene, structure_config.velocity_history_seconds)
        if variant is None:
            print(f"[{index}/{len(scenes)}] {key} 건너뜀 (온셋 이전 이력 없음)", flush=True)
            continue
        try:
            built = prebuilt.get(key)
            if built is None:
                built = build_local_game_payoff_audit(
                    scene, payoff_config, structure_config
                )
            counterfactual = build_local_game_payoff_audit(
                variant, payoff_config, structure_config
            )
            actual = measures(built)
            without = measures(counterfactual)
            actual_table = response_table(built)
            without_table = response_table(counterfactual)
        except Exception as error:  # noqa: BLE001 - one bad scene must not stop the sweep
            print(f"[{index}/{len(scenes)}] {key} 실패: {error}", flush=True)
            continue
        for defender_id, row in actual.items():
            base = without.get(defender_id)
            if base is None:
                continue
            results.append(
                {
                    "match_id": key[0],
                    "onset_frame_id": key[1],
                    "defender_id": defender_id,
                    "runner_name": scene.get("runner_name"),
                    "actual": row,
                    "without_run": base,
                    "actual_table": actual_table.get(defender_id),
                    "without_run_table": without_table.get(defender_id),
                }
            )
        elapsed = time.time() - started
        print(
            f"[{index}/{len(scenes)}] {scene['runner_name']} · {key[1]} "
            f"· 수비수 {len(actual)}명 짝지음 {sum(1 for d in actual if d in without)} "
            f"· {elapsed / index:.0f}s/장면 · 남은 {(len(scenes) - index) * elapsed / index / 60:.0f}분",
            flush=True,
        )
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(
            json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    print(f"\n삼각 {len(results)}개 → {options.output}")


if __name__ == "__main__":
    main()
