"""Lock in every criterion variant's prediction BEFORE the new labels exist.

The seven original labels are burnt: five formalisations of the same
follow-gain idea score anywhere from 2/7 to 6/7 on them, so choosing between
the formalisations on that evidence would be pure overfitting
(docs/goal_danger_surface_v0_3_8_audit.md section 6).

Twenty-four fresh defender-games have been analysed but NOT yet labelled.
Writing down what every variant predicts on them now — before any human
judgement exists — turns the next review session into a genuine
out-of-sample test.  Whichever variant survives it was not chosen by
looking at the answers.

Output per defender-game: the pick under each (criterion, goal-danger
surface) pair, the margin over the runner-up, and whether the variants
disagree.  Games where they disagree are the discriminating ones and should
be reviewed first.

Usage:
    python scripts/preregister_derived_predictions.py \
        --audit-dir data/processed/candidate_scenes_geometric_g:geometric \
        --audit-dir data/processed/candidate_scenes_learned_g:learned \
        --out docs/preregistration_derived_criterion_v0_1.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_follow_gain", Path(__file__).with_name("evaluate_follow_gain_selection.py")
)
_follow_gain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_follow_gain)  # type: ignore[union-attr]
_gains = _follow_gain._gains

VARIANTS = [
    ("option", "type", False, "옵션별 follow-gain (차분 전에 max)"),
    ("action", "type", False, "행동별 · 채널 일치"),
    ("action", "full", False, "행동별 · 채널+lead+시각"),
    ("action", "type", True, "행동별 · 채널, 자리표시자 제외"),
    ("action", "full", True, "행동별 · 전부 고정, 자리표시자 제외"),
]

# The SHIPPED pipeline's own answer (`derived_option_id`, produced by
# local_game_selection_mode="pairwise_cross_cost") is none of the five above
# — it agrees with option/type/all on only 11/24 games under the geometric
# surface. It is also the number the audit UI displays, so leaving it off the
# table would let it be adopted after the fact. Register it.
INCUMBENT_KEY = "shipped/pairwise_cross_cost"


def _predict(audit_dir: Path) -> dict[str, dict]:
    games = json.loads((audit_dir / "local_game_payoff_audits.json").read_text())
    out: dict[str, dict] = {}
    for game in games:
        scene = str(game["onset_frame_id"])
        runner_id = str(game["runner_id"])
        for defender in game["candidate_defenders"]:
            key = f"{game['match_id']}:{scene}:{defender['defender_id']}"
            responses = [
                r for r in defender["responses"] if r.get("is_search_candidate", True)
            ] or defender["responses"]
            option_ids = [str(o["option_id"]) for o in defender["options"]]
            names = {
                str(o["option_id"]): str(o.get("option_name") or o["option_id"])
                for o in defender["options"]
            }
            if runner_id not in option_ids:
                continue
            # A tied "follow" response makes every prediction for this game
            # depend on JSON row order rather than on the criterion, so the
            # game carries no information and is excluded from scoring.
            runner_values = [
                0.0
                if (cell := r["cells"].get(runner_id)) is None
                or cell.get("legal") is False
                or cell.get("q") is None
                else float(cell["q"])
                for r in responses
            ]
            floor = min(runner_values)
            follow_ties = sum(1 for v in runner_values if v <= floor + 1e-12)

            record = {
                "scene": scene,
                "defender": str(defender.get("defender_name") or defender["defender_id"]),
                "runner": str(game.get("runner_name") or runner_id),
                "carrier": str(game.get("carrier_name") or game["carrier_id"]),
                "follow_response_ties": follow_ties,
                "scorable": follow_ties == 1,
                "predictions": {},
            }
            shipped = defender.get("derived_option_id")
            if shipped is not None:
                record["predictions"][INCUMBENT_KEY] = {
                    "pick": names.get(str(shipped), str(shipped)),
                    "gain": None,
                    "margin_over_runner_up": None,
                }
            for mode, action_key, executable, _label in VARIANTS:
                gains, _ = _gains(
                    responses, option_ids, runner_id, mode, action_key, executable
                )
                if not gains:
                    continue
                ranked = sorted(gains.items(), key=lambda kv: -kv[1])
                pick, best = ranked[0]
                second = ranked[1][1] if len(ranked) > 1 else 0.0
                record["predictions"][f"{mode}/{action_key}/{'exec' if executable else 'all'}"] = {
                    "pick": names.get(pick, pick),
                    "gain": round(float(best), 6),
                    "margin_over_runner_up": round(float(best - second), 6),
                }
            out[key] = record
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", action="append", required=True,
                        help="PATH:label")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    surfaces: dict[str, dict[str, dict]] = {}
    for spec in args.audit_dir:
        path_text, _, label = spec.partition(":")
        surfaces[label or path_text] = _predict(Path(path_text))

    keys = sorted(set().union(*(set(v) for v in surfaces.values())))
    payload = {
        "surfaces": surfaces,
        "variants": [INCUMBENT_KEY]
        + [f"{m}/{a}/{'exec' if e else 'all'}" for m, a, e, _ in VARIANTS],
        "adoption_rule": (
            "Adopt a criterion only if it is the unique top scorer, leads the "
            "runner-up by >=2 games, and at least 12 games were scored; "
            "otherwise adopt nothing and report the batch as inconclusive. "
            "Games with follow_response_ties > 1 are excluded from scoring "
            "(their pick depends on JSON row order, not on the criterion). "
            "Primary result: option-vs-action. Suggestive: type-vs-full. "
            "NOT TESTED by this batch and never to be reported as settled: "
            "terminal_structure inclusion (all-vs-exec) and the goal-danger "
            "surface, both of which have too few discordant games."
        ),
        "scoring": (
            "1.0 for matching `beneficiary`, 0.5 for matching "
            "`beneficiary_alt`, 0 otherwise. Rows with beneficiary=none or "
            "defender_reacts=no are reported separately as whole-family "
            "failures, not scored for any variant."
        ),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    json_path = args.out.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(blob, encoding="utf-8")

    lines = [
        "# 사전등록: 파생 선수 선정 기준 (v0.1)",
        "",
        f"내용 해시 `{digest}` · 예측 대상 {len(keys)}개 수비수 게임 · 라벨 0개",
        "",
        "7개 라벨은 소진되었다(같은 아이디어의 다섯 형식화가 2/7~6/7로 갈린다).",
        "따라서 **기준을 그 라벨로 고르는 것은 과적합**이다. 아래는 아직 아무도",
        "판정하지 않은 24개 수비수 게임에 대해, 다섯 기준 × 두 골위험 지도가",
        "무엇을 예측하는지를 **라벨이 생기기 전에** 고정한 기록이다.",
        "검수 후 이 표와 대조하면, 살아남은 기준은 답을 보고 고른 것이 아니게 된다.",
        "",
        "## 변형 목록",
        "",
    ]
    for mode, action_key, executable, label in VARIANTS:
        lines.append(f"- `{mode}/{action_key}/{'exec' if executable else 'all'}` — {label}")
    lines += ["", "## 기준들이 갈리는 게임 (여기부터 검수하면 정보량 최대)", ""]

    split, agree, unscorable = [], [], []
    for key in keys:
        row = {label: surf.get(key) for label, surf in surfaces.items()}
        picks = {
            f"{label}|{variant}": data["predictions"][variant]["pick"]
            for label, data in row.items()
            if data
            for variant in data["predictions"]
        }
        if any(data and not data["scorable"] for data in row.values()):
            unscorable.append((key, row, picks))
        elif len(set(picks.values())) > 1:
            split.append((key, row, picks))
        else:
            agree.append((key, row, picks))

    def render(bucket, header):
        lines.append(f"### {header} ({len(bucket)}개)")
        lines.append("")
        for key, row, picks in bucket:
            any_row = next(v for v in row.values() if v)
            lines.append(
                f"**{key}** · 러너 {any_row['runner']} · 캐리어 {any_row['carrier']} "
                f"· 수비수 {any_row['defender']}"
            )
            lines.append("")
            lines.append("| 지도 | 기준 | 예측 | gain | 2위와 차 |")
            lines.append("|---|---|---|---|---|")
            for label, data in row.items():
                if not data:
                    continue
                for variant, pred in data["predictions"].items():
                    gain = pred["gain"]
                    margin = pred["margin_over_runner_up"]
                    lines.append(
                        f"| {label} | `{variant}` | **{pred['pick']}** | "
                        f"{'—' if gain is None else f'{gain:.4f}'} | "
                        f"{'—' if margin is None else f'{margin:.4f}'} |"
                    )
            lines.append("")

    render(split, "예측이 갈리는 게임 — 채점 대상")
    render(agree, "모든 변형이 같은 답을 내는 게임 — 채점하되 판별력 없음")
    render(
        unscorable,
        "채점 제외 — follow 응답이 동점이라 예측이 JSON 행 순서에 좌우됨",
    )

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"locked {len(keys)} defender-games · hash {digest}")
    print(f"  discriminating (variants disagree): {len(split)}")
    print(f"  unanimous:                          {len(agree)}")
    print(f"  excluded (tied follow response):    {len(unscorable)}")
    print(f"wrote {args.out} and {json_path}")


if __name__ == "__main__":
    main()
