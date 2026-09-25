#!/usr/bin/env python3
"""Draw the shortlisted (runner, defender, beneficiary) triples on a pitch.

The audit page per scene is a 2 MB interactive viewer: right for studying one
moment, wrong for asking "what kind of thing is the pipeline picking?" across
forty of them. This renders each triple as one small pitch diagram and puts
them all on a single self-contained page you can scroll.

The triple is the three coloured players:

  runner       the off-ball run the pipeline detected (game['runner_id'])
  defender     the candidate defender that run puts in a bind
  beneficiary  the other attacker who gains if the defender follows the runner
               -- the defender's derived_option_id, chosen by assignment_rule_v1

The ball carrier is often the beneficiary and not a fourth thing: R1's
carrier-override branch returns the carrier by design when he is advancing at
the defender -- 204 of the 1,903 triples, and every override among them. When
the carrier is NOT the beneficiary he is drawn as a hollow ring for context,
and deliberately gets no categorical hue: only three slots of the reference
palette clear the all-pairs colour-blindness gate, and the triple needs all
three.

Each card prints the branch that picked the beneficiary (override / nearest /
exclusion), because that is what tells you whether to argue with the pick.

Every diagram is rotated so the attack runs left to right, so forty cards can
be compared without re-orienting each time.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import pandas as pd

from build_review_shortlist import build_scene_index, pick

PITCH_X, PITCH_Y = 105.0, 68.0
SCALE = 6.0                      # px per metre
PAD = 10.0
W = PITCH_X * SCALE + 2 * PAD
H = PITCH_Y * SCALE + 2 * PAD

# Slot 3 of the reference palette is aqua, which the green pitch swallows, so
# the beneficiary takes slot 5 (magenta) instead. Blue / orange / magenta is
# the same triad Okabe-Ito settles on for three categories. The palette's
# documented all-pairs colour-blindness guarantee covers slots 1-3 against a
# near-white surface, so neither that guarantee nor its validator applies
# here; this substitution is reasoned, not measured.
ROLES = [
    ("runner", "러너", "--series-1"),
    ("defender", "수비수", "--series-2"),
    ("beneficiary", "수혜자", "--series-5"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ranking", type=Path,
                   default=Path("data/processed/triple_ranking_xt.csv"))
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v5_hybrid"))
    p.add_argument("--output", type=Path,
                   default=Path("out/triple_overview/index.html"))
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--dedupe-seconds", type=float, default=20.0)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--trail-stride", type=int, default=4,
                   help="궤적 샘플 간격 (프레임). 25fps에서 4 = 0.16초")
    p.add_argument("--arms", default="consensus,vacated_xt",
                   help=("정렬 기준을 쉼표로. 앞에 '-' 를 붙이면 작은 값이 "
                         "위로 옵니다 (예: -option_spread)."))
    p.add_argument("--extra", type=Path, default=None,
                   help=("(match_id, onset_frame_id) 로 조인할 추가 지표 CSV. "
                         "러너 목적지 xT 같은 장면 단위 값을 arms 에 쓸 때."))
    return p.parse_args()


def xy(x: float, y: float, flip: bool) -> tuple[float, float]:
    """Pitch metres -> card px, rotated 180 degrees when the attack runs -x.

    A 180 degree rotation rather than an x-mirror: mirroring would swap the
    left and right flanks, so a run down the right wing would read as a run
    down the left.
    """
    if flip:
        x, y = -x, -y
    return (PAD + (x + PITCH_X / 2) * SCALE,
            PAD + (PITCH_Y / 2 - y) * SCALE)


def pitch_markings(flip: bool) -> str:
    """Hairline pitch. Recessive: it orients the reader, it is not the data."""
    def rect(x0, y0, x1, y1):
        ax, ay = xy(x0, y0, flip)
        bx, by = xy(x1, y1, flip)
        return (f"<rect x='{min(ax,bx):.1f}' y='{min(ay,by):.1f}' "
                f"width='{abs(bx-ax):.1f}' height='{abs(by-ay):.1f}' "
                f"fill='none' stroke='var(--pitch-line)' stroke-width='1'/>")
    cx, cy = xy(0, 0, flip)
    top, _ = xy(0, PITCH_Y / 2, flip)
    parts = [
        rect(-52.5, -34, 52.5, 34),
        rect(-52.5, -20.16, -36, 20.16), rect(52.5, -20.16, 36, 20.16),
        rect(-52.5, -9.16, -47, 9.16), rect(52.5, -9.16, 47, 9.16),
        f"<line x1='{cx:.1f}' y1='{PAD:.1f}' x2='{cx:.1f}' "
        f"y2='{H - PAD:.1f}' stroke='var(--pitch-line)' stroke-width='1'/>",
        f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{9.15 * SCALE:.1f}' "
        f"fill='none' stroke='var(--pitch-line)' stroke-width='1'/>",
    ]
    return "".join(parts)


def trail(frames: list[dict], pid: str, flip: bool, stride: int) -> str:
    pts = []
    for i, f in enumerate(frames):
        if i % stride and i != len(frames) - 1:
            continue
        for p in f["players"]:
            if p[0] == pid:
                px, py = xy(p[2], p[3], flip)
                pts.append(f"{px:.1f},{py:.1f}")
                break
    if len(pts) < 2:
        return ""
    return (f"<polyline points='{' '.join(pts)}' fill='none' "
            f"stroke='currentColor' stroke-width='2' stroke-linecap='round' "
            f"stroke-linejoin='round' opacity='.5'/>")


R_PLAYER = 7.5          # every player the same size, so size means nothing
R_BALL = 4.0

# Team by fill, role by a glowing ring. Two channels that do not collide:
# a player belongs to one team and may or may not have a role, and the ring
# sits outside the disc so it never hides the fill underneath it.
ROLE_RING = {
    "runner": "var(--glow-runner)",
    "defender": "var(--glow-defender)",
    "beneficiary": "var(--glow-benef)",
}


def card_svg(game: dict, defender: dict, beneficiary_id: str | None,
             stride: int) -> tuple[str, dict]:
    """The pitch, plus the frames needed to play it back.

    A still frame shows where everyone stood; it does not show whether the
    run was a run. Judging that is the whole point of the review, so the card
    ships the whole clip -- 63 frames, onset minus two seconds to plus three
    -- and the page animates it.

    Coordinates are converted to card pixels HERE rather than in the browser,
    so the 180-degree rotation that puts every attack left-to-right is
    applied once, in the same code that drew the static frame. The playback
    cannot drift from the picture.

    Names are not drawn. Twenty-two labels over a pitch is unreadable, and
    the question being asked -- is that a run, is that the right defender --
    is answered by watching movement, not by reading. The page shows a name
    on hover instead.
    """
    frames = game["background_frames"]
    onset = game["onset_frame_id"]
    flip = int(game.get("attacking_direction", 1)) < 0
    at_onset = min(frames, key=lambda f: abs(f["frame_id"] - onset))

    roles = {
        str(game["runner_id"]): "runner",
        str(defender["defender_id"]): "defender",
    }
    if beneficiary_id and str(beneficiary_id) not in roles:
        roles[str(beneficiary_id)] = "beneficiary"
    carrier = str(game["carrier_id"])
    attacking_team = str(game["attacking_team_id"])

    out = [f"<svg viewBox='0 0 {W:.0f} {H:.0f}' class=pitch "
           f"role=img aria-label='피치 다이어그램'>",
           f"<rect width='{W:.0f}' height='{H:.0f}' rx='8' "
           f"fill='var(--pitch-fill)'/>",
           pitch_markings(flip)]

    # Trails sit under everything and stay faint: they are context for the
    # moving dot, not a thing to read on their own.
    out.append("<g class=trails>")
    for pid, role in roles.items():
        line = trail(frames, pid, flip, stride)
        if line:
            out.append(f"<g style='color:{ROLE_RING[role]}'>{line}</g>")
    out.append("</g>")

    order = [str(p[0]) for p in at_onset["players"]]
    for slot, p in enumerate(at_onset["players"]):
        pid, team, x, y, name = str(p[0]), str(p[1]), p[2], p[3], p[4]
        px, py = xy(x, y, flip)
        attacking = team == attacking_team
        fill = "var(--team-att)" if attacking else "var(--team-def)"
        role = roles.get(pid)
        side = "공격" if attacking else "수비"
        title = f"{name} · {side}" + (f" · {role_kr(role)}" if role else "")
        glow = ""
        if role:
            glow = (f"<circle class='gl' data-i='{slot}' cx='{px:.1f}' "
                    f"cy='{py:.1f}' r='{R_PLAYER + 2.5:.1f}' fill='none' "
                    f"stroke='{ROLE_RING[role]}' stroke-width='3' "
                    f"style='filter:drop-shadow(0 0 5px {ROLE_RING[role]})'/>")
        dot = (f"<circle class='pl' data-i='{slot}' data-n=\'{escape(str(title))}\' "
               f"cx='{px:.1f}' cy='{py:.1f}' r='{R_PLAYER}' fill='{fill}' "
               f"stroke='var(--dot-line)' stroke-width='1'/>")
        # The carrier keeps a small white pip so the ball's owner is readable
        # even in the frames where the ball is behind a body.
        pip = (f"<circle class='pip' data-i='{slot}' cx='{px:.1f}' "
               f"cy='{py:.1f}' r='2' fill='#fff' opacity='.9' "
               f"pointer-events='none'/>") if pid == carrier else ""
        out.append(glow + dot + pip)

    bx, by = xy(at_onset["ball"][0], at_onset["ball"][1], flip)
    out.append(f"<circle class=ball cx='{bx:.1f}' cy='{by:.1f}' r='{R_BALL}' "
               f"fill='#fff' stroke='#111' stroke-width='1.4' "
               f"pointer-events='none'/>")
    out.append("</svg>")

    # Playback payload: player pixel positions per frame, in `order`, plus
    # the ball and the clock. Rounded to 0.1 px -- finer than the eye and
    # than the pixels-per-metre scale, and it keeps the page a few hundred KB.
    positions, ball, clock = [], [], []
    for f in frames:
        by_id = {str(p[0]): (p[2], p[3]) for p in f["players"]}
        row = []
        for pid in order:
            if pid in by_id:
                fx, fy = xy(by_id[pid][0], by_id[pid][1], flip)
                row.append([round(fx, 1), round(fy, 1)])
            else:
                row.append(None)
        positions.append(row)
        fbx, fby = xy(f["ball"][0], f["ball"][1], flip)
        ball.append([round(fbx, 1), round(fby, 1)])
        clock.append(round(float(f["relative_time_s"]), 2))
    onset_index = min(range(len(frames)),
                      key=lambda i: abs(frames[i]["frame_id"] - onset))
    return "".join(out), {
        "p": positions, "b": ball, "t": clock, "onset": onset_index,
    }


def role_kr(role: str) -> str:
    return {"runner": "러너", "defender": "수비수",
            "beneficiary": "수혜자"}.get(role, role)


def player_bar() -> str:
    """Transport for one clip: play, scrub, and the clock.

    The slider is the part that matters. Deciding whether a run is a run
    means going back over the same second a few times, and a play button
    alone makes that a matter of timing the click.
    """
    return (
        "<div class=play>"
        "<button type=button class=pp aria-label='재생'>▶</button>"
        "<input type=range class=scrub min=0 max=0 value=0 step=1 "
        "aria-label='시간'>"
        "<span class=clock>0.00s</span>"
        "</div>"
    )


VERDICTS = (("ok", "잘 잡음"), ("bad", "틀림"), ("meh", "애매"))
PARTS = (("runner", "러너"), ("defender", "수비수"), ("beneficiary", "수혜자"))


def review_block(card_id: str, runner: str, defender: str, beneficiary: str) -> str:
    """Three verdicts and a note, per card.

    The point of looking at forty of these is to decide whether stage 1 found
    the right run and stage 2 the right pair, and that judgement is worth
    nothing if it stays in the reviewer's head. It is also the label the
    ranking criteria cannot currently be checked against: there are nine
    triples with a named runner to test, and forty verdicts here would more
    than quadruple that.

    Stored in localStorage, which is per-browser and never leaves the
    machine, so the export button is the only way the verdicts become a
    file. Every read and write is wrapped, because storage throws in a
    private window rather than returning empty.
    """
    who = {"runner": runner, "defender": defender, "beneficiary": beneficiary}
    rows = []
    for part, label in PARTS:
        buttons = "".join(
            f"<button type=button class=v data-k=\'{escape(card_id)}\' "
            f"data-p=\'{part}\' data-v=\'{value}\'>{escape(text)}</button>"
            for value, text in VERDICTS
        )
        rows.append(
            f"<div class=vrow><span class=vlabel>{label}"
            f"<i>{escape(who[part])}</i></span>"
            f"<span class=vbtns>{buttons}</span></div>"
        )
    return (
        f"<div class=review data-card=\'{escape(card_id)}\'>"
        + "".join(rows)
        + f"<input class=note data-k=\'{escape(card_id)}\' "
          f"placeholder=\'메모 (선택)\'>"
        + "</div>"
    )


ARM_NOTE = {
    "consensus": "minimax_worst_q · second_best_q 백분위 평균 — 수비수가 최선으로 "
                 "대응해도 위험하고, 공격의 2순위 옵션도 아픈 순간",
    "vacated_xt": "수비수가 비워둔 공간의 PAUSA 합 — 수비가 무엇을 내주는가",
    "-option_spread": "1순위와 2순위 옵션의 값 차이가 가장 작은 순 — 우리가 "
                      "가진 유일한 '선택의 어려움' 지표. 작을수록 수비수가 "
                      "하나를 고르기 어렵다는 뜻입니다.",
    "-xt_max": "러너가 지나는 경로의 xT 최댓값이 가장 낮은 순 — 오늘 측정에서 "
               "전문가가 strong 으로 찍은 장면이 이쪽에 몰렸습니다(AUC 0.213). "
               "이론이 아니라 관측을 따라간 기준입니다.",
}

CSS = """
:root{color-scheme:light;
 --page:#f9f9f7; --surface:#fcfcfb; --pitch-fill:#fcfcfb; --pitch-line:#e1e0d9;
 --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
 --border:rgba(11,11,11,.10);
 --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
 --series-5:#e87ba4;
 /* A dull sage rather than a broadcast green: the pitch is the backdrop the
    three role colours have to stay legible against, not a feature. */
 --pitch-fill:#7f9a7c; --pitch-line:rgba(255,255,255,.45);
 --team-att:#e8722c; --team-def:#2f6ccc; --dot-line:rgba(0,0,0,.35);
 --glow-runner:#ff2f4d; --glow-defender:#19e0ff; --glow-benef:#ffe81f;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){
 color-scheme:dark;
 --page:#0d0d0d; --surface:#1a1a19; --pitch-fill:#1a1a19; --pitch-line:#2c2c2a;
 --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
 --border:rgba(255,255,255,.10);
 --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
 --series-5:#d55181;
 --pitch-fill:#33452f; --pitch-line:rgba(255,255,255,.26);
 --team-att:#d9661f; --team-def:#2a63c0; --dot-line:rgba(0,0,0,.45);
 --glow-runner:#ff4a63; --glow-defender:#33e6ff; --glow-benef:#ffe94d;}}
:root[data-theme=dark]{color-scheme:dark;
 --page:#0d0d0d; --surface:#1a1a19; --pitch-fill:#1a1a19; --pitch-line:#2c2c2a;
 --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
 --border:rgba(255,255,255,.10);
 --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
 --series-5:#d55181;
 --pitch-fill:#33452f; --pitch-line:rgba(255,255,255,.26);
 --team-att:#d9661f; --team-def:#2a63c0; --dot-line:rgba(0,0,0,.45);
 --glow-runner:#ff4a63; --glow-defender:#33e6ff; --glow-benef:#ffe94d;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);
 font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:28px 16px 72px}
h1{font-size:25px;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--text-secondary);margin:0 0 20px;max-width:64ch}
h2{font-size:18px;margin:40px 0 2px}
.note{color:var(--text-secondary);font-size:13px;margin:0 0 18px;max-width:72ch}
.legend{display:flex;flex-wrap:wrap;gap:16px;align-items:center;
 padding:12px 14px;background:var(--surface);border:1px solid var(--border);
 border-radius:8px;margin:0 0 8px;font-size:13.5px}
.key{display:inline-flex;align-items:center;gap:7px}
.dot{width:11px;height:11px;border-radius:50%;flex:none}
.ring{width:11px;height:11px;border-radius:50%;flex:none;
 border:1.6px solid var(--text-secondary)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));
 gap:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
 padding:12px 12px 10px;min-width:0}
.pitch{display:block;width:100%;height:auto}
.trails{opacity:.30}
.trails polyline{stroke-width:1.6}
.stage .card.hidetrails .trails{display:none}
circle.pl{cursor:default}
circle.pl:hover{stroke:#fff;stroke-width:2}
.tip{position:fixed;z-index:50;pointer-events:none;opacity:0;
 transition:opacity .12s ease;padding:4px 9px;border-radius:6px;
 background:rgba(17,17,17,.92);color:#fff;font-size:12.5px;
 white-space:nowrap;transform:translate(-50%,-140%)}
.tip.on{opacity:1}
.hd{display:flex;justify-content:space-between;gap:10px;align-items:baseline;
 margin:0 0 6px}
.rank{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.match{font-size:13px;color:var(--text-secondary);min-width:0;overflow:hidden;
 text-overflow:ellipsis;white-space:nowrap}
.mets{display:flex;gap:14px;margin:8px 2px 0;font-size:12px;
 color:var(--text-secondary);font-variant-numeric:tabular-nums;flex-wrap:wrap}
.mets b{font-weight:600;color:var(--text-primary)}
table{border-collapse:collapse;width:100%;background:var(--surface);
 border:1px solid var(--border);border-radius:8px;overflow:hidden;
 margin-top:10px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--border);
 font-size:13px}
th{background:var(--page);font-weight:600;color:var(--text-secondary);
 font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:0}
td.num{text-align:right;font-variant-numeric:tabular-nums;
 color:var(--text-secondary)}
details{margin-top:14px}
summary{cursor:pointer;color:var(--text-secondary);font-size:13.5px}
.review{margin:10px 2px 0;padding-top:9px;border-top:1px solid var(--border)}
.vrow{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.vlabel{flex:0 0 132px;font-size:12.5px;color:var(--text-secondary)}
.vlabel i{display:block;font-style:normal;font-size:11.5px;color:var(--muted);
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vbtns{display:flex;gap:5px}
button.v{font:inherit;font-size:12px;padding:3px 10px;min-height:26px;
 border:1px solid var(--border);border-radius:6px;background:var(--page);
 color:var(--text-secondary);cursor:pointer}
button.v:hover{border-color:var(--muted)}
button.v[aria-pressed=true]{color:#fff;border-color:transparent}
button.v[data-v=ok][aria-pressed=true]{background:#0ca30c}
button.v[data-v=bad][aria-pressed=true]{background:#d03b3b}
button.v[data-v=meh][aria-pressed=true]{background:#898781}
.note{width:100%;margin-top:6px;padding:5px 8px;font:inherit;font-size:12.5px;
 border:1px solid var(--border);border-radius:6px;background:var(--page);
 color:var(--text-primary)}
.bar{position:sticky;top:0;z-index:5;display:flex;gap:12px;align-items:center;
 flex-wrap:wrap;padding:10px 14px;margin:0 0 16px;background:var(--surface);
 border:1px solid var(--border);border-radius:8px}
.bar button{font:inherit;font-size:13px;padding:6px 14px;min-height:32px;
 border:1px solid var(--border);border-radius:6px;background:var(--page);
 color:var(--text-primary);cursor:pointer}
.bar button:hover{border-color:var(--muted)}
.prog{font-size:13px;color:var(--text-secondary);
 font-variant-numeric:tabular-nums}
.play{display:flex;align-items:center;gap:9px;margin:7px 2px 0}
.play button.pp{flex:none;width:32px;height:32px;padding:0;font-size:13px;
 border:1px solid var(--border);border-radius:6px;background:var(--page);
 color:var(--text-primary);cursor:pointer;line-height:1}
.play button.pp:hover{border-color:var(--muted)}
.scrub{flex:1;min-width:0;accent-color:var(--series-1);height:24px}
.clock{flex:none;width:52px;text-align:right;font-size:12px;
 color:var(--text-secondary);font-variant-numeric:tabular-nums}
.tdot{width:13px;height:13px;border-radius:50%;flex:none;
 border:1px solid rgba(0,0,0,.35)}
.gdot{width:13px;height:13px;border-radius:50%;flex:none;
 background:var(--muted);border:2.5px solid currentColor}
.nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap;
 padding:10px 14px;margin:0 0 16px;background:var(--surface);
 border:1px solid var(--border);border-radius:8px}
.nav button{font:inherit;font-size:13.5px;padding:7px 16px;min-height:36px;
 border:1px solid var(--border);border-radius:6px;background:var(--page);
 color:var(--text-primary);cursor:pointer}
.nav button:hover:not(:disabled){border-color:var(--muted)}
.nav button:disabled{opacity:.4;cursor:default}
.count{font-size:14px;font-weight:600;min-width:74px;text-align:center;
 font-variant-numeric:tabular-nums}
.spacer{flex:1}
.speed{font-size:13px;color:var(--text-secondary)}
.speed select{font:inherit;font-size:13px;padding:5px 8px;min-height:32px;
 border:1px solid var(--border);border-radius:6px;background:var(--page);
 color:var(--text-primary)}
/* One scene at a time: forty pitches at once is a contact sheet, and the
   judgement being asked for needs the clip watched, not skimmed. */
.stage{max-width:820px;margin:0 auto}
.stage .card{display:none}
.stage .card.on{display:block}
"""

NAV_BAR = (
    "<div class=nav>"
    "<button type=button class=prev>← 이전</button>"
    "<span class=count>1 / 1</span>"
    "<button type=button class=next>다음 →</button>"
    "<span class=spacer></span>"
    "<label class=speed>속도 "
    "<select class=rate>"
    "<option value='0.25'>0.25배</option>"
    "<option value='0.5' selected>0.5배</option>"
    "<option value='1'>1배 (실제 속도)</option>"
    "</select></label>"
    "<span style='font-size:12.5px;color:var(--muted)'>← → 키로도 넘어갑니다</span>"
    "</div>"
)

REVIEW_BAR = (
    "<div class=bar>"
    "<span class=prog>0 / 0 판정함</span>"
    "<button type=button class=export>CSV 내려받기</button>"
    "<button type=button class=clear>판정 지우기</button>"
    "<span style='font-size:12.5px;color:var(--muted)'>"
    "판정은 이 브라우저에만 저장됩니다 — 끝나면 CSV로 내려받으세요."
    "</span></div>"
)

PLAY_JS = """
<script>
(function () {
  var CLIPS = __CLIPS__;
  var FPS = 25;
  var rate = 0.5;

  function setup(card) {
    var clip = CLIPS[card.dataset.clip];
    if (!clip) return;
    var svg = card.querySelector('svg.pitch');
    var dots = {};
    svg.querySelectorAll('circle.pl, circle.gl, circle.pip').forEach(
      function (c) { (dots[c.dataset.i] = dots[c.dataset.i] || []).push(c); });
    var ball = svg.querySelector('circle.ball');
    var scrub = card.querySelector('.scrub');
    var clock = card.querySelector('.clock');
    var pp = card.querySelector('.pp');
    var last = clip.t.length - 1;
    scrub.max = last;
    scrub.value = clip.onset;

    function draw(i) {
      var row = clip.p[i];
      for (var k in dots) {
        var xy = row[k];
        if (!xy) continue;
        dots[k].forEach(function (c) {
          c.setAttribute('cx', xy[0]);
          c.setAttribute('cy', xy[1]);
        });
      }
      ball.setAttribute('cx', clip.b[i][0]);
      ball.setAttribute('cy', clip.b[i][1]);
      clock.textContent = clip.t[i].toFixed(2) + 's';
    }

    var timer = null;
    function stop() {
      if (timer) { clearInterval(timer); timer = null; }
      pp.textContent = '▶';
      pp.setAttribute('aria-label', '재생');
    }
    function play() {
      if (timer) { stop(); return; }
      /* restart from the top when parked at the end */
      if (+scrub.value >= last) { scrub.value = 0; draw(0); }
      pp.textContent = '❚❚';
      pp.setAttribute('aria-label', '일시정지');
      timer = setInterval(function () {
        var i = +scrub.value + 1;
        if (i > last) { stop(); return; }
        scrub.value = i;
        draw(i);
      }, 1000 / (FPS * rate));
    }
    stops.push(stop);
    pp.addEventListener('click', play);
    scrub.addEventListener('input', function () { stop(); draw(+scrub.value); });
    draw(clip.onset);
  }

  var stops = [];

  window.addEventListener('DOMContentLoaded', function () {
    var cards = document.querySelectorAll('.stage .card[data-clip]');
    cards.forEach(setup);

    var at = 0;
    var count = document.querySelector('.count');
    var prev = document.querySelector('.prev');
    var next = document.querySelector('.next');

    function show(i) {
      if (i < 0 || i >= cards.length) return;
      stops.forEach(function (f) { f(); });   // never leave a clip running offscreen
      cards[at].classList.remove('on');
      at = i;
      cards[at].classList.add('on');
      count.textContent = (at + 1) + ' / ' + cards.length;
      prev.disabled = at === 0;
      next.disabled = at === cards.length - 1;
      window.scrollTo({top: 0, behavior: 'smooth'});
    }
    if (cards.length) show(0);
    prev.addEventListener('click', function () { show(at - 1); });
    next.addEventListener('click', function () { show(at + 1); });
    document.addEventListener('keydown', function (ev) {
      // not while typing a note
      if (ev.target && ev.target.matches && ev.target.matches('input, select')) return;
      if (ev.key === 'ArrowLeft') { ev.preventDefault(); show(at - 1); }
      if (ev.key === 'ArrowRight') { ev.preventDefault(); show(at + 1); }
    });

    /* Hover name. A <title> element works but the browser delays it by
       about a second, which is too slow when scrubbing frame by frame. */
    var tip = document.createElement('div');
    tip.className = 'tip';
    document.body.appendChild(tip);
    document.addEventListener('mouseover', function (ev) {
      var c = ev.target;
      if (!c.classList || !c.classList.contains('pl')) return;
      tip.textContent = c.dataset.n || '';
      var r = c.getBoundingClientRect();
      tip.style.left = (r.left + r.width / 2) + 'px';
      tip.style.top = r.top + 'px';
      tip.classList.add('on');
    });
    document.addEventListener('mouseout', function (ev) {
      if (ev.target.classList && ev.target.classList.contains('pl')) {
        tip.classList.remove('on');
      }
    });

    var tog = document.querySelector('.trailtog');
    if (tog) tog.addEventListener('change', function () {
      cards.forEach(function (c) { c.classList.toggle('hidetrails', !tog.checked); });
    });

    var sel = document.querySelector('.rate');
    if (sel) {
      rate = parseFloat(sel.value) || 0.5;
      sel.addEventListener('change', function () {
        rate = parseFloat(sel.value) || 0.5;
        stops.forEach(function (f) { f(); });
      });
    }
  });
})();
</script>
"""

REVIEW_JS = """
<script>
(function () {
  var KEY = 'offball-review-v1';
  var data = {};
  function load() {
    try { data = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; }
    catch (e) { data = {}; }
  }
  function save() {
    try { localStorage.setItem(KEY, JSON.stringify(data)); } catch (e) {}
  }
  function entry(k) { if (!data[k]) data[k] = {}; return data[k]; }
  function cards() { return document.querySelectorAll('.review'); }
  function progress() {
    var done = 0, all = cards();
    all.forEach(function (c) {
      var e = data[c.dataset.card];
      if (e && e.runner && e.defender && e.beneficiary) done++;
    });
    var el = document.querySelector('.prog');
    if (el) el.textContent = done + ' / ' + all.length + ' 판정함';
  }
  function paint() {
    document.querySelectorAll('button.v').forEach(function (b) {
      var e = data[b.dataset.k] || {};
      b.setAttribute('aria-pressed',
        e[b.dataset.p] === b.dataset.v ? 'true' : 'false');
    });
    document.querySelectorAll('input.note').forEach(function (i) {
      var e = data[i.dataset.k] || {};
      if (e.note !== undefined) i.value = e.note;
    });
    progress();
  }
  document.addEventListener('click', function (ev) {
    var b = ev.target.closest && ev.target.closest('button.v');
    if (!b) return;
    var e = entry(b.dataset.k);
    /* clicking the same verdict again clears it, so a misclick is undoable */
    e[b.dataset.p] = (e[b.dataset.p] === b.dataset.v) ? '' : b.dataset.v;
    save(); paint();
  });
  document.addEventListener('input', function (ev) {
    if (!ev.target.matches || !ev.target.matches('input.note')) return;
    entry(ev.target.dataset.k).note = ev.target.value;
    save(); progress();
  });
  function csv() {
    var lines = ['match_id,onset_frame_id,defender_id,runner,defender,beneficiary,note'];
    cards().forEach(function (c) {
      var k = c.dataset.card, e = data[k] || {};
      if (!e.runner && !e.defender && !e.beneficiary && !e.note) return;
      var p = k.split(':');
      var cell = function (v) {
        v = (v === undefined || v === null) ? '' : String(v);
        return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
      };
      lines.push([p[0], p[1], p[2], e.runner, e.defender, e.beneficiary, e.note]
                 .map(cell).join(','));
    });
    return lines.join('\n') + '\n';
  }
  window.addEventListener('DOMContentLoaded', function () {
    load(); paint();
    var ex = document.querySelector('.export');
    if (ex) ex.addEventListener('click', function () {
      var blob = new Blob([csv()], {type: 'text/csv;charset=utf-8'});
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'triple_review.csv';
      a.click();
      URL.revokeObjectURL(a.href);
    });
    var cl = document.querySelector('.clear');
    if (cl) cl.addEventListener('click', function () {
      if (!confirm('이 브라우저에 저장된 판정을 모두 지웁니다. 계속할까요?')) return;
      data = {}; save(); paint();
      document.querySelectorAll('input.note').forEach(function (i) { i.value = ''; });
    });
  });
})();
</script>
"""


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.ranking)
    frame["consensus"] = (
        frame["minimax_worst_q"].rank(pct=True)
        + frame["second_best_q"].rank(pct=True)
    ) / 2
    if args.extra is not None:
        extra = pd.read_csv(args.extra)
        frame = frame.merge(extra, on=["match_id", "onset_frame_id"], how="left")

    arms = [a.strip() for a in str(args.arms).split(",") if a.strip()]
    for arm in arms:
        # A leading '-' means small is good; pick() always takes the largest,
        # so the column is negated rather than the sort being special-cased.
        if arm.startswith("-"):
            base = arm[1:]
            if base not in frame:
                raise SystemExit(f"'{base}' 열이 랭킹 표에 없습니다")
            frame[arm] = -frame[base]
        elif arm not in frame:
            raise SystemExit(f"'{arm}' 열이 랭킹 표에 없습니다")

    index = build_scene_index(args.build)

    cache: dict[Path, dict] = {}
    body: list[str] = []
    tables: list[str] = []
    clips: dict[str, dict] = {}

    all_cards: list[str] = []
    for arm in arms:
        sel = pick(frame.dropna(subset=[arm]), arm, args.top_n,
                   args.dedupe_seconds, args.fps)
        cards, rows = [], []
        for n, r in enumerate(sel.itertuples(), 1):
            scene_dir = index.get((r.match_id, int(r.onset_frame_id)))
            if scene_dir is None:
                continue
            if scene_dir not in cache:
                payload = json.loads(
                    (scene_dir / "local_game_payoff_audits.json").read_text())
                cache[scene_dir] = payload[0] if payload else {}
            game = cache[scene_dir]
            if not game:
                continue
            defender = next(
                (d for d in game.get("candidate_defenders") or []
                 if str(d["defender_id"]) == str(r.defender_id)), None)
            if defender is None:
                continue
            names = {p[0]: p[4] for p in game["background_frames"][0]["players"]}
            ben_id = defender.get("derived_option_id")
            ben = names.get(ben_id, "—")
            match = str(game.get("match_label", r.match_id)).split(" · ")[0]
            card_id = f"{r.match_id}:{int(r.onset_frame_id)}:{r.defender_id}"
            svg, clip = card_svg(game, defender, ben_id, args.trail_stride)
            clips[card_id] = clip
            cards.append(
                f"<div class=card data-clip='{escape(card_id)}' "
                f"data-arm='{arm}'>"
                f"<div class=hd><span class=rank>{arm} #{n}</span>"
                f"<span class=match title='{escape(match)}'>{escape(match)}"
                f" · f{int(r.onset_frame_id)}</span></div>"
                f"{svg}"
                f"{player_bar()}"
                f"<div class=mets>"
                f"<span>minimax <b>{r.minimax_worst_q:.3f}</b></span>"
                f"<span>2nd <b>{r.second_best_q:.3f}</b></span>"
                f"<span>spread <b>{r.option_spread:.3f}</b></span>"
                f"<span>vacated_xt <b>{r.vacated_xt:.2f}</b></span>"
                f"<span>거리 <b>{r.distance_to_runner_m:.1f} m</b></span>"
                f"<span>규칙 <b>"
                f"{escape(str(defender.get('derived_rule_branch')))}</b></span>"
                f"</div>"
                + review_block(
                    f"{r.match_id}:{int(r.onset_frame_id)}:{r.defender_id}",
                    str(game.get("runner_name")),
                    str(defender.get("defender_name")),
                    str(ben),
                )
                + "</div>"
            )
            rows.append(
                f"<tr><td class=num>{n}</td><td>{escape(match)}</td>"
                f"<td class=num>{int(r.onset_frame_id)}</td>"
                f"<td>{escape(str(game.get('runner_name')))}</td>"
                f"<td>{escape(str(defender.get('defender_name')))}</td>"
                f"<td>{escape(str(ben))}</td>"
                f"<td>{escape(str(game.get('carrier_name')))}</td>"
                f"<td>{escape(str(defender.get('derived_rule_branch')))}</td>"
                f"<td class=num>{r.minimax_worst_q:.3f}</td>"
                f"<td class=num>{r.second_best_q:.3f}</td>"
                f"<td class=num>{r.option_spread:.3f}</td>"
                f"<td class=num>{r.vacated_xt:.2f}</td></tr>"
            )
        all_cards.extend(cards)
        tables.append(
            f"<details><summary>{escape(arm)} — 표로 보기 ({len(rows)}행)"
            f"<br><span style='font-weight:400;font-size:12.5px'>"
            f"{escape(ARM_NOTE.get(arm, ''))}</span></summary>"
            f"<table><tr><th>#</th><th>경기</th><th>프레임</th><th>러너</th>"
            f"<th>수비수</th><th>수혜자</th><th>볼 소유자</th><th>규칙</th>"
            f"<th class=num>minimax</th><th class=num>2nd</th>"
            f"<th class=num>spread</th><th class=num>vacated_xt</th></tr>"
            f"{''.join(rows)}</table></details>"
        )

    legend = (
        "<div class=legend>"
        "<span class=key><span class=tdot "
        "style='background:var(--team-att)'></span>공격팀</span>"
        "<span class=key><span class=tdot "
        "style='background:var(--team-def)'></span>수비팀</span>"
        "<span class=key><span class=gdot "
        "style='color:var(--glow-runner)'></span>러너</span>"
        "<span class=key><span class=gdot "
        "style='color:var(--glow-defender)'></span>반응 수비수</span>"
        "<span class=key><span class=gdot "
        "style='color:var(--glow-benef)'></span>수혜자</span>"
        "<span class=key>가운데 흰 점 = 볼 소유자</span>"
        "<span class=key>점에 마우스를 올리면 이름</span>"
        "<label class=key><input type=checkbox class=trailtog checked> "
        "궤적 보기</label></div>"
    )

    html = (
        "<!doctype html><html lang=ko><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>삼중항 개요</title>"
        f"<style>{CSS}</style><body><div class=wrap>"
        "<h1>(러너, 수비수, 수혜자) 삼중항</h1>"
        "<p class=sub>분데스리가 7경기 · 기준별 상위 "
        f"{args.top_n}개. 모든 다이어그램은 공격이 왼쪽에서 오른쪽으로 "
        "흐르도록 회전했습니다. 각 장면에서 러너 · 수비수 · 수혜자가 "
        "제대로 잡혔는지 판정해 주세요.</p>"
        + REVIEW_BAR + legend + NAV_BAR
        + f"<div class=stage>{''.join(all_cards)}</div>"
        + "".join(tables)
        + "</div>"
        + PLAY_JS.replace("__CLIPS__", json.dumps(clips, separators=(",", ":")))
        + REVIEW_JS + "</body></html>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"{args.output}  ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
