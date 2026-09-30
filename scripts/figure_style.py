"""The print style shared by the abstract's Figure 1 (render_figure1_dilemma.py) and Figure 2
(render_figure2_abstract.py): one face, one palette, one marker per role, one arrow, one match line.

Set 2026-09-30 from the team's feedback (a Helvetica face; restrained colours after recent SSAC papers --
white ground, blue / red players, thin grey pitch lines; thin arrows with small heads; the match's date,
teams and clock in a corner). The earlier dark style is kept in scripts/archive_v1_dark/.

Font: Helvetica is not installed on the cluster these figures were made on (nor Arial or Liberation Sans). Nimbus Sans -- URW's
metric-compatible Helvetica clone, the face Ghostscript uses for Helvetica -- is loaded from its files
by path, Regular and Bold; `use_font` refuses to run if matplotlib would resolve either weight to another
file, and `check_text` refuses a figure with a glyph the face lacks (no silent fallback).

Colours (the team's recommended values, not an official SSAC palette): attack blue #0072B2, defence
vermillion #D55E00 (the Okabe-Ito pair, told apart under every common colour-vision deficiency),
charcoal #222222 for text and the ball, grey #666666 for secondary text and the real (tracked)
movement, #D0D0D0 for pitch lines. The runner and the ball carrier are both attackers, so both blue:
told apart by marker (runner diamond, ball carrier disc, defender square) and by direct labels, so no
role rests on colour alone. Players outside the game keep their team's colour as a light tint (45%
on white) -- smaller, no outline -- so the team split stays readable without competing with the three.
"""

from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import ArrowStyle, FancyArrowPatch, Rectangle, Circle, Arc
import matplotlib.patheffects as pe

# ------------------------------------------------------------------------------------------------ font
FONT_DIR = Path("/usr/share/fonts/urw-base35")
FONT_FILES = {"normal": FONT_DIR / "NimbusSans-Regular.otf", "bold": FONT_DIR / "NimbusSans-Bold.otf"}
FAMILY = "Nimbus Sans"


def use_font() -> dict:
    """Load Nimbus Sans Regular and Bold from their files and make them the only text face (maths too).
    Returns {weight: file} as matplotlib resolves it; raises if either weight would come from elsewhere."""
    for f in FONT_FILES.values():
        if not f.exists():
            raise SystemExit(f"font file missing: {f}")
        font_manager.fontManager.addfont(str(f))
    matplotlib.rcParams.update({
        "font.family": FAMILY, "font.sans-serif": [FAMILY], "font.weight": "normal",
        "mathtext.fontset": "custom", "mathtext.rm": FAMILY, "mathtext.it": f"{FAMILY}:italic",
        "mathtext.bf": f"{FAMILY}:bold", "mathtext.sf": FAMILY, "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    used = {}
    for weight, want in FONT_FILES.items():
        got = Path(font_manager.findfont(font_manager.FontProperties(family=FAMILY, weight=weight),
                                         fallback_to_default=False))
        if got.resolve() != want.resolve():
            raise SystemExit(f"{FAMILY} {weight} resolves to {got}, not {want}")
        used[weight] = str(got)
    return used


def check_text(fig) -> list[str]:
    """Every text artist of the figure: in Nimbus Sans Regular or Bold, and every character in the face.
    Returns the strings checked; raises on any other face or a missing glyph."""
    from fontTools.ttLib import TTFont
    cmaps = {str(f): TTFont(str(f)).getBestCmap() for f in FONT_FILES.values()}
    seen = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s:
            continue
        f = font_manager.findfont(t.get_fontproperties(), fallback_to_default=False)
        if f not in cmaps:
            raise SystemExit(f"text {s!r} set in {f}")
        missing = [c for c in s if not c.isspace() and ord(c) not in cmaps[f]]
        if missing:
            raise SystemExit(f"text {s!r}: {missing} not in {f}")
        seen.append(s)
    return seen


# pt, at the printed size
FS_TITLE, FS_PANEL, FS_LABEL, FS_NOTE, FS_SMALL = 10.0, 9.0, 9.0, 8.0, 7.5

# ---------------------------------------------------------------------------------------------- colour
ATTACK, DEFENCE = "#0072B2", "#D55E00"
INK, MUTED, FAINT, LINE, PAGE = "#222222", "#666666", "#B3B3B3", "#D0D0D0", "#FFFFFF"
PITCH = "#F7F7F7"                 # the pitch: a hair off white, so a panel reads as a panel with no frame


def tint(color: str, k: float) -> str:
    r, g, b = matplotlib.colors.to_rgb(color)
    return matplotlib.colors.to_hex((1 - k + k * r, 1 - k + k * g, 1 - k + k * b))


TEAM = {"att": ATTACK, "def": DEFENCE, "dfn": DEFENCE}
TEAM_TINT = {s: tint(c, 0.45) for s, c in TEAM.items()}
ROLE_COLOR = {"runner": ATTACK, "ball carrier": ATTACK, "beneficiary": ATTACK, "teammate": ATTACK, "defender": DEFENCE}
ROLE_NAME = {"runner": "Runner", "ball carrier": "Ball carrier", "beneficiary": "Ball carrier",
             "teammate": "Teammate", "defender": "Defender"}

# ---------------------------------------------------------------------------------------------- marks
# markers: size (pt) chosen so the three shapes cover about the same area
KEY_D = 8.5                                     # pt, the ball carrier's disc diameter
ROLE_MARKER = {"runner": ("D", 0.80 * KEY_D), "ball carrier": ("o", KEY_D), "beneficiary": ("o", KEY_D),
               "defender": ("s", 0.86 * KEY_D),
               # 3v1 (Figure 2 only): the second attacker in the game, when a scripted passer has the ball
               "teammate": ("^", 1.15 * KEY_D)}
KEY_EDGE = 0.8                                  # pt, charcoal outline of the three
OTHER_D, BALL_D = 6.0, 4.6                      # pt


def marker_radius(role) -> float:
    """pt from a key player's centre to his marker's farthest edge (a diamond's corner, a square's)."""
    m, ms = ROLE_MARKER[role]
    return ms / 2 * (math.sqrt(2) if m in "Ds^" else 1.0) + KEY_EDGE / 2


def key_radius() -> float:
    """The largest of the three: the clearance a path keeps from any key player."""
    return max(marker_radius(r) for r in ("runner", "ball carrier", "defender"))


def key_player(ax, xy, role, z=7):
    m, ms = ROLE_MARKER[role]
    ax.plot(*xy, m, ms=ms, color=ROLE_COLOR[role], mec=INK, mew=KEY_EDGE, zorder=z)


def key_hollow(ax, xy, role, color=MUTED, lw=0.8, z=6.5):
    """Where a key player really was: his marker, hollow."""
    m, ms = ROLE_MARKER[role]
    ax.plot(*xy, m, ms=ms, mfc="none", mec=color, mew=lw, zorder=z)


def other_player(ax, xy, side, z=3):
    ax.plot(*xy, "o", ms=OTHER_D, color=TEAM_TINT[side], mec=PAGE, mew=0.5, zorder=z)


def ball(ax, xy, z=8.5):
    ax.plot(*xy, "o", ms=BALL_D, color=INK, mec=PAGE, mew=0.6, zorder=z)


def halo(lw=2.2, color=PITCH):
    """A thin ground-coloured stroke round text, so a label stays readable over a pitch line."""
    return [pe.withStroke(linewidth=lw, foreground=color)]


# ---------------------------------------------------------------------------------------------- lines
MOVE_LW = 1.5                   # pt, a player's move still to come (Figure 1)
PAST_LW, PAST_ALPHA = 1.0, 0.45  # pt, a path already run: thinner and lighter
BALL_LW = 1.3                    # pt, the ball's moves (pass, shot)
BALL_DASH = (5.0, 2.8)           # pt on / off, the same at every width (divided by lw below, since
                                 # matplotlib scales a dash pattern by the line width)
REAL_LW = 0.8                    # pt, a real (tracked) move shown for comparison (Figure 2)
REAL_DOTS = (0.9, 1.7)
HEAD = ArrowStyle("-|>", head_length=0.5, head_width=0.2)


def head_scale(lw: float) -> float:
    """Arrowhead mutation scale (pt) for a line this wide: one small triangle, growing a little with lw."""
    return 7.0 + 1.6 * lw


def head_length(lw: float) -> float:
    return HEAD.head_length * head_scale(lw)                    # pt


def head_width(lw: float) -> float:
    return 2 * HEAD.head_width * head_scale(lw)                 # pt, full width


def metres_per_point(ax) -> float:
    (x0, x1) = ax.get_xlim()
    return (x1 - x0) / (ax.bbox.width / ax.figure.dpi * 72)


def _cut(pts, back):
    """(the path up to `back` metres before its end, that point) -- the head's base."""
    tip, left, i = pts[-1], back, len(pts) - 1
    while i > 0 and math.dist(pts[i], pts[i - 1]) < left:
        left -= math.dist(pts[i], pts[i - 1])
        i -= 1
    if i == 0:
        return [pts[0]], pts[0]
    (ax_, ay), (bx, by) = pts[i - 1], pts[i]
    k = left / max(math.dist(pts[i - 1], pts[i]), 1e-9)
    base = (bx + (ax_ - bx) * k, by + (ay - by) * k)
    return pts[:i] + [base], base


def arrow(ax, pts, color, lw, *, dashed=False, dots=False, alpha=1.0, z=6, mpp=None):
    """A path (metres, drawn through every point given) ending in one small filled '-|>' head. The head
    sits on the path's last head-length, pointing along that chord -- the last stretch of the move, not
    its last tracking step. The shaft stops a little inside the head, so the head's tip is never widened
    by the line's butt end. Returns the head's tip."""
    mpp = mpp or metres_per_point(ax)
    pts = [tuple(p) for p in pts]
    shaft, base = _cut(pts, head_length(lw) * mpp)
    tip = pts[-1]
    u = (tip[0] - base[0], tip[1] - base[1])
    n = math.hypot(*u) or 1.0
    tuck = (base[0] + u[0] / n * 0.35 * head_length(lw) * mpp, base[1] + u[1] / n * 0.35 * head_length(lw) * mpp)
    ls = (0, (BALL_DASH[0] / lw, BALL_DASH[1] / lw)) if dashed else (0, REAL_DOTS) if dots else "-"
    ax.add_line(Line2D(*zip(*(shaft + [tuck])), color=color, lw=lw, alpha=alpha, zorder=z, ls=ls,
                       solid_capstyle="butt", dash_capstyle="butt", solid_joinstyle="round"))
    ax.add_patch(FancyArrowPatch(base, tip, arrowstyle=HEAD, mutation_scale=head_scale(lw), lw=0.4,
                                 color=color, alpha=alpha, shrinkA=0, shrinkB=0, zorder=z + 0.05,
                                 joinstyle="miter"))
    return tip


def fig_arrow(fig, a, b, *, color=FAINT, lw=0.9):
    """An arrow in figure fractions (the tree's connectors, the attack direction): the same head."""
    fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle=HEAD,
                                   mutation_scale=head_scale(lw), lw=lw, color=color, shrinkA=0, shrinkB=0,
                                   capstyle="butt", joinstyle="miter"))


# ---------------------------------------------------------------------------------------------- pitch
def pitch(ax):
    """The pitch in metres from the centre spot (105 x 68), thin light-grey lines on a near-white ground."""
    ax.set_facecolor(PITCH)
    kw = dict(color=LINE, lw=0.6, zorder=0.5)
    ax.add_patch(Rectangle((-52.5, -34), 105, 68, fill=False, **kw))
    ax.plot([0, 0], [-34, 34], **kw)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, **kw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * 52.5 - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, fill=False, **kw))
        ax.add_patch(Rectangle((s * 52.5 - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, fill=False, **kw))
        ax.add_patch(Rectangle((s * 52.5 + (0 if s > 0 else -2), -3.66), 2, 7.32, fill=False, **kw))
        ax.plot([s * 41.5], [0], "o", ms=1.6, color=LINE, zorder=0.5)
        ax.add_patch(Arc((s * 41.5, 0), 18.3, 18.3, theta1=127 if s > 0 else -53, theta2=233 if s > 0 else 53, **kw))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---------------------------------------------------------------------------------------------- match
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/bundesliga-integrated"
SCENES = ROOT / "data/processed/showcase_v1/scenes.csv"
FPS = 25
# DFL position data number each half's frames from a fixed start: firstHalf from 10000, secondHalf from
# 100000 (checked 2026-09-30 on DFL-MAT-J03WOH: the first Frame N of every firstHalf / secondHalf FrameSet).
# At N = 10000 the ball is already 5.7 m from the centre spot at 16 m/s, so the kick-off was a fraction of a
# second earlier: the clock below (rounded to the second) may run up to ~1 s behind the stadium clock.
HALF_START = {1: 10000, 2: 100000}


def match_line(code: str, frame: int) -> str:
    """'YYYY-MM-DD · Home vs Guest · MM:SS' for a scene: the date (local, Europe/Berlin) and teams from the
    DFL match information, the clock = time since the half's first tracking frame at `frame` (first half:
    the match clock; second half: '2H' + time since the second half's first frame)."""
    row = next(r for r in csv.DictReader(SCENES.open(encoding="utf-8-sig")) if r["code"] == code)
    info = next(RAW.glob(f"DFL_02_01_matchinformation_*_{row['match_id']}.xml"))
    g = ET.parse(info).getroot().find("MatchInformation/General").attrib
    kickoff = datetime.fromisoformat(g["KickoffTime"]).astimezone(ZoneInfo("Europe/Berlin"))
    period = int(row["period"])
    s = round((frame - HALF_START[period]) / FPS)
    if s < 0:
        raise SystemExit(f"{code}: frame {frame} is before the half's first frame {HALF_START[period]}")
    clock = f"{s // 60:02d}:{s % 60:02d}" if period == 1 else f"2H {s // 60:02d}:{s % 60:02d}"
    return f"{kickoff:%Y-%m-%d} · {g['HomeTeamName']} vs {g['GuestTeamName']} · {clock}"


def match_note(fig, text, right_in, y_in):
    """The match line, once per figure, in the white margin under the panels (inches from bottom-left)."""
    W, H = fig.get_size_inches()
    fig.text(right_in / W, y_in / H, text, fontsize=FS_SMALL, color=MUTED, ha="right", va="baseline")
