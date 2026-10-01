"""Add a caption strip under a finished figure PNG (2026-10-01).

The figure itself is not redrawn or rescaled: a white strip of the same width and dpi is rendered with the
figures' own font (figure_style: Nimbus Sans, the Helvetica clone) and stacked under it. The caption's
"Figure N." is bold, the rest regular, left-aligned at the figure's margin, wrapped to the width.

    python scripts/add_caption.py --input fig.png --output fig_caption.png --number "Figure 1." --text "..."
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import figure_style as st

FS_CAPTION = st.FS_NOTE          # 8 pt: under the panel titles (9 pt), readable at 180 mm in the abstract


def main() -> None:
    global FS_CAPTION
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--number", required=True, help="the bold lead, e.g. 'Figure 1.'")
    p.add_argument("--text", required=True)
    p.add_argument("--size", type=float, default=FS_CAPTION, help="caption, pt")
    p.add_argument("--one-line", action="store_true", help="refuse to wrap")
    p.add_argument("--dpi", type=float, default=600)
    p.add_argument("--margin-in", type=float, default=0.05, help="left/right margin, inches (the figure's own)")
    args = p.parse_args()
    st.use_font()
    FS_CAPTION = args.size

    base = Image.open(args.input).convert("RGB")
    W = base.width / args.dpi
    line_h = FS_CAPTION * 1.3 / 72                      # inches per line
    fig = plt.figure(figsize=(W, 1.0), facecolor="white")
    r = fig.canvas.get_renderer()

    def width(s, bold=False):                           # inches
        t = fig.text(0, 0, s, fontsize=FS_CAPTION, fontweight="bold" if bold else "normal")
        w = t.get_window_extent(r).width / fig.dpi
        t.remove()
        return w

    avail = W - 2 * args.margin_in
    lead = args.number + " "
    lines, cur, first = [], "", True
    for word in args.text.split():                      # greedy wrap; the bold lead takes room on line 1
        trial = (cur + " " + word).strip()
        room = avail - (width(lead, True) if first else 0)
        if cur and width(trial) > room:
            lines.append(cur); cur, first = word, False
        else:
            cur = trial
    lines.append(cur)
    assert not (args.one_line and len(lines) > 1), f"caption needs {len(lines)} lines at {FS_CAPTION} pt"

    pad_top, pad_bot = 0.06, 0.06
    H = pad_top + len(lines) * line_h + pad_bot
    fig.set_size_inches(W, H)
    for i, s in enumerate(lines):
        y = (H - pad_top - (i + 0.8) * line_h) / H
        x = args.margin_in
        if i == 0:
            fig.text(x / W, y, lead, fontsize=FS_CAPTION, fontweight="bold", color=st.INK, ha="left", va="baseline")
            x += width(lead, True)
        fig.text(x / W, y, s, fontsize=FS_CAPTION, color=st.INK, ha="left", va="baseline")
    fig.canvas.draw()
    for t in fig.texts:                                 # nothing may run past the right margin
        assert t.get_window_extent().x1 / fig.dpi <= W - args.margin_in + 1e-3, t.get_text()
    tmp = args.output.with_suffix(".strip.png")
    fig.savefig(tmp, dpi=args.dpi, facecolor="white")
    strip = Image.open(tmp).convert("RGB")
    tmp.unlink()
    if strip.width != base.width:                       # rounding: pad/crop to the figure's exact width
        canvas = Image.new("RGB", (base.width, strip.height), "white"); canvas.paste(strip, (0, 0)); strip = canvas
    out = Image.new("RGB", (base.width, base.height + strip.height), "white")
    out.paste(base, (0, 0)); out.paste(strip, (0, base.height))
    assert np.array_equal(np.asarray(out)[: base.height], np.asarray(base))   # the figure is untouched
    out.save(args.output, dpi=(args.dpi, args.dpi))
    print(f"→ {args.output} · {len(lines)} caption line(s) · +{strip.height} px under {base.width}x{base.height}")


if __name__ == "__main__":
    main()
