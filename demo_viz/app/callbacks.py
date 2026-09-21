"""Callbacks: clicking, scrubbing, role logic, exports."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update

from .. import palette
from ..core.figures import ViewOptions, scene_figure, space_chart
from ..core.role_logic import (
    auto_triplet,
    defender_reaction_index,
    rank_beneficiaries,
    rank_defenders,
)
from ..core.selection import Selection
from .components import CARD, LABEL
from .state import (
    get_bundle,
    onset_for,
    pipeline_source,
    pipeline_triplets,
    scene_options,
    time_marks,
)

ROLE_COLOUR = {
    "runner": palette.RUNNER,
    "defender": palette.DEFENDER,
    "beneficiary": palette.BENEFICIARY,
}
ROLE_LABEL = {"runner": "Runner", "defender": "Defender", "beneficiary": "Beneficiary"}


def register(app):
    _scene_callback(app)
    _selection_callback(app)
    _render_callback(app)
    _playback_callbacks(app)
    _export_callbacks(app)


# ---------------------------------------------------------------------------
# scene switching
# ---------------------------------------------------------------------------
def _scene_callback(app):
    @app.callback(
        Output("scene", "options"),
        Output("scene", "value"),
        Input("effects", "value"),
        State("scene", "value"),
    )
    def on_effects(effects, current):
        options = scene_options(tuple(effects or ("strong", "medium")))
        values = [o["value"] for o in options]
        if current in values:
            return options, current
        return options, (values[0] if values else "synthetic")

    @app.callback(
        Output("store-scene", "data"),
        Output("store-selection", "data"),
        Output("time", "max"),
        Output("time", "value"),
        Output("time", "marks"),
        Output("notes", "children"),
        Output("pick", "data"),
        Input("scene", "value"),
        Input("mode", "value"),
    )
    def on_scene(scene_id, mode):
        bundle = get_bundle(scene_id)
        scene = bundle.scene
        selection = _initial_selection(scene_id, bundle, mode)
        start = _opening_frame(scene_id, bundle, selection)
        notes = bundle.notes or "—"
        if bundle.effect:
            notes = f"[{bundle.effect}]  {notes}"
        return (
            {"scene_id": scene_id, "mode": mode},
            selection.to_dict(),
            max(scene.n_frames - 1, 1),
            int(start),
            time_marks(scene),
            notes,
            selection.pick,
        )


def _opening_frame(scene_id, bundle, selection: Selection) -> int:
    """Open on the moment the gain peaks, not on an empty pre-run frame."""

    scene, cache = bundle.scene, bundle.cache
    freeze_index, _ = onset_for(scene_id, selection.runner)
    if freeze_index is None:
        return scene.n_frames // 3
    if selection.defenders and selection.beneficiaries:
        peak = _peak_gain_index(bundle, selection, freeze_index)
        if peak is not None:
            return peak
    return int(min(freeze_index + int(1.5 * scene.fps), scene.n_frames - 1))


def _peak_gain_index(bundle, selection: Selection, freeze_index) -> int | None:
    """Frame where the beneficiary gains most against the no-reaction baseline."""

    cache = bundle.cache
    if not (selection.defenders and selection.beneficiaries) or freeze_index is None:
        return None
    swap = tuple((d, int(freeze_index), "hold") for d in selection.defenders)
    factual = np.sum([cache.residual_series(p) for p in selection.beneficiaries], axis=0)
    counter = np.sum([cache.residual_series(p, swap) for p in selection.beneficiaries], axis=0)
    gain = factual - counter
    after = cache.indices >= int(freeze_index)
    if not after.any():
        return None
    masked = np.where(after, gain, -np.inf)
    return int(cache.indices[int(np.argmax(masked))])


def _initial_selection(scene_id, bundle, mode) -> Selection:
    scene = bundle.scene
    if mode == "annotation" and (scene.runner_ids or scene.defender_ids):
        selection = Selection.from_scene(scene, source="annotation")
        selection.pick = "runner"
        return selection
    if mode == "pipeline":
        records = pipeline_triplets(scene_id)
        if records:
            return _selection_from_record(scene, records[0])
        suggested = auto_triplet(scene, bundle.cache)
        selection = _selection_from_triplet(suggested)
        selection.source = "suggested"
        return selection
    return Selection(source="manual")


def _selection_from_record(scene, record) -> Selection:
    def resolve(values, side):
        out = []
        for token in values or []:
            token = str(token)
            if token in scene.players:
                out.append(token)
                continue
            player = scene.by_shirt(token, side)
            if player is not None:
                out.append(player.player_id)
        return out

    return Selection(
        runners=resolve(record.get("runner") or record.get("runners"), "attack"),
        defenders=resolve(record.get("defender") or record.get("defenders"), "defend"),
        beneficiaries=resolve(
            record.get("beneficiary") or record.get("beneficiaries"), "attack"
        ),
        pick="runner",
        source="pipeline",
    )


def _selection_from_triplet(triplet) -> Selection:
    selection = Selection(pick="runner", source="suggested")
    for role, key in (("runner", "runner"), ("defender", "defender"),
                      ("beneficiary", "beneficiary")):
        value = triplet.get(key)
        if value:
            selection.ids(role).append(value)
    return selection


# ---------------------------------------------------------------------------
# clicking and role buttons
# ---------------------------------------------------------------------------
def _selection_callback(app):
    @app.callback(
        Output("store-selection", "data", allow_duplicate=True),
        Output("pick", "data", allow_duplicate=True),
        Input("pitch", "clickData"),
        Input("btn-reset", "n_clicks"),
        Input("btn-auto", "n_clicks"),
        Input("btn-swap", "n_clicks"),
        Input("slot-runner", "n_clicks"),
        Input("slot-beneficiary", "n_clicks"),
        Input("slot-defender", "n_clicks"),
        Input({"type": "cand-row", "role": ALL, "player": ALL}, "n_clicks"),
        Input({"type": "slot-clear", "role": ALL, "player": ALL}, "n_clicks"),
        State("pick", "data"),
        State("store-selection", "data"),
        State("store-scene", "data"),
        prevent_initial_call=True,
    )
    def on_pick(click, _reset, _auto, _swap, _sr, _sb, _sd, _rows, _clears,
                pick, stored, scene_store):
        trigger = callback_context.triggered_id
        fired = callback_context.triggered[0].get("value") or 0
        scene_id = (scene_store or {}).get("scene_id")
        if not scene_id:
            return no_update, no_update
        bundle = get_bundle(scene_id)
        selection = Selection.from_dict(stored)
        selection.pick = pick or selection.pick

        if trigger == "btn-reset":
            return Selection(pick="runner", source="manual").to_dict(), "runner"

        if trigger == "btn-swap":
            selection.swap_attack_roles()
            return selection.to_dict(), selection.pick

        if isinstance(trigger, str) and trigger.startswith("slot-"):
            selection.arm(trigger.split("-", 1)[1])
            return selection.to_dict(), selection.pick

        if isinstance(trigger, dict) and trigger.get("type") == "slot-clear":
            if not fired:
                return no_update, no_update
            role, player_id = trigger.get("role"), trigger.get("player")
            bucket = selection.ids(role)
            if player_id in bucket:
                bucket.remove(player_id)
            selection.pick = role
            selection.source = "manual"
            return selection.to_dict(), role

        if trigger == "btn-auto":
            runner = selection.runner
            suggested = auto_triplet(bundle.scene, bundle.cache, runner_id=runner)
            if not suggested:
                return no_update, no_update
            return _selection_from_triplet(suggested).to_dict(), "beneficiary"

        if isinstance(trigger, dict) and trigger.get("type") == "cand-row":
            # a row in the hint list is just another way to pick that player;
            # ignore the zero-click fire that pattern inputs emit on render
            if not fired:
                return no_update, no_update
            selection.arm(trigger.get("role", selection.pick))
            selection.apply_click(bundle.scene, trigger.get("player"))
            return selection.to_dict(), selection.pick

        player_id = _clicked_player(click)
        if player_id is None or player_id not in bundle.scene.players:
            return no_update, no_update

        selection.apply_click(bundle.scene, player_id)
        return selection.to_dict(), selection.pick


def _clicked_player(click) -> str | None:
    if not click:
        return None
    points = click.get("points") or []
    if not points:
        return None
    value = points[0].get("customdata")
    if isinstance(value, list):
        value = value[0] if value else None
    return value


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _render_callback(app):
    @app.callback(
        Output("pitch", "figure"),
        Output("chart", "figure"),
        Output("body-runner", "children"),
        Output("body-beneficiary", "children"),
        Output("body-defender", "children"),
        Output("slot-runner", "style"),
        Output("slot-beneficiary", "style"),
        Output("slot-defender", "style"),
        Output("dock-meta", "children"),
        Output("stat-label", "children"),
        Output("stat-value", "children"),
        Output("stat-value", "style"),
        Output("candidates-label", "children"),
        Output("candidates", "children"),
        Output("time-readout", "children"),
        Output("hint", "children"),
        Input("store-selection", "data"),
        Input("time", "value"),
        Input("layers", "value"),
        Input("wake-mode", "value"),
        Input("pick", "data"),
        State("store-scene", "data"),
    )
    def on_render(stored, index, layers, wake_mode, pick, scene_store):
        scene_id = (scene_store or {}).get("scene_id")
        if not scene_id:
            return (no_update,) * 16
        bundle = get_bundle(scene_id)
        scene, cache = bundle.scene, bundle.cache
        selection = Selection.from_dict(stored)
        selection.pick = pick or selection.pick
        index = int(np.clip(index or 0, 0, scene.n_frames - 1))

        freeze_index, onset_method = onset_for(scene_id, selection.runner)
        options = ViewOptions(layers=tuple(layers or ()), wake_mode=wake_mode or "gain")

        candidate_ids, candidate_rows, candidate_label = _candidates(
            bundle, selection, freeze_index, index
        )
        active_side = _ACTIVE_SIDE.get(selection.pick) if selection.armed else None
        figure = scene_figure(
            scene, index, selection, cache, options,
            freeze_index=freeze_index, candidates=tuple(candidate_ids),
            active_side=active_side,
        )
        chart = space_chart(scene, cache, selection, index, freeze_index)
        stat_label, stat_value, stat_style = _stat(bundle, selection, freeze_index, index)
        bodies = [_slot_body(scene, selection, role)
                  for role in ("runner", "beneficiary", "defender")]
        styles = [_slot_style(selection, role)
                  for role in ("runner", "beneficiary", "defender")]
        readout = f"{scene.times[index]:+.2f} s"
        return (
            figure, chart, *bodies, *styles,
            _dock_meta(scene, selection, freeze_index, onset_method),
            stat_label, stat_value, stat_style,
            candidate_label, candidate_rows, readout,
            _hint(selection),
        )


def _hint(selection: Selection) -> str:
    if selection.armed:
        return f"Now click a{'n' if selection.pick == 'attacker' else ''} " + {
            "runner": "attacking player.",
            "beneficiary": "attacking player.",
            "defender": "defending player.",
        }[selection.pick]
    if not selection.runners:
        return "Click an attacker, or a slot then a player."
    if not selection.defenders:
        return "Click a defender. Ringed ones react most."
    if not selection.beneficiaries:
        return "Click a teammate. Ringed ones gain most."
    return "Click a slot to replace it, or × to clear."


_ACTIVE_SIDE = {"runner": "attack", "beneficiary": "attack", "defender": "defend"}

SLOT_BASE = {
    "borderRadius": "9px", "padding": "7px 9px", "cursor": "pointer",
    "background": palette.INK, "transition": "border-color .12s, background .12s",
}


def _slot_style(selection: Selection, role: str) -> dict:
    colour = ROLE_COLOUR[role]
    if selection.armed and selection.pick == role:
        return {**SLOT_BASE, "border": f"1px solid {colour}",
                "background": "rgba(255,255,255,0.05)",
                "boxShadow": f"0 0 0 2px {colour}33"}
    if selection.ids(role):
        return {**SLOT_BASE, "border": f"1px solid {palette.GRID}"}
    return {**SLOT_BASE, "border": f"1px dashed {palette.GRID}"}


def _slot_body(scene, selection: Selection, role: str):
    colour = ROLE_COLOUR[role]
    ids = [pid for pid in selection.ids(role) if pid in scene.players]
    if not ids:
        armed = selection.armed and selection.pick == role
        return html.Span(
            "pick on pitch" if armed else "empty",
            style={"color": colour if armed else palette.TEXT_MUTED, "fontSize": "11px"},
        )
    chips = []
    for player_id in ids:
        player = scene.players[player_id]
        chips.append(html.Span(
            style={"background": colour, "color": palette.INK, "borderRadius": "6px",
                   "padding": "3px 5px 3px 8px", "fontSize": "11px", "fontWeight": 700,
                   "display": "inline-flex", "alignItems": "center", "gap": "6px",
                   "whiteSpace": "nowrap"},
            children=[
                html.Span(f"#{player.label} {_short(player.name)}"),
                html.Span("×", id={"type": "slot-clear", "role": role,
                                   "player": player_id},
                          n_clicks=0, className="chip-x",
                          style={"cursor": "pointer", "fontWeight": 700,
                                 "fontSize": "13px", "lineHeight": "11px"}),
            ],
        ))
    return chips


def _short(name: str, width: int = 16) -> str:
    return name if len(name) <= width else name[: width - 1] + "\u2026"


def _dock_meta(scene, selection: Selection, freeze_index, onset_method):
    parts = []
    if selection.runners and freeze_index is not None:
        parts.append(html.Div(
            f"run starts {scene.times[int(freeze_index)]:+.1f}s · {onset_method}",
            style={"marginBottom": "3px"}))
    if selection.source != "manual":
        note = selection.source
        if selection.source == "pipeline":
            _path, kind = pipeline_source()
            note = "from pipeline file" if kind == "pipeline" else "from example file"
        elif selection.source == "suggested":
            note = "suggested by demo_viz"
        parts.append(html.Div(note, style={"marginBottom": "3px"}))
    parts.append(html.Div(
        style={"display": "flex", "gap": "12px", "marginTop": "5px",
               "paddingTop": "6px", "borderTop": f"1px solid {palette.GRID}"},
        children=[
            _team_key(palette.ATTACK_NEUTRAL, scene.attacking_team_name, "attacking"),
            _team_key(palette.DEFEND_NEUTRAL, scene.defending_team_name, "defending"),
        ],
    ))
    return parts


def _team_key(colour, team, side):
    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "6px",
               "fontSize": "11px", "color": palette.TEXT_MUTED, "minWidth": 0},
        children=[
            html.Span(style={"background": colour, "width": "11px", "height": "11px",
                             "borderRadius": "2px", "display": "inline-block",
                             "flex": "0 0 11px"}),
            html.Span(f"{team}", title=f"{team} ({side})",
                      style={"overflow": "hidden", "textOverflow": "ellipsis",
                             "whiteSpace": "nowrap"}),
        ],
    )


def _stat(bundle, selection: Selection, freeze_index, index):
    cache = bundle.cache
    style = {"fontSize": "30px", "fontWeight": 700, "lineHeight": "1.05"}
    if not selection.beneficiaries:
        return "Opened space", "—", {**style, "color": palette.TEXT_MUTED}
    slot = cache.slot_for(index)
    factual = cache.combined(selection.beneficiaries, slot)
    if not selection.defenders or freeze_index is None:
        return ("Space held", f"{factual.value:.1f}",
                {**style, "color": palette.BENEFICIARY})
    swap = tuple((d, int(freeze_index), "hold") for d in selection.defenders)
    counter = cache.combined(selection.beneficiaries, slot, swap)
    gain = factual.value - counter.value
    colour = palette.BENEFICIARY if gain >= 0 else palette.RUNNER
    return "Opened space", f"{gain:+.1f}", {**style, "color": colour}


def _candidates(bundle, selection: Selection, freeze_index, index):
    scene, cache = bundle.scene, bundle.cache
    if not selection.runners:
        return (), html.Div("Pick a runner first.",
                            style={"color": palette.TEXT_MUTED, "fontSize": "12px"}), "Hints"

    if selection.pick == "beneficiary" or (selection.defenders and not selection.beneficiaries):
        if not selection.defenders:
            return (), html.Div("Pick a defender first.",
                                style={"color": palette.TEXT_MUTED, "fontSize": "12px"}), "Gains most"
        rows = rank_beneficiaries(
            cache, selection.runner, tuple(selection.defenders),
            int(freeze_index or 0), cache.slot_for(index), limit=6,
        )
        ids = tuple(r.player_id for r in rows[:3] if r.gain > 0.05)
        return (ids,
                _candidate_list(scene, rows, palette.BENEFICIARY, selection, "beneficiary"),
                "Gains most  ·  click to pick")

    rows = rank_defenders(scene, selection.runner, from_index=int(freeze_index or 0), limit=6)
    ids = tuple(r.player_id for r in rows[:4])
    return (ids,
            _candidate_list(scene, rows, palette.DEFENDER, selection, "defender"),
            "Reacts most  ·  click to pick")


def _candidate_list(scene, rows, colour, selection: Selection, role: str):
    if not rows:
        return html.Div("—", style={"color": palette.TEXT_MUTED, "fontSize": "12px"})
    items = []
    for row in rows:
        chosen = selection.role_of(row.player_id) is not None
        items.append(html.Div(
            id={"type": "cand-row", "role": role, "player": row.player_id},
            n_clicks=0,
            style={"display": "flex", "alignItems": "baseline", "gap": "8px",
                   "padding": "5px 4px", "cursor": "pointer", "borderRadius": "5px",
                   "background": palette.INK if chosen else "transparent",
                   "borderBottom": f"1px solid {palette.GRID}"},
            children=[
                html.Span(f"#{row.shirt}", style={
                    "color": colour if chosen else palette.TEXT_PRIMARY,
                    "fontWeight": 700, "fontSize": "12px",
                    "width": "34px", "flex": "0 0 34px"}),
                html.Span(row.name, style={
                    "color": palette.TEXT_SECONDARY, "fontSize": "12px",
                    "flex": "1 1 auto", "overflow": "hidden",
                    "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                html.Span(row.caption(), style={
                    "color": palette.TEXT_MUTED, "fontSize": "11px",
                    "fontFamily": "ui-monospace, Menlo, monospace"}),
            ],
        ))
    return items


# ---------------------------------------------------------------------------
# playback
# ---------------------------------------------------------------------------
def _playback_callbacks(app):
    @app.callback(
        Output("tick", "disabled"),
        Output("btn-play", "children"),
        Input("btn-play", "n_clicks"),
        State("tick", "disabled"),
        prevent_initial_call=True,
    )
    def on_play(_clicks, disabled):
        return (not disabled), ("▶" if not disabled else "❚❚")

    @app.callback(
        Output("time", "value", allow_duplicate=True),
        Input("btn-jump-run", "n_clicks"),
        Input("btn-jump-reaction", "n_clicks"),
        Input("btn-jump-peak", "n_clicks"),
        State("store-selection", "data"),
        State("store-scene", "data"),
        State("time", "value"),
        prevent_initial_call=True,
    )
    def on_jump(_run, _reaction, _peak, stored, scene_store, current):
        scene_id = (scene_store or {}).get("scene_id")
        if not scene_id:
            return no_update
        bundle = get_bundle(scene_id)
        selection = Selection.from_dict(stored)
        freeze_index, _ = onset_for(scene_id, selection.runner)
        trigger = callback_context.triggered_id
        if trigger == "btn-jump-run":
            return int(freeze_index) if freeze_index is not None else no_update
        if trigger == "btn-jump-reaction":
            if not (selection.runner and selection.defenders):
                return no_update
            index, _method = defender_reaction_index(
                bundle.scene, selection.runner, selection.defenders[0],
                from_index=int(freeze_index or 0),
            )
            return int(index) if index is not None else no_update
        peak = _peak_gain_index(bundle, selection, freeze_index)
        return int(peak) if peak is not None else no_update

    @app.callback(
        Output("time", "value", allow_duplicate=True),
        Input("tick", "n_intervals"),
        State("time", "value"),
        State("time", "max"),
        prevent_initial_call=True,
    )
    def on_tick(_n, value, maximum):
        step = 2
        nxt = int((value or 0) + step)
        return 0 if nxt > (maximum or 0) else nxt


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------
def _export_callbacks(app):
    @app.callback(
        Output("download", "data"),
        Output("export-status", "children"),
        Input("btn-png", "n_clicks"),
        Input("btn-json", "n_clicks"),
        State("store-selection", "data"),
        State("store-scene", "data"),
        State("time", "value"),
        State("layers", "value"),
        prevent_initial_call=True,
    )
    def on_export(_png, _json, stored, scene_store, index, layers):
        trigger = callback_context.triggered_id
        scene_id = (scene_store or {}).get("scene_id")
        if not scene_id:
            return no_update, ""
        bundle = get_bundle(scene_id)
        scene = bundle.scene
        selection = Selection.from_dict(stored)
        index = int(np.clip(index or 0, 0, scene.n_frames - 1))
        stamp = datetime.now().strftime("%H%M%S")
        slug = "".join(ch if ch.isalnum() else "_" for ch in scene_id)

        if trigger == "btn-json":
            freeze_index, method = onset_for(scene_id, selection.runner)
            payload = {
                "scene_id": scene_id,
                "source": selection.source,
                "frame_index": index,
                "time_s": round(float(scene.times[index]), 3),
                "run_start_s": (round(float(scene.times[int(freeze_index)]), 3)
                                if freeze_index is not None else None),
                "run_start_method": method,
                "roles": {
                    role: [
                        {"player_id": pid,
                         "shirt": scene.players[pid].label,
                         "name": scene.players[pid].name}
                        for pid in selection.ids(role) if pid in scene.players
                    ]
                    for role in ("runner", "defender", "beneficiary")
                },
            }
            return (dcc.send_string(json.dumps(payload, indent=2, ensure_ascii=False),
                                    f"{slug}_{stamp}.json"),
                    "saved json")

        try:
            path = _render_png(bundle, selection, index, layers, slug, stamp)
        except Exception as error:                     # keep the app alive
            return no_update, f"png failed: {error}"
        return dcc.send_file(str(path)), f"saved {path.name}"


def _render_png(bundle, selection: Selection, index, layers, slug, stamp):
    """Reuse the video renderer for a full-quality still of the current pick.

    The scene object is shared with every other callback, so the roles it
    carries are swapped in and restored around the render rather than left
    pointing at whatever was last exported.
    """

    import matplotlib

    matplotlib.use("Agg")

    from ..animate import render_still
    from ..config import demo_paths
    from ..loader import figure_config_for
    from ..quantities import compute_surfaces
    from ..story import build_storyboard

    scene = bundle.scene
    snapshot = (scene.runner_ids, scene.defender_ids, scene.beneficiary_ids,
                scene.surfaces, dict(scene.moments), dict(scene.series), scene.source)
    try:
        scene.runner_ids = tuple(selection.runners)
        scene.defender_ids = tuple(selection.defenders)
        scene.beneficiary_ids = tuple(selection.beneficiaries)
        freeze_index, _ = onset_for(scene.scene_id, selection.runner)
        if freeze_index is not None:
            scene.moments["onset"] = float(scene.times[int(freeze_index)])
        if selection.beneficiaries and selection.defenders:
            scene.surfaces = compute_surfaces(scene, every=5, grid_resolution_m=1.5)
            _fill_series(scene)
        else:
            scene.surfaces = None
            for key in ("_surface_t", "_factual_value", "_counterfactual_value",
                        "_value_gain"):
                scene.series.pop(key, None)
        if selection.source == "manual":
            scene.source = "interactive pick"
        elif selection.source == "suggested":
            scene.source = "demo_viz suggestion"
        config = figure_config_for(scene)
        config.wake_mode = "residual" if scene.surfaces is not None else "geometric"
        storyboard = build_storyboard(scene, wake_is_illustrative=scene.surfaces is None)
        out = demo_paths().export_dir / f"{slug}_{stamp}.png"
        return render_still(scene, storyboard, out,
                            at_time=float(scene.times[index]), figure_config=config)
    finally:
        (scene.runner_ids, scene.defender_ids, scene.beneficiary_ids,
         scene.surfaces, scene.moments, scene.series, scene.source) = snapshot


def _fill_series(scene) -> None:
    """Populate the series the still renderer's chart reads from a surface stack."""

    stack = scene.surfaces
    if stack is None or stack.factual_value is None:
        return
    scene.series["_surface_t"] = stack.times
    scene.series["_factual_value"] = stack.factual_value
    if stack.counterfactual_value is not None:
        scene.series["_counterfactual_value"] = stack.counterfactual_value
        scene.series["_value_gain"] = stack.factual_value - stack.counterfactual_value
