// The public panel: who is in the play, and the numbers for the open mode.
//
// Mode content, not shell -- the showcase fetches it on its first render, the
// explorer never does. Everything it needs arrives in one context object, so
// this module holds no state of its own and nothing here recomputes science:
// the marking distance and the reaction are read from `data/compare`, the
// space from the influence cache, and the equilibrium and evaluation rows from
// the solved panels and the paper-story payload.

export function renderLegend(c) {
  const list = c.el("legend-list");
  if (!list) return;
  list.replaceChildren();
  const card = c.el("legend-card");
  if (card) card.hidden = !c.isPublic();
  if (!c.isPublic()) return;
  const { scene } = c.state;
  const roles = c.solverRolesAt(c.state.frame) || {};
  const order = ["runner", "ball carrier", "beneficiary", "defender"];
  const seen = new Set();
  const entries = [];
  for (const [playerId, solverRole] of Object.entries(roles)) {
    const player = scene?.byId.get(playerId);
    const name = c.PUBLIC_ROLE[solverRole];
    if (!player || !name || seen.has(playerId)) continue;
    seen.add(playerId);
    entries.push({ player, solverRole, name });
  }
  entries.sort((a, b) => order.indexOf(a.solverRole) - order.indexOf(b.solverRole));

  const compared = c.comparedPlayer();
  if (compared && !seen.has(compared)) {
    const player = scene?.byId.get(compared);
    if (player) {
      entries.push({
        player,
        solverRole: player.side === "defend" ? "defender" : "beneficiary",
        name: player.side === "defend" ? "Defender" : "Teammate",
        comparing: true,
      });
    }
  }

  for (const entry of entries) {
    const row = document.createElement("li");
    row.dataset.player = entry.player.id;
    row.classList.toggle("is-compared", Boolean(entry.comparing));
    const colour = entry.solverRole === "defender"
      ? (c.state.pitch?.roles?.defender || c.ROLE_COLOUR.defender)
      : (c.state.pitch?.roles?.runner || c.ROLE_COLOUR.runner);
    row.style.color = colour;
    const mark = document.createElement("span");
    mark.className = `lgmark is-${c.SHAPE_CLASS[entry.solverRole] || "circle"}`;
    mark.style.background = colour;
    const shirt = document.createElement("span");
    shirt.className = "lgshirt";
    shirt.textContent = `#${entry.player.shirt}`;
    const who = document.createElement("span");
    who.className = "lgname";
    who.textContent = c.surnameOf(entry.player);
    const what = document.createElement("span");
    what.className = "lgrole";
    what.textContent = entry.comparing ? "comparing" : entry.name;
    row.append(mark, shirt, who, what);
    // the panel is the same control as the pitch: Dilemma compares, the other
    // modes have numbers for the curated cast only and ignore the click
    row.addEventListener("click", () => {
      if (c.state.mode === "counterfactual") c.compareWith(entry.player.id);
    });
    list.appendChild(row);
  }
  const foot = c.el("legend-foot");
  if (foot) foot.hidden = !compared;
}

/**
 * The panel's numbers, big enough to read: one card, one row per quantity.
 *
 * A comparison shows the compared player's value as the headline and the
 * curated one beneath it, so which is which never needs a sentence.
 */
export function renderMetrics(c, slot) {
  const card = c.el("metrics-card");
  const list = c.el("metrics-list");
  if (!card || !list) return;
  const mode = c.state.mode;
  const on = c.isPublic() && (mode === "counterfactual" || mode === "game_solution"
                            || mode === "evaluation");
  card.hidden = !on;
  if (!on) return;
  const moment = mode === "evaluation" ? momentLabel(c) : null;
  c.el("metrics-h").textContent = mode === "counterfactual" ? "The dilemma"
    : (mode === "game_solution" ? "Equilibrium"
       : `Player evaluation${moment ? ` \u00b7 ${moment}` : ""}`);
  list.replaceChildren();
  list.classList.remove("is-moments", "is-two-up");
  // the equilibrium is three separate solves, so it gets all three at once
  if (mode === "game_solution") { renderMomentTable(c, list); return; }
  const rowsOf = mode === "counterfactual" ? dilemmaMetrics(c, slot)
    : evaluationMetrics(c);
  // Dilemma reads as two columns while a comparison is open: the curated play
  // on the left, the one the reader built on the right, the same three rows
  // across both. Everywhere else is one column.
  const comparing = mode === "counterfactual" && rowsOf.some((r) => r[2] != null);
  list.classList.toggle("is-two-up", comparing);
  if (comparing) {
    const head = document.createElement("div");
    head.className = "m-head";
    head.append(cell("", "m-label"), cell("Default", "m-col"),
                cell("Selected", "m-col"));
    list.append(head);
  }
  for (const [label, value, was, trend] of rowsOf) {
    const row = document.createElement("div");
    row.className = "m-row";
    row.append(cell(label, "m-label"));
    if (comparing) {
      row.append(cell(was == null ? "\u2014" : was, "m-value is-was"),
                 cell(value, `m-value${trend ? ` is-${trend}` : ""}`));
    } else {
      row.append(cell(value, `m-value${trend ? ` is-${trend}` : ""}`));
    }
    list.append(row);
  }
}

/**
 * The equilibrium read at every solved moment, not only the one on screen.
 *
 * The bundle solves three separate games -- 0.0, 0.6 and 1.2 s -- and a reader
 * comparing them should not have to hold two of them in memory while the
 * playhead moves. The column the playhead is standing on is marked, so the
 * table and the pitch always agree about which game is being drawn.
 */
function renderMomentTable(c, list) {
  const data = c.solverFor();
  if (data?.kind !== "bundle_panels") return;
  const panels = (data.panels || []).slice().sort((a, b) => a.dt - b.dt);
  if (!panels.length) return;
  const here = c.panelAt(data, c.state.frame)?.panel;
  const rows = [];
  const add = (label, read) => {
    const cells = panels.map(read);
    if (cells.some((v) => v != null)) rows.push([label, cells]);
  };
  add("Defensive dilemma", (p) => (p.dilemma?.is_dilemma == null
    ? null : (p.dilemma.is_dilemma ? "Yes" : "No")));
  // the two readings of the same attack, and the gap between them: what a
  // defender held to his observed command makes the attack look worth,
  // against what it is worth once he may answer
  add("Fixed defender", (p) => (p.static?.static_attack_appeared == null
    ? null : c.NUMBER(p.static.static_attack_appeared, 2)));
  add("Responding defender", (p) => (p.static?.static_attack_responsive == null
    ? null : c.NUMBER(p.static.static_attack_responsive, 2)));
  add("Fixed-defence overestimate", (p) => (p.static?.static_attack_loss_rel == null
    ? null : `+${c.NUMBER(100 * p.static.static_attack_loss_rel, 0)}%`));
  if (!rows.length) return;
  list.classList.add("is-moments");
  list.style.setProperty("--moment-cols", String(panels.length));
  const head = document.createElement("div");
  head.className = "m-head";
  head.append(cell("", "m-label"));
  for (const panel of panels) {
    head.append(cell(`${panel.dt.toFixed(1)} s`,
                     `m-col${panel === here ? " is-now" : ""}`));
  }
  list.append(head);
  for (const [label, cells] of rows) {
    const row = document.createElement("div");
    row.className = "m-row";
    row.append(cell(label, "m-label"));
    cells.forEach((value, i) => row.append(
      cell(value == null ? "\u2014" : value,
           `m-value${panels[i] === here ? " is-now" : " is-other"}`)));
    list.append(row);
  }
}

/** Which solved moment the panel's numbers come from. */
function momentLabel(c) {
  const data = c.solverFor();
  if (data?.kind !== "bundle_panels") return null;
  const found = c.panelAt(data, c.state.frame);
  return found ? `${found.panel.dt.toFixed(1)} s` : null;
}

function cell(text, className) {
  const node = document.createElement("div");
  node.className = className;
  node.textContent = text;
  return node;
}

/**
 * Marking distance, reaction and space created.
 *
 * The first two are read from `data/compare`, which ran the repository's
 * `dynamic_marking.marking_sample` and `role_logic.defender_reaction_index`;
 * the third is the influence cache's own number. None is a selection rule.
 */
function dilemmaMetrics(c, slot) {
  const { scene, selection, cache } = c.state;
  const lib = c.compareLib();
  const payload = c.compareFor();
  const defenderId = selection.defenders[0];
  const teammateId = selection.beneficiaries[0];
  const out = [];

  // the marking numbers follow whichever runner is in question, measured for
  // that pairing: every outfield attacker has his own rows in `data/compare`
  const other = c.state.compare?.defender || null;
  const sweptRunner = c.state.compare?.runner || null;
  if (lib && payload) {
    // the pairing the reader built -- either half of it may differ from the
    // curated play, and both halves are read together: a compared defender is
    // measured against the compared runner, not against the curated one
    const show = lib.defenderRow(payload, other || defenderId, sweptRunner);
    // the Default column is always the curated pairing, so the two columns
    // answer "against what?" the same way however many roles were swapped.
    // It appears whenever any comparison is open, even one that leaves the
    // marking alone: an empty cell beside a number reads as a missing number.
    const against = c.state.compare ? lib.defenderRow(payload, defenderId) : null;
    // both numbers are read where the playhead stands, so they move with the
    // play rather than reporting one window mean for the whole clip
    const frame = c.state.frame;
    const times = scene?.times || [];
    const marking = (row) => lib.metres(lib.markingAt(row, frame));
    const reaction = (row) => lib.reactionText(
      row, lib.secondsToReaction(payload, row, frame, times));
    out.push(["Marking distance", marking(show),
              against ? marking(against) : null]);
    out.push(["Reaction", reaction(show), against ? reaction(against) : null]);
  }

  if (cache && defenderId) {
    // the same held-defender device the pitch draws, so the number under the
    // field is the field
    const swap = c.state.compare?.defender
      ? [{ playerId: c.state.compare.defender,
           freezeIndex: c.freezeIndex(), mode: "hold" }]
      : c.swapSpec();
    const spaceOf = (id) => (id
      ? cache.combined([id], slot).value - cache.combined([id], slot, swap).value
      : null);
    const show = (value) => {
      if (value == null) return "\u2014";
      const rounded = Math.abs(value) < 0.05 ? 0 : value;
      return `${rounded > 0 ? "+" : ""}${c.NUMBER(rounded, 1)} m\u00b2`;
    };
    const them = c.state.compare?.teammate;
    // before -> after: the space he had with the defender held at the run's
    // onset, and the space he actually had. The delta is the gap between them.
    const spanOf = (id) => {
      if (!id) return null;
      const after = cache.combined([id], slot).value;
      const before = cache.combined([id], slot, swap).value;
      return { before, after };
    };
    const span = (s) => (s
      ? `${c.NUMBER(s.before, 1)} \u2192 ${c.NUMBER(s.after, 1)} m\u00b2` : "\u2014");
    const trend = (s) => {
      if (!s) return null;
      const d = s.after - s.before;
      return Math.abs(d) < 0.05 ? null : (d > 0 ? "up" : "down");
    };
    // the Default column is always the curated play: the curated teammate,
    // against the curated defender being held. Comparing a defender changes
    // which hold the Selected column uses, so both columns stay meaningful.
    const mineSpan = spanOf(them || teammateId);
    const comparing = Boolean(them || c.state.compare?.defender
                              || c.state.compare?.runner);
    const wasSpan = comparing
      ? { before: cache.combined([teammateId], slot, c.swapSpec()).value,
          after: cache.combined([teammateId], slot).value }
      : null;
    out.push(["Space created", span(mineSpan), wasSpan ? span(wasSpan) : null,
              trend(mineSpan)]);
  }
  return out;
}

/**
 * The ranked options for the role and solved moment on screen.
 *
 * Clicking one draws it on the pitch beside what he did and the best-valued
 * alternative. The numbers under the list are the bundle's own for the
 * observed action; the list's own order is the exported rank.
 */
export function renderOptions(c) {
  const host = c.el("options-card");
  const list = c.el("options-list");
  if (!host || !list) return;
  const on = c.isPublic() && c.state.mode === "evaluation";
  host.hidden = !on;
  if (!on) return;
  const lib = c.optionsLib();
  const payload = lib ? c.optionsFor() : null;
  const entry = payload && lib
    ? lib.at(payload, c.momentSeconds(), c.solverRoleOfStory()) : null;
  list.replaceChildren();
  if (!entry) return;
  const observed = lib.observedOption(entry);
  const best = lib.bestOption(entry);
  for (const option of entry.options) {
    const row = document.createElement("button");
    row.type = "button";
    const chosen = c.state.option === option.command;
    // the row and the line on the pitch are the same colour, so a click needs
    // no legend: sky for the best-valued option, pink for the one picked
    const accent = chosen && option !== best ? " is-picked"
      : (option === best ? " is-bestrow" : "");
    row.className = `optrow${chosen ? " is-on" : ""}${accent}`;
    const rank = document.createElement("span");
    rank.className = "optrank";
    // several options can be genuinely tied -- the bundle's own rule gives
    // them all the best position -- so a tie says so instead of printing
    // "1st" three times with no explanation
    const tied = entry.options.filter((o) => o.rank === option.rank).length > 1;
    rank.textContent = entry.rank_partial ? "\u2014"
      : `${lib.ordinal(option.rank)}${tied ? "=" : ""}`;
    const name = document.createElement("span");
    name.className = "optname";
    name.textContent = c.publicActionName(c.solverRoleOfStory(), option.label)
      || option.label;
    const tags = document.createElement("span");
    tags.className = "opttags";
    if (option === best) tags.append(tag("Best"));
    if (option === observed) tags.append(tag("Played"));
    const prob = document.createElement("span");
    prob.className = "optprob";
    prob.textContent = `${Math.round(option.prob * 100)}%`;
    row.append(rank, name, tags, prob);
    row.addEventListener("click", () => {
      c.state.option = chosen ? null : option.command;
      c.render();
    });
    list.appendChild(row);
  }
}

function tag(text) {
  const node = document.createElement("i");
  node.className = `opttag is-${text.toLowerCase()}`;
  node.textContent = text;
  return node;
}

/** What he did, and how it compared. */
function evaluationMetrics(c) {
  const story = c.storyFor();
  const block = story?.evaluation?.[c.state.storyRole];
  if (!block) return [];
  const atMoment = c.metricsAtFrame(block, c.state.frame);
  const shown = atMoment ? atMoment.metrics : block.metrics;
  const byName = new Map(shown.map((m) => [m.name, m]));
  const value = (name) => {
    const record = byName.get(name);
    return record && record.availability === "available" ? record.value : null;
  };
  const out = [];
  const first = c.presentSeries(block)[0]?.frames?.[0];
  if (atMoment && first != null && atMoment.frame === first) {
    const observed = c.publicObservedRow(story?.counterfactual?.[c.state.storyRole])[0];
    if (observed) out.push(["Observed", observed[1]]);
  }
  const rank = value("observed_action_rank");
  if (rank != null) out.push(["Rank", c.ordinal(rank)]);
  const probability = value("similarity_to_optimal");
  if (probability != null) out.push(["Probability under equilibrium", c.percent(probability)]);
  const regret = value("regret");
  if (regret != null) {
    // the value gap from the best-valued feasible action, which is what
    // "regret" means here; the public name says it in words
    out.push(["Expected loss vs best", c.NUMBER(regret, 2)]);
  }
  return out;
}

/**
 * Player evaluation's role picker, in the scene's own terms.
 *
 * Only roles the solved moment actually has: a 2v1 has a ball carrier, a
 * runner and a defender; a 3v1 has a runner, a teammate and a defender, and
 * its carrier is the scripted passer, who makes no decision to evaluate.
 */
function renderEvalRoles(c) {
  const node = c.el("sweep-pick");
  if (!node || !c.isPublic() || c.state.mode !== "evaluation") return false;
  const story = c.storyFor();
  const roles = ["runner", "passer", "defender"].filter(
    (role) => story?.evaluation?.[role]);
  node.hidden = roles.length < 2;
  node.replaceChildren();
  if (node.hidden) return true;
  for (const role of roles) {
    const button = document.createElement("button");
    button.className = `seg${c.state.storyRole === role ? " is-on" : ""}`;
    button.type = "button";
    button.textContent = c.PUBLIC_STORY_ROLE[role] || c.STORY_ROLE_LABEL[role];
    button.addEventListener("click", () => c.setStoryRole(role));
    node.appendChild(button);
  }
  return true;
}

/** "Compare as: Runner | Teammate | Defender", in Dilemma only. */
export function renderSweep(c) {
  const node = c.el("sweep-pick");
  if (!node) return;
  if (renderEvalRoles(c)) return;      // the same strip serves evaluation
  const on = c.isPublic() && c.state.mode === "counterfactual";
  node.hidden = !on;
  node.replaceChildren();
  if (!on) return;
  for (const [role, label] of [["runner", "Runner"], ["teammate", "Teammate"],
                               ["defender", "Defender"]]) {
    const button = document.createElement("button");
    button.className = `seg${c.state.sweep === role ? " is-on" : ""}`;
    button.type = "button";
    button.dataset.sweep = role;
    button.textContent = label;
    button.addEventListener("click", () => {
      if (c.state.sweep === role) return;
      // which role the next click fills; what is already chosen stays
      c.state.sweep = role;
      c.render();
    });
    node.appendChild(button);
  }
}

/**
 * Player evaluation on the pitch: what he did, the best-valued option, and
 * whichever ranked option the reader picked.
 */
export function renderDecisionPaths(c, index) {
  if (!c.isPublic() || c.state.mode !== "evaluation") return;
  const lib = c.optionsLib();
  const payload = lib ? c.optionsFor() : null;
  if (!lib || !payload) return;
  const data = c.solverFor();
  if (data?.kind !== "bundle_panels") return;
  const found = c.panelAt(data, index);
  if (!found?.exact) return;              // only at a solved moment
  const role = c.solverRoleOfStory();
  const entry = lib.at(payload, found.panel.dt, role);
  if (!entry) return;
  const figure = c.figureLib();
  const colour = figure?.ROLE_COLOR?.[role] || c.ROLE_COLOUR.runner;
  const toView = (path) => (path || []).map(([x, y]) => c.viewVector0(c.state.scene, x, y));
  const selected = (entry.options || []).find((o) => o.command === c.state.option);
  const best = lib.bestOption(entry);
  const observed = lib.observedOption(entry);
  lib.drawDecision(c.state.pitch, c.state.scene, {
    // every feasible option, so the three that are named have a fan to be
    // named out of rather than standing alone
    all: (entry.options || []).map((option) => toView(option.path)),
    best: best ? toView(best.path) : null,
    observed: observed ? toView(observed.path) : null,
    selected: selected ? toView(selected.path) : null,
    colour, solverRole: role,
  });
}

/**
 * The three defender-start grids for this scene, or null.
 *
 * Null is the normal c.state: the files are not in this build, so the control
 * that would draw them stays hidden. Nothing is estimated in their place.
 */
function gridsFor(c) {
  const data = c.solverFor();
  const code = data?.code;
  if (!code || data.kind !== "bundle_panels") return null;
  if (c.state.grids?.code === code) return c.state.grids.moments;
  const lib = c.gridLib();
  if (!lib) return null;
  if (c.state.grids?.pending === code) return null;
  c.state.grids = { pending: code };
  const moments = (data.panels || []).map((panel) => panel.dt);
  Promise.all(moments.map((dt) => lib.loadGrid(code, dt))).then((grids) => {
    if (c.solverFor()?.code !== code) return;
    const map = new Map();
    moments.forEach((dt, index) => map.set(dt.toFixed(1), grids[index]));
    // all or nothing: a partly-covered field would show one moment shaded and
    // the next bare, which reads as a result rather than a missing file
    const complete = grids.every(Boolean);
    c.state.grids = { code, moments: complete ? map : null };
    c.render();
  });
  return null;
}

/** Figure 2's background and move field, when the grids exist. */
export function renderDefenderField(c, index) {
  const control = c.el("field-control");
  const grids = c.state.mode === "game_solution" && c.isPublic() ? gridsFor(c) : null;
  if (control) control.hidden = !grids;
  if (!grids || !c.state.field) return;
  const lib = c.gridLib();
  const data = c.solverFor();
  const panel = c.panelAt(data, index)?.panel;
  const grid = panel ? grids.get(panel.dt.toFixed(1)) : null;
  if (!lib || !grid) return;
  lib.drawValueField(c.state.pitch, lib.valueRaster(grid), c.state.scene,
                     { low: lib.SHADE.low, high: lib.SHADE.high });
  lib.drawMoveField(c.state.pitch, lib.movePoints(grid), c.state.scene,
                    { colour: lib.FLOW.colour, alpha: lib.FLOW.alpha });
}
