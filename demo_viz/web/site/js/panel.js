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

  const compared = c.state.compare?.defender || c.state.compare?.teammate || null;
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
    const who = document.createElement("span");
    who.className = "lgname";
    who.textContent = c.surnameOf(entry.player);
    const what = document.createElement("span");
    what.className = "lgrole";
    what.textContent = entry.comparing ? "comparing" : entry.name;
    row.append(mark, who, what);
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
  c.el("metrics-h").textContent = mode === "counterfactual" ? "The dilemma"
    : (mode === "game_solution" ? "Equilibrium" : "Player evaluation");
  list.replaceChildren();
  const rowsOf = mode === "counterfactual" ? dilemmaMetrics(c, slot)
    : (mode === "game_solution" ? nashMetrics(c) : evaluationMetrics(c));
  for (const [label, value, was] of rowsOf) {
    const row = document.createElement("div");
    row.className = "m-row";
    const name = document.createElement("div");
    name.className = "m-label";
    name.textContent = label;
    const pair = document.createElement("div");
    pair.className = "m-pair";
    const main = document.createElement("div");
    main.className = "m-value";
    main.textContent = value;
    pair.append(main);
    if (was != null) {
      const before = document.createElement("span");
      before.className = "m-was";
      before.textContent = was;
      pair.append(before);
    }
    row.append(name, pair);
    list.append(row);
  }
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

  const other = c.state.compare?.defender || null;
  if (lib && payload) {
    const mine = lib.defenderRow(payload, defenderId);
    const theirs = other ? lib.defenderRow(payload, other) : null;
    out.push(["Marking distance",
              lib.metres((theirs || mine)?.marking_distance_m),
              theirs ? lib.metres(mine?.marking_distance_m) : null]);
    out.push(["Reaction", lib.reactionText(theirs || mine),
              theirs ? lib.reactionText(mine) : null]);
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
    out.push(["Space created", show(spaceOf(them || teammateId)),
              them ? show(spaceOf(teammateId)) : null]);
  }
  return out;
}

/** The solved moment's dilemma and the static comparison, as the paper has it. */
function nashMetrics(c) {
  const data = c.solverFor();
  if (data?.kind !== "bundle_panels") return [];
  const found = c.panelAt(data, c.state.frame);
  if (!found) return [];
  const d = found.panel.dilemma || {};
  const s = found.panel.static || {};
  const out = [["Dilemma", d.is_dilemma ? "yes" : "no"]];
  if (s.static_attack_appeared != null) {
    out.push(["Fixed defence", c.NUMBER(s.static_attack_appeared, 2)]);
    out.push(["Responding defence", c.NUMBER(s.static_attack_responsive, 2)]);
    if (s.static_attack_loss_rel != null) {
      out.push(["Overestimate", `${c.NUMBER(100 * s.static_attack_loss_rel, 0)}%`]);
    }
  }
  return out;
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
  if (probability != null) out.push(["Equilibrium probability", c.percent(probability)]);
  const regret = value("regret");
  if (regret != null) out.push(["Regret", c.NUMBER(regret, 2)]);
  return out;
}
