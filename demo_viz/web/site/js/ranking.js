// Candidate rankings, ported from demo_viz.core.role_logic.
//
// "Reacts most" is a geometric heuristic on the goal-side marking target from
// offball_value.dynamic_marking -- not a model of defensive intent.
// "Gains most" is the residual-space difference itself, so it is the same
// quantity the demo is about.

import { velocityAt } from "./influence.js";

const GOAL_SIDE_OFFSET = 1.5;
const LOOKAHEAD_S = 0.4;

function goalSideTarget(ax, ay, gx, gy, offset) {
  const dx = gx - ax;
  const dy = gy - ay;
  const distance = Math.hypot(dx, dy);
  if (distance <= 1e-12) return [ax, ay];
  return [ax + (offset * dx) / distance, ay + (offset * dy) / distance];
}

/** Opponents ordered by how much they behave like a reaction to the runner. */
export function rankDefenders(scene, runnerId, fromIndex = 0, minimumSpeed = 1.0) {
  const runner = scene.byId.get(runnerId);
  if (!runner) return [];
  const window = Math.max(2, Math.round(0.4 * scene.fps));
  const goalX = -scene.attacking_direction * 52.5;
  const goalY = 0;
  const rows = [];

  for (const defender of scene.defenders) {
    if (defender.gk) continue;
    let distanceSum = 0;
    let distanceCount = 0;
    let alignSum = 0;
    let alignCount = 0;
    for (let index = fromIndex; index < scene.n_frames; index += 1) {
      const ax = runner.x[index];
      const ay = runner.y[index];
      const dx = defender.x[index];
      const dy = defender.y[index];
      if (![ax, ay, dx, dy].every((v) => v != null && Number.isFinite(v))) continue;
      distanceSum += Math.hypot(dx - ax, dy - ay);
      distanceCount += 1;

      const [rvx, rvy] = velocityAt(runner.x, runner.y, index, scene.fps, window);
      const [target0, target1] = goalSideTarget(
        ax + rvx * LOOKAHEAD_S, ay + rvy * LOOKAHEAD_S, goalX, goalY, GOAL_SIDE_OFFSET,
      );
      const [dvx, dvy] = velocityAt(defender.x, defender.y, index, scene.fps, window);
      const speed = Math.hypot(dvx, dvy);
      if (speed < minimumSpeed) continue;
      const tx = target0 - dx;
      const ty = target1 - dy;
      const norm = Math.hypot(tx, ty);
      if (speed <= 1e-6 || norm <= 1e-6) continue;
      alignSum += (dvx * tx + dvy * ty) / (speed * norm);
      alignCount += 1;
    }
    if (!distanceCount) continue;
    const meanDistance = distanceSum / distanceCount;
    const pursuit = alignCount ? alignSum / alignCount : 0;
    const proximity = Math.min(Math.max(1 - meanDistance / 25, 0), 1);
    const score = Math.min(Math.max(0.55 * Math.max(pursuit, 0) + 0.45 * proximity, 0), 1);
    rows.push({
      id: defender.id,
      shirt: defender.shirt,
      name: defender.name,
      score,
      meanDistance,
      pursuit,
      caption: `${meanDistance.toFixed(0)} m · pursuit ${pursuit.toFixed(2)}`,
    });
  }
  rows.sort((a, b) => b.score - a.score);
  return rows;
}

/** Teammates ordered by the space they gain from the defenders' movement. */
export function rankBeneficiaries(cache, scene, runnerId, defenderIds, freezeIndex, slot,
                                  baseline = "hold") {
  if (!defenderIds.length) return [];
  const swap = defenderIds.map((playerId) => ({ playerId, freezeIndex, mode: baseline }));
  const after = cache.indices.map((i) => i >= freezeIndex);
  const anyAfter = after.some(Boolean);
  const rows = [];
  for (const player of scene.attackers) {
    if (player.gk || player.id === runnerId) continue;
    const factual = cache.residualSeries(player.id);
    const counter = cache.residualSeries(player.id, swap);
    let sum = 0;
    let count = 0;
    for (let i = 0; i < factual.length; i += 1) {
      if (anyAfter && !after[i]) continue;
      sum += factual[i] - counter[i];
      count += 1;
    }
    const gain = count ? sum / count : 0;
    rows.push({
      id: player.id,
      shirt: player.shirt,
      name: player.name,
      gain,
      gainNow: factual[slot] - counter[slot],
      caption: `${gain >= 0 ? "+" : ""}${gain.toFixed(1)} avg`,
    });
  }
  rows.sort((a, b) => b.gain - a.gain);
  return rows;
}

/** Best triplet by space gain: the app's own suggestion, not a pipeline output. */
export function autoTriplet(cache, scene, runnerId = null, defenderPool = 3) {
  const runners = runnerId
    ? [runnerId]
    : scene.attackers.filter((p) => !p.gk).map((p) => p.id);
  let best = null;
  let bestGain = -Infinity;
  for (const candidate of runners) {
    const onset = scene.onsets[candidate];
    if (!onset) continue;
    const slot = cache.slotFor(onset.index);
    const defenders = rankDefenders(scene, candidate, onset.index).slice(0, defenderPool);
    for (const defender of defenders) {
      const gainers = rankBeneficiaries(cache, scene, candidate, [defender.id],
                                        onset.index, slot);
      if (!gainers.length) continue;
      if (gainers[0].gain > bestGain) {
        bestGain = gainers[0].gain;
        best = {
          runner: candidate,
          defender: defender.id,
          beneficiary: gainers[0].id,
          freezeIndex: onset.index,
          gain: gainers[0].gain,
        };
      }
    }
  }
  return best;
}
