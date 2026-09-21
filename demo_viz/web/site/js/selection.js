// Role selection, mirroring demo_viz.core.selection.Selection.

export const ROLES = ["runner", "defender", "beneficiary"];
const FIELD = { runner: "runners", defender: "defenders", beneficiary: "beneficiaries" };
const SIDE = { runner: "attack", beneficiary: "attack", defender: "defend" };

export class Selection {
  constructor(data = {}) {
    this.runners = [...(data.runners || [])];
    this.defenders = [...(data.defenders || [])];
    this.beneficiaries = [...(data.beneficiaries || [])];
    this.pick = data.pick || "runner";
    this.armed = Boolean(data.armed);
    this.source = data.source || "manual";
  }

  ids(role) { return this[FIELD[role]]; }

  roleOf(playerId) {
    for (const role of ROLES) if (this.ids(role).includes(playerId)) return role;
    return null;
  }

  get runner() { return this.runners[0] || null; }

  get isEmpty() {
    return !this.runners.length && !this.defenders.length && !this.beneficiaries.length;
  }

  toggle(role, playerId, single = false) {
    const bucket = this.ids(role);
    const at = bucket.indexOf(playerId);
    if (at >= 0) { bucket.splice(at, 1); return; }
    for (const other of ROLES) {
      if (other === role) continue;
      const index = this.ids(other).indexOf(playerId);
      if (index >= 0) this.ids(other).splice(index, 1);
    }
    if (single) bucket.length = 0;
    bucket.push(playerId);
  }

  clear(role = null) {
    for (const name of role ? [role] : ROLES) this.ids(name).length = 0;
  }

  advance() {
    if (this.pick === "runner" && this.runners.length) this.pick = "defender";
    else if (this.pick === "defender" && this.defenders.length) this.pick = "beneficiary";
  }

  arm(role) {
    if (ROLES.includes(role)) { this.pick = role; this.armed = true; }
    return this;
  }

  /**
   * Assign a clicked player to a role.
   *
   * Every role holds a list, because the annotations contain scenes with
   * several runners, several defenders and several beneficiaries. A click
   * toggles membership; it never replaces the whole role.
   *
   * 1. an armed slot toggles the clicked player in or out of it, if the
   *    player is on the right side;
   * 2. otherwise a player who already holds a role loses it;
   * 3. otherwise the clicked side decides, and an attacker joins Runner while
   *    it is empty, else Beneficiary.
   */
  applyClick(scene, playerId) {
    const player = scene.byId.get(playerId);
    if (!player) return this;

    if (this.armed && SIDE[this.pick] === player.side) {
      this.toggle(this.pick, playerId);
      this.source = "manual";
      return this;
    }

    const existing = this.roleOf(playerId);
    let role;
    if (existing) role = existing;
    else if (player.side === "defend") role = "defender";
    else if (this.pick === "runner" || !this.runners.length) role = "runner";
    else role = "beneficiary";

    this.toggle(role, playerId);
    this.armed = false;
    this.source = "manual";
    this.pick = role;
    this.advance();
    return this;
  }

  /**
   * Put one player in one role, leaving the rest of that role alone.
   * This is what a drag means: dragging one runner chip onto Beneficiary
   * moves that player and nobody else.
   */
  move(scene, role, playerId) {
    const player = scene.byId.get(playerId);
    if (!player || SIDE[role] !== player.side) return this;
    if (!this.ids(role).includes(playerId)) this.toggle(role, playerId);
    this.armed = false;
    this.source = "manual";
    this.pick = role;
    return this;
  }

  /** Exchange the whole Runner and Beneficiary lists (deliberate "move all"). */
  swapAttackRoles() {
    const runners = this.runners;
    this.runners = this.beneficiaries;
    this.beneficiaries = runners;
    this.armed = false;
    this.source = "manual";
    return this;
  }

  static fromRoles(roles, source = "annotation") {
    return new Selection({
      runners: roles.runner || [],
      defenders: roles.defender || [],
      beneficiaries: roles.beneficiary || [],
      source,
    });
  }
}
