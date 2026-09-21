// Scene loading and the view transform.
//
// Mirrors demo_viz.scene.Scene: metres, origin at the centre spot, and a view
// transform that mirrors the pitch so the attacking team always plays
// left-to-right on screen. All maths stays in raw coordinates.

export async function loadIndex(base = "data") {
  const response = await fetch(`${base}/index.json`);
  if (!response.ok) throw new Error(`index.json: ${response.status}`);
  return (await response.json()).scenes;
}

export async function loadScene(file, base = "data") {
  const response = await fetch(`${base}/${file}`);
  if (!response.ok) throw new Error(`${file}: ${response.status}`);
  const scene = await response.json();
  scene.byId = new Map(scene.players.map((p) => [p.id, p]));
  scene.flip = scene.attacking_direction < 0;
  scene.times = Array.from({ length: scene.n_frames }, (_, i) => scene.t0 + i / scene.fps);
  scene.attackers = scene.players.filter((p) => p.side === "attack");
  scene.defenders = scene.players.filter((p) => p.side === "defend");
  return scene;
}

/**
 * Raw pitch coordinates -> SVG coordinates.
 *
 * Two transforms at once: the mirror that makes the attacking team play
 * left-to-right (as in demo_viz.scene.Scene.view_xy), and the y negation that
 * turns a y-up pitch into y-down SVG space.
 */
export function view(scene, x, y) {
  if (x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y)) return null;
  return scene.flip ? [-x, y] : [x, -y];
}

export function playerAt(scene, player, index) {
  return view(scene, player.x[index], player.y[index]);
}

export function ballAt(scene, index) {
  return view(scene, scene.ball.x[index], scene.ball.y[index]);
}

export function label(scene, playerId) {
  const player = scene.byId.get(playerId);
  return player ? `#${player.shirt} ${player.name}` : "";
}

export function onsetFor(scene, runnerId) {
  const entry = runnerId ? scene.onsets[runnerId] : null;
  if (!entry) return { index: 0, method: "clip start" };
  return entry;
}
