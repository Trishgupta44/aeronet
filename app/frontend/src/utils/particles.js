import { sampleVelocity } from './field.js';

/** Small seeded random generator (returns numbers in [0, 1)); same seed, same sequence. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const INFLOW_FRACTION = 0.04; // particles re-enter in the left 4 % of the window
const LIFE_MIN = 80;
const LIFE_SPAN = 120; // life is 80 to 200 frames

/**
 * Put a particle at a random free spot: in the inflow strip (`fromLeft`) or anywhere in the window.
 * A particle is {x, y, px, py, age, life}; (px, py) is where it was one frame ago (for the trail).
 */
export function respawn(p, f, rand, fromLeft) {
  const w = f.xmax - f.xmin;
  const h = f.ymax - f.ymin;
  let x = f.xmin;
  let y = f.ymin;
  for (let tries = 0; tries < 30; tries++) {
    x = f.xmin + rand() * (fromLeft ? INFLOW_FRACTION : 1) * w;
    y = f.ymin + rand() * h;
    if (sampleVelocity(f, x, y)) break;
  }
  p.x = x;
  p.y = y;
  p.px = x;
  p.py = y;
  p.age = 0;
  p.life = LIFE_MIN + rand() * LIFE_SPAN;
  return p;
}

/** n particles spread over the free part of the window, with random ages so they do not all die together. */
export function createParticles(n, f, rand) {
  const ps = [];
  for (let k = 0; k < n; k++) {
    const p = respawn({}, f, rand, false);
    p.age = Math.floor(rand() * p.life);
    ps.push(p);
  }
  return ps;
}

/**
 * Advance every particle one frame with a midpoint (RK2) step of length dt in world units per unit speed.
 * Leaving the window or touching the body sends a particle back to the inflow edge; old age respawns it anywhere.
 */
export function stepParticles(ps, f, dt, rand) {
  for (const p of ps) {
    const v1 = sampleVelocity(f, p.x, p.y);
    if (!v1) {
      respawn(p, f, rand, true);
      continue;
    }
    const v2 = sampleVelocity(f, p.x + 0.5 * dt * v1[0], p.y + 0.5 * dt * v1[1]);
    if (!v2) {
      respawn(p, f, rand, true);
      continue;
    }
    p.px = p.x;
    p.py = p.y;
    p.x += dt * v2[0];
    p.y += dt * v2[1];
    p.age += 1;
    if (p.age >= p.life) respawn(p, f, rand, false);
    else if (!sampleVelocity(f, p.x, p.y)) respawn(p, f, rand, true);
  }
}
