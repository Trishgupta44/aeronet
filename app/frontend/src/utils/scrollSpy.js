/**
 * Which section is the reader in? `offsets` is [{id, top}] with `top` the page-relative top of each section.
 * Returns the id of the last section whose top is at or above `scrollY + lead` (null while above the first one).
 */
export function activeSection(offsets, scrollY, lead = 80) {
  let current = null;
  let best = -Infinity;
  for (const o of offsets) {
    if (o.top <= scrollY + lead && o.top > best) {
      best = o.top;
      current = o.id;
    }
  }
  return current;
}
