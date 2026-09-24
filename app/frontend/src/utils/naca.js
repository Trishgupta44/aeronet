/**
 * Validate a custom NACA 4-digit code typed by the user.
 * Returns an error message, or '' when the input is empty or valid. Also returns `valid`.
 * (The backend validates again; these messages match its rules.)
 */
export function checkNacaCode(v) {
  if (!/^\d{4}$/.test(v)) return { valid: false, message: v.length ? 'Enter exactly 4 digits.' : '' };
  const m = +v[0];
  const p = +v[1];
  const t = +v.slice(2);
  if (t < 4) return { valid: false, message: 'Thickness (last two digits) must be at least 04.' };
  if (m > 0 && p < 1) return { valid: false, message: 'Camber position (2nd digit) must be 1-9 when camber is not 0.' };
  return { valid: true, message: '' };
}
