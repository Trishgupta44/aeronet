import { describe, expect, it } from 'vitest';
import { pageFromAddress } from './route.js';

describe('pageFromAddress: which page an address opens', () => {
  it('uses the #hash page first', () => {
    expect(pageFromAddress('#results', '')).toBe('results');
    expect(pageFromAddress('#pinn', '?pinn=naca0012_100x50_a4_m04_gs&round=3')).toBe('pinn');
    expect(pageFromAddress('#home', '?tab=draw')).toBe('home');
  });
  it('opens the PINN Lab for a ?pinn= deep link without a hash, and for a hash that is no page', () => {
    expect(pageFromAddress('', '?pinn=naca2412_100x50_a4_m03&round=2')).toBe('pinn');
    expect(pageFromAddress('#credit', '?pinn=naca0012_50x25_a4_m04')).toBe('pinn');
  });
  it('opens the tool named by ?tab=', () => {
    expect(pageFromAddress('', '?tab=draw&shape=bird')).toBe('draw');
    expect(pageFromAddress('', '?tab=explore&code=2412')).toBe('explore');
  });
  it('falls back to the home page; unrelated flags do not matter', () => {
    expect(pageFromAddress('', '')).toBe('home');
    expect(pageFromAddress(undefined, undefined)).toBe('home');
    expect(pageFromAddress('', '?present&light')).toBe('home');
    expect(pageFromAddress('', '?tab=bogus')).toBe('home');
  });
});
