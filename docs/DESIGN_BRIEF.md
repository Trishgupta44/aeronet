# AeroNet design brief: "aerospace-corporate" look

*Reference studied on 2026-11-07: https://www.embraer.com/en (read with a browser: computed styles and screenshots).
We borrow the **style language** (typography feel, colour feel, layout rhythm). We do **not** copy their logo, name,
photographs, text, icons or the licensed GT America typefaces. The site must read as AeroNet, a sibling in the
aerospace space, not as an imitation.*

## 1. What the reference does (measured)

| Aspect | Reference (measured) | Our equivalent |
|---|---|---|
| Headlines | Tall **compressed sans**, UPPERCASE, weight 500, tight line height (70 px on 60 px lines in the hero, 35-45 px for section titles) | **Barlow Condensed** 500/600, uppercase, `line-height: .9` to `1` |
| Body | Light/regular grotesque, 16-18 px, black on white | **Inter** (or Archivo) 400, 16-17 px, `#0b1020` on white |
| Eyebrow / captions | Letter-spaced **mono-style** small caps line under headings ("A GLOBAL AEROSPACE COMPANY") | **IBM Plex Mono** 500, 12-13 px, uppercase, `letter-spacing: .12em` |
| Top strip | Slim utility links on the right (About, Careers, ...), language picker on the left | Slim strip: version, "Inviscid · subsonic · screening tool", GitHub/Docs link |
| Main nav | Logo left, 5 uppercase links (12-13 px), search icon right | Wordmark left, links: **EXPLORE · DRAW · PINN LAB · RESULTS · ABOUT** |
| Sticky bar | After scrolling, a **solid navy bar (47 px)** with the logo mark, the same links and "BACK TO TOP" | Same behaviour (navy `#0a1a5c`, 48 px) |
| Hero | Full-bleed image, dark overlay, giant white uppercase headline bottom-left | **Full-bleed live flow canvas** (animated particles over the airfoil) with a dark navy overlay, headline bottom-left |
| Buttons | Rectangles, **2 px radius**, navy fill `#100690`, white 14 px text, 40 px tall, wide padding; secondary slate `#1e3137` | Same shape: primary navy, secondary slate, outline variant |
| Links | Text + thin arrow ("Find out more →") | Same |
| Cards | Large photo cards in a 2-column grid, white uppercase compressed title overlaid, arrow link bottom right; a pale `#eff4f7` band behind the "more" cards | Cards for **Explore / Draw / PINN Lab / Results**, each with a live mini-picture (canvas thumbnail), on `#eff4f7` |
| Rhythm | Lots of white space, sections 500-1100 px tall, one idea per section, headline left / paragraph right on wide screens | Same: one idea per band, 12-column grid, 1180 px max width |
| Colour | White page; deep navy/indigo `#100690` for headings and buttons; navy bar `#0a1a5c`; pale blue-grey `#eff4f7` bands; near-black body | See tokens below |

## 2. Tokens (put in `app/frontend/src/styles.css` as CSS variables; keep the existing light/dark switch)

```
--navy-900:#0a1a5c;   /* sticky bar, hero overlay base */
--indigo-700:#1a0f9e; /* headings, primary buttons (our own shade, close to but not the reference's) */
--slate-800:#1e3137;  /* secondary buttons */
--sky-50:#eff4f7;     /* pale section band */
--ink:#0b1020;  --muted:#566070;  --line:#dfe5ec;  --paper:#ffffff;
--flow-suction:#1f4fd6; --flow-stagnation:#d4502a;   /* Cp colours stay as defined by the colour map */
--radius:2px;  --maxw:1180px;  --gutter:16px (mobile) / 32px (desktop)
```
Dark mode (the app already supports it): `--paper:#0a0f1f`, `--ink:#e8ecf4`, bands `#111936`, headings `#9db0ff`.
Contrast must stay at least 4.5:1 for body text.

## 3. Fonts (offline-safe)

The demo must work offline, so **do not link Google Fonts**. Install self-hosted packages and import them in
`main.jsx`: `@fontsource/barlow-condensed` (500, 600), `@fontsource/inter` (400, 500, 600) and
`@fontsource/ibm-plex-mono` (500). Set a fallback stack: `"Barlow Condensed", "Arial Narrow", Impact, sans-serif`
for headlines and `Inter, system-ui, sans-serif` for body. (This adds three small font packages: that is the only
new npm dependency allowed beyond Plan B's "none" rule.)

## 4. Page structure (maps to the tab layout in Plan B; one page, sections stack, nav scrolls/switches)

1. **Utility strip** (24 px, tiny mono text) and **main nav** (wordmark "AERONET" in compressed caps with a small
   chevron mark drawn in SVG by us: do not copy the reference's chevron logo; use a simple airfoil-section glyph).
2. **Hero** (min 70 vh): live flow canvas full-bleed behind a navy gradient. Headline, uppercase compressed,
   bottom-left: **"SEE THE AIR"**. One line under it in mono: "Physics-informed airfoil flow, on a laptop". Two
   buttons: **Try the explorer** (primary), **Draw your own shape** (outline).
3. **"We are AeroNet" band**: heading left (navy, 45 px), mono eyebrow "AN AIRFOIL FLOW EXPLORER", paragraph right,
   button **About the project**. Include 3 big numbers (from `results/summary.csv`, labelled measured): PINN Cp error
   vs FVM on the same mesh, speed, mesh size. Never hard-code a number without its source.
4. **Tool section "Explore"**: the existing explorer (controls left, `FlowCanvas` right) styled as a clean panel with
   hairline borders (`1px var(--line)`), 2 px radius, no heavy shadows. Tabs rendered as the uppercase nav links.
5. **"Draw" section**: the `DrawPad` with the shape library as a row of cards.
6. **"PINN Lab" section**: Plan C's lab (placeholder card now: "Trained cases appear here").
7. **Cards band on `--sky-50`**: four cards "Explore / Draw / PINN Lab / Results", each with a thumbnail, uppercase
   compressed title, "Find out more →".
8. **Evidence section "Results"**: the headline table in a plain, airy table (hairline rows, mono numbers) with
   measured / quoted / fitted labels as small mono tags.
9. **Footer** (navy): project name, credit line "Built on NNfoil by Cao, Song and Zhang (arXiv 2401.08705)", honest
   scope note, link to the report. No logos of other organisations.

## 5. Motion and behaviour

- The hero particles follow the real velocity field and pause on `prefers-reduced-motion` or when off-screen.
- Sticky bar appears after 64 px of scroll (fade, 150 ms). Smooth scroll to sections; `scroll-margin-top: 56px`.
- Buttons: 120 ms colour transition, visible focus ring (2 px `--indigo-700` + 2 px offset). Arrow links nudge 4 px right on hover.
- Cards: image zooms 1.03 on hover (disabled for reduced motion).
- Mobile: nav collapses to a hamburger with a full-width navy sheet; hero headline 44 px; one column.

## 6. Do / don't

- **Do** keep the data honest: every number has its label, the "idealized flow" note stays on cars.
- **Do** make the explorer feel like an instrument: lots of white, hairlines, mono numerals (tabular figures).
- **Don't** use the reference's logo, wordmark, photos, product names, copy text or exact colour `#100690`.
- **Don't** add heavy gradients, glassmorphism or neon: the look is restrained and corporate.
- **Don't** break the 30 fps particle target or the offline-first rule.

## 7. Acceptance (screenshots, desktop 1440 and phone 390)

Hero, "We are AeroNet" band, explorer, draw section, cards band, results table, footer, sticky bar. Check: no
horizontal scroll at 390 px, Lighthouse accessibility >= 90, all text contrast >= 4.5:1, fonts load offline
(disable the network and reload).
