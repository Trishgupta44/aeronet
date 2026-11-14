// Colours for the drawing code that cannot use CSS variables (pixel work on canvases). One object per theme, so a canvas reads
// `canvasPalette(currentTheme())` once per draw. The pressure colour map has its own dark version (see utils/colormap.js):
// in the dark theme the free-stream colour is the page's own deep navy, so the field sits in the page instead of glowing on it.

export const CANVAS = {
  light: {
    stage: '#f7f7f4', // plot background behind the colour map (equal to the map's value at Cp = 0)
    bodyFill: '#20242c',
    bodyStroke: '#0c0e12',
    halo: 'rgba(255,255,255,.85)',
    isobar: 'rgba(0,0,0,.28)',
    roadFill: '#d6d7d3',
    roadHatch: 'rgba(60,66,78,.55)',
    roadLine: '#20242c',
    roadLabel: '#3c424e',
    ink: '#0b1020', // lift arrow, pinned outline
    flow: '#20242c', // freestream arrow and label
    pen: [26, 15, 158],
    penHalo: 'rgba(255,255,255,.9)',
  },
  dark: {
    stage: '#0c1430',
    bodyFill: '#03071a',
    bodyStroke: 'rgba(214,224,247,.9)',
    halo: 'rgba(3,7,26,.8)',
    isobar: 'rgba(214,224,247,.26)',
    roadFill: '#121b3b',
    roadHatch: 'rgba(150,165,205,.22)',
    roadLine: '#9fb0d8',
    roadLabel: '#9fb0d8',
    ink: '#e6ebf7',
    flow: '#c8d3ee',
    pen: [169, 184, 255],
    penHalo: 'rgba(3,7,26,.85)',
  },
};

export const canvasPalette = (theme) => (theme === 'dark' ? CANVAS.dark : CANVAS.light);
