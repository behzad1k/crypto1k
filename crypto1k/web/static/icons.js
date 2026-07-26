/*
 * Phosphor icon helper for markup built in JavaScript.
 *
 * The sprite itself lives in templates/_icons.html, which every page includes;
 * this only emits <use> references into it. Keep the two in sync — asking for
 * a name that isn't in the sprite renders nothing rather than throwing.
 */

const PH_ICONS = new Set([
  'broadcast', 'brain', 'warning', 'chart-bar', 'lightning', 'fish', 'star',
  'check-circle', 'x-circle', 'target', 'trend-up', 'magnifying-glass',
  'arrow-circle-up', 'arrow-circle-down', 'minus-circle', 'currency-btc',
]);

function escapeAttr(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

/**
 * Inline <svg> string for a Phosphor icon.
 *
 * @param {string} name    sprite id without the "ph-" prefix
 * @param {string} [cls]   extra classes, e.g. 'ph-lg ph-pos'
 * @param {string} [title] accessible label; omit for purely decorative icons
 * @returns {string} HTML, safe to concatenate into a template literal
 */
function icon(name, cls = '', title = '') {
  if (!PH_ICONS.has(name)) {
    console.warn(`icon(): "${name}" is not in the sprite`);
    return '';
  }
  const classAttr = `class="${escapeAttr(['ph', cls].filter(Boolean).join(' '))}"`;
  const a11y = title
    ? `role="img" aria-label="${escapeAttr(title)}"`
    : 'aria-hidden="true"';
  const titleEl = title ? `<title>${escapeAttr(title)}</title>` : '';
  return `<svg ${classAttr} ${a11y}>${titleEl}<use href="#ph-${name}"/></svg>`;
}

/*
 * Alert summaries are authored once, in Python, and sent to three places:
 * Telegram, plain-text email, and this UI. Only the last can render SVG, so
 * the string keeps its emoji and the browser swaps them for Phosphor icons at
 * render time. Changing the Python to emit markup instead would put raw
 * <svg> tags into Telegram messages.
 *
 * Keep in sync with the emoji used in scanner.build_summary().
 */
const SUMMARY_ICONS = [
  ['\u{1F7E2}', 'arrow-circle-up',   'ph-pos',  'buy signal'],
  ['\u{1F534}', 'arrow-circle-down', 'ph-neg',  'sell signal'],
  ['\u{26AA}',  'minus-circle',      '',        'neutral'],
  ['⚡️', 'lightning',      '',        'fast path'],
  ['⚡',    'lightning',         '',        'fast path'],
  ['\u{1F9E0}', 'brain',             '',        'smart money'],
  ['\u{1F433}', 'fish',              '',        'whale flow'],
  ['⚠️', 'warning',        'ph-warn', 'warning'],
  ['⚠',    'warning',           'ph-warn', 'warning'],
  ['\u{1F3AF}', 'target',            '',        'exit plan'],
  ['₿',    'currency-btc',      'ph-btc',  'bitcoin'],
  ['\u{1F4CA}', 'chart-bar',         '',        'stats'],
];

/**
 * Escape a summary string and swap its emoji for Phosphor icons.
 * Returns HTML — assign with innerHTML, not textContent.
 *
 * @param {string} text summary as produced by scanner.build_summary()
 */
function iconifySummary(text) {
  if (!text) return '';
  let out = escapeAttr(text);
  for (const [emoji, name, cls, label] of SUMMARY_ICONS) {
    if (!out.includes(emoji)) continue;
    out = out.split(emoji).join(icon(name, cls, label));
  }
  return out;
}
