const $ = s => document.querySelector(s);
const FONT = 'Arial, Helvetica, sans-serif';
const esc = t => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;');
const attr = t => esc(t).replace(/"/g, '&quot;');
const clean = s => String(s).replace(/\s*research prototype\s*/gi, ' ').replace(/ {2,}/g, ' ');
const showErr = m => { $('#err').textContent = m; $('#err').hidden = false; };
const radio = name => document.querySelector(`input[name="${name}"]:checked`).value;
const TONES = {NORMAL: 'green', ELEVATED: 'amber', DANGER: 'red', EXTREME: 'purple'};
let STATIONS = [], EVENTS = [];

async function post(url, body) {
  const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || 'Request failed');
  return d;
}
async function busy(btn, fn) {
  const label = btn.textContent; btn.disabled = true; btn.textContent = 'Loading…'; $('#err').hidden = true;
  try { await fn(); } catch (e) { showErr(e.message); }
  btn.disabled = false; btn.textContent = label;
}
const fill = (sel, items, val) => { sel.innerHTML = items.map(i => `<option>${esc(i)}</option>`).join(''); if (val) sel.value = val; };

// searchable station box: clears on focus so the full list shows; restores if left empty
function combo(el) {
  let prev = '';
  el.addEventListener('focus', () => { prev = el.value; el.value = ''; });
  el.addEventListener('blur', () => { if (!el.value) el.value = prev; });
}

function draw(el, f) {
  if (f.type === 'plotly') {
    if (typeof Plotly === 'undefined') { el.textContent = 'Chart library failed to load.'; return; }
    const p = JSON.parse(f.json);
    p.layout = p.layout || {};
    p.layout.font = Object.assign({}, p.layout.font, {family: FONT});
    Plotly.newPlot(el, p.data, p.layout, {responsive: true, displayModeBar: false});
  } else el.innerHTML = `<img alt="chart" src="data:image/png;base64,${f.data}">`;
}
function cell(c) {
  const k = String(c).trim().toUpperCase();
  return TONES[k] ? `<span class="badge ${TONES[k]}">${k}</span>` : esc(c);
}
function table(el, t) {
  el.innerHTML = '<thead><tr>' + t.columns.map(c => `<th>${esc(c)}</th>`).join('') + '</tr></thead><tbody>' +
    t.rows.map(r => '<tr>' + r.map(c => `<td>${cell(c)}</td>`).join('') + '</tr>').join('') + '</tbody>';
}
function md(s) {
  const inline = t => esc(t).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/(^|\s)_(.+?)_(?=\s|$)/g, '$1<i>$2</i>')
    .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s.,;:)]|$)/g, '$1<i>$2</i>');
  let out = '', rows = [];
  const flush = () => { if (!rows.length) return;
    out += '<table>' + rows.map((r, i) => '<tr>' + r.map(c => `<${i ? 'td' : 'th'}>${inline(c)}</${i ? 'td' : 'th'}>`).join('') + '</tr>').join('') + '</table>'; rows = []; };
  for (const line of s.split('\n')) {
    if (line.trim().startsWith('|')) { if (!/^\|[\s|:-]+\|?$/.test(line.trim())) rows.push(line.trim().replace(/^\||\|$/g, '').split('|').map(x => x.trim())); continue; }
    flush();
    if (/^#{2,3}\s/.test(line)) out += `<h3>${inline(line.replace(/^#+\s*/, ''))}</h3>`;
    else if (/^<small>.*<\/small>$/.test(line.trim())) out += `<p class="note">${inline(line.trim().slice(7, -8))}</p>`;
    else if (line.trim()) out += `<p>${inline(line)}</p>`;
  }
  flush(); return out;
}
function download(link, name, text) {
  link.onclick = e => { e.preventDefault(); const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], {type: 'text/plain'})); a.download = name; a.click(); };
}
const tone = n => {
  n = n.toUpperCase();
  if (/EXTREME/.test(n)) return 'purple';
  if (/NORMAL|LOW|SAFE/.test(n)) return 'green';
  if (/ELEVATED|WATCH|MODERATE/.test(n)) return 'amber';
  if (/SEVERE|MAJOR|DANGER|HIGH|WARNING|FLOOD/.test(n)) return 'red';
  return 'gray';
};

// tabs
function showTab() {
  const id = (location.hash || '#safe').slice(1);
  document.querySelectorAll('main section').forEach(s => s.hidden = s.id !== id);
  document.querySelectorAll('#tabs a').forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + id));
  window.scrollTo(0, 0);
}
window.addEventListener('hashchange', showTab);
showTab();

// quick-pick floods fill station + date
function presetOptions(sel) {
  sel.innerHTML = '<option value="">Choose a known flood…</option>' + EVENTS.map((e, i) => `<option value="${i}">${esc(e.label)}</option>`).join('');
}

// Am I safe?
function syncMode() {
  const r = radio('mode') === 'replay';
  document.querySelectorAll('.rep').forEach(x => x.hidden = !r);
  $('#stationCount').hidden = r;
}
document.querySelectorAll('input[name="mode"]').forEach(r => r.onchange = syncMode);
$('#preset').onchange = () => {
  const e = EVENTS[$('#preset').value]; if (!e) return;
  $('#river').value = e.station; $('#asof').value = e.date;
};
async function check() {
  const mode = radio('mode');
  if (!STATIONS.includes($('#river').value)) throw new Error('Pick a station from the list: type a name or ID, then choose a match.');
  if (mode === 'replay' && !$('#asof').value) throw new Error('Choose a replay date.');
  const d = await post('/api/check', {mode, river: $('#river').value, date: $('#asof').value});
  $('#safeOut').hidden = false; $('#card').innerHTML = clean(d.card); draw($('#chart'), d.fig);
  $('#alert').value = clean(d.alert);
}
$('#checkBtn').onclick = e => busy(e.target, check);
$('#copyBtn').onclick = async e => {
  try { await navigator.clipboard.writeText($('#alert').value); }
  catch { $('#alert').select(); document.execCommand('copy'); }
  e.target.textContent = 'Copied'; setTimeout(() => e.target.textContent = 'Copy', 1500);
};

// Officer dashboard
function syncOMode() { document.querySelectorAll('.orep').forEach(x => x.hidden = radio('o_mode') !== 'replay'); }
document.querySelectorAll('input[name="o_mode"]').forEach(r => r.onchange = syncOMode);
$('#o_preset').onchange = () => { const e = EVENTS[$('#o_preset').value]; if (e) $('#o_asof').value = e.date; };
$('#window').oninput = () => $('#winVal').textContent = $('#window').value;

function stats(t) {
  const i = t.columns.findIndex(c => /peak risk/i.test(c));
  if (i < 0) { $('#stats').innerHTML = ''; return; }
  const n = {}; t.rows.forEach(r => { const k = String(r[i]).trim().toUpperCase(); n[k] = (n[k] || 0) + 1; });
  const tiles = [['Stations', t.rows.length, 'gray']].concat(['EXTREME', 'DANGER', 'ELEVATED', 'NORMAL'].map(k => [k[0] + k.slice(1).toLowerCase(), n[k] || 0, TONES[k]]));
  $('#stats').innerHTML = tiles.map(([l, v, c]) => `<div class="stat"><b>${v}</b><span><i class="badge ${c}" style="padding:1px 8px">${l}</i></span></div>`).join('');
}
$('#refreshBtn').onclick = e => busy(e.target, async () => {
  const mode = radio('o_mode');
  if (mode === 'replay' && !$('#o_asof').value) throw new Error('Choose a replay date.');
  const d = await post('/api/dashboard', {mode, date: $('#o_asof').value, window: +$('#window').value});
  $('#o_out').hidden = false; $('#o_note').textContent = clean(d.note.replace(/[*_`#]/g, ''));
  draw($('#map'), d.map); table($('#rank'), d.table); stats(d.table); fill($('#pick'), d.stations);
  if (d.stations.length) await detail();
});
$('#rank').addEventListener('click', ev => {
  const tr = ev.target.closest('tbody tr'); if (!tr) return;
  const m = tr.firstElementChild.textContent.match(/\(([^()]+)\)\s*$/); if (!m) return;
  const opt = [...$('#pick').options].find(o => o.text.includes('(' + m[1] + ')'));
  if (opt) { $('#pick').value = opt.value; busy($('#detailBtn'), detail); $('#d_chart').scrollIntoView({behavior: 'smooth'}); }
});
async function detail() {
  const d = await post('/api/detail', {station: $('#pick').value, extra: +$('#extra').value || 0, day: +$('#day').value || 1});
  draw($('#d_chart'), d.fig); $('#d_card').innerHTML = clean(d.card);
  $('#capLink').hidden = false; download($('#capLink'), 'cap_alert.xml', d.cap);
}
$('#detailBtn').onclick = e => busy(e.target, detail);
$('#pick').onchange = () => busy($('#detailBtn'), detail);

// Batch
const FILE_IDS = ['f_test', 'f_train', 'f_answers'];
const removeBtn = id => document.querySelector(`button.remove[data-for="${id}"]`);
function clearFile(id) { $('#' + id).value = ''; removeBtn(id).hidden = true; }
FILE_IDS.forEach(id => {
  $('#' + id).addEventListener('change', () => { removeBtn(id).hidden = !$('#' + id).files.length; });
  removeBtn(id).onclick = () => clearFile(id);
});
$('#clearFiles').onclick = () => { FILE_IDS.forEach(clearFile); $('#f_robust').checked = false; $('#b_out').hidden = true; };
$('#batchBtn').onclick = e => busy(e.target, async () => {
  if (!$('#f_test').files[0]) throw new Error('Upload test.csv first.');
  const fd = new FormData(); fd.append('test', $('#f_test').files[0]);
  if ($('#f_train').files[0]) fd.append('train', $('#f_train').files[0]);
  if ($('#f_answers').files[0]) fd.append('answers', $('#f_answers').files[0]);
  fd.append('robustness', $('#f_robust').checked ? 'true' : 'false');
  const r = await fetch('/api/batch', {method: 'POST', body: fd}); const d = await r.json();
  if (!r.ok) throw new Error(d.detail || 'Request failed');
  $('#b_out').hidden = false; $('#b_score').innerHTML = md(d.score);
  table($('#b_sum'), d.summary); table($('#b_prev'), d.preview);
  $('#b_metrics').innerHTML = md(d.metrics);
  download($('#dlPred'), 'predictions.csv', d.predictions); download($('#dlRisk'), 'predictions_with_risk.csv', d.risk);
});

// Manual input
const FIELDS = [['sid','Station ID','text',''],['day','Date (YYYY-MM-DD)','text','2024-05-15'],['rain0','Rain today (mm)',20],
  ['rain7','Rain, last 7 days (mm)',60],['rain30','Rain, last 30 days (mm)',120],['tmax','Max temp (°C)',25],['tmin','Min temp (°C)',14],
  ['soil','Soil moisture (m³/m³ or %)',0.32],['area','Drainage area (sq mi)'],['elev','Mean elevation (ft)'],['slope','Slope (%)'],['forest','Forest cover (%)']];
$('#m_fields').innerHTML = FIELDS.map(([id, label, type, v]) =>
  `<label>${label}<input id="m_${id}" ${type === 'text' ? 'type="text"' : 'type="number" step="any"'} value="${type === 'text' ? v : (type ?? '')}"></label>`).join('');
async function prefill() {
  const v = await (await fetch('/api/station?label=' + encodeURIComponent($('#m_pick').value))).json();
  ['sid', 'area', 'elev', 'slope', 'forest'].forEach((k, i) => {
    let x = v[i]; if (typeof x === 'number') x = Math.round(x * 100) / 100;
    $('#m_' + k).value = x ?? '';
  });
}
$('#m_pick').onchange = () => { if (STATIONS.includes($('#m_pick').value)) prefill().catch(e => showErr(e.message)); };

function manualCard(text) {
  const lines = text.split('\n').map(s => s.trim()).filter(Boolean);
  const m = (lines[0] || '').match(/^#+\s*(.+?)\s+[—-]\s+([\d,.]+)\s*cfs\s*\(([\d,.]+)\s*m³\/s\)/);
  if (!m) return md(text);
  const note = (lines[2] || '').replace(/^_|_$/g, '');
  return `<div class="card result">
    <div><span class="badge ${tone(m[1])}">${esc(m[1])}</span>
      <div class="big">${esc(m[2])}<small>cfs</small></div>
      <div class="alt">${esc(m[3])} m³/s</div></div>
    <div class="text"><p>${esc(lines[1] || '')}</p><p class="note">${esc(note)}</p></div></div>`;
}
$('#manualBtn').onclick = e => busy(e.target, async () => {
  const n = id => { const v = $('#m_' + id).value; return v === '' ? null : +v; };
  const d = await post('/api/manual', {sid: $('#m_sid').value, day: $('#m_day').value, rain0: n('rain0'), rain7: n('rain7'),
    rain30: n('rain30'), tmax: n('tmax'), tmin: n('tmin'), soil: n('soil'), area: n('area'), elev: n('elev'), slope: n('slope'), forest: n('forest')});
  $('#m_out').innerHTML = manualCard(d.md);
});

// init
(async () => {
  try {
    const meta = await (await fetch('/api/meta')).json();
    STATIONS = meta.stations; EVENTS = meta.events;
    $('#stationList').innerHTML = STATIONS.map(s => `<option value="${attr(s)}">`).join('');
    $('#river').value = meta.default_river; $('#m_pick').value = meta.default_river;
    combo($('#river')); combo($('#m_pick'));
    $('#stationCount').textContent = `${STATIONS.length} stations available. Click the box and type to search by name or ID.`;
    presetOptions($('#preset')); presetOptions($('#o_preset'));
    $('#asof').max = meta.date_max; $('#o_asof').max = meta.date_max;
    if (EVENTS[0]) { $('#asof').value = EVENTS[0].date; $('#o_asof').value = EVENTS[0].date; }
    syncMode(); syncOMode();
    await prefill();
    busy($('#checkBtn'), check);
  } catch (e) { showErr('Could not load data from the server: ' + e.message); }
})();