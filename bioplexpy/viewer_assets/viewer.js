/* BioPlexPy structure viewer: displays what bioplexpy/viewer.py computed
   (job.js, model_<n>.js). Nothing about contacts is decided here except
   the two sliders, which only threshold values from the tables.

   Only the object Mol below touches the 3D viewer (Mol*). */
(function () {
'use strict';
const BPV = window.BPV = window.BPV || {models: {}};
const job = BPV.job;
const $ = id => document.getElementById(id);
if (!job) { document.body.textContent = 'job.js was not found next to this page.'; return; }

const N = job.models.length;
const nodeById = Object.fromEntries(job.nodes.map(n => [n.id, n]));
const chainById = Object.fromEntries(job.chains.map(c => [c.id, c]));
const key = (a, b) => a < b ? a + '|' + b : b + '|' + a;
// the contact rule in use: job.rules[state.rule] (pairs), and its contacts in every model
let rule = job.rules[0], pairByKey = {};
const SCORE_LABELS = {ipsae_calc: 'ipSAE', pair_iptm: 'pair ipTM', ipsae: 'ipSAE (ColabFold)',
  pdockq_calc: 'pDockQ', pdockq2_calc: 'pDockQ2', pdockq: 'pDockQ (ColabFold)',
  pdockq2: 'pDockQ2 (ColabFold)', lis: 'LIS', clis: 'cLIS', ilis: 'iLIS'};

const scoreName = job.score_names.includes(job.filter.score) ? job.filter.score
  : (job.score_names.includes('pair_iptm') ? 'pair_iptm' : job.score_names[0]);
const state = {
  model: 0, rule: 0, color: 'chain', minK: 1,
  scoreOn: job.filter.min_score != null && scoreName === job.filter.score,
  cut: job.filter.min_score != null ? job.filter.min_score : (job.filter.suggested != null ? job.filter.suggested : 0.3),
  sel: null,          // {a, b}: the selected protein pair
  cell: null,         // [row, column] of a clicked PAE cell
  hoverNode: null, hoverChain: null, hoverResidue: null,
  R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
};

/* ---------- contacts across models ---------- */
// every contact seen in any model, protein or not: key -> {a, b, contact: [per model]}
let edges = {};
function useRule(index) {
  state.rule = index; rule = job.rules[index];
  pairByKey = Object.fromEntries(rule.pairs.map(p => [key(p.a, p.b), p]));
  edges = {};
  job.models.forEach((m, i) => m.edges[rule.id].forEach(([a, b]) => {
    const e = edges[key(a, b)] = edges[key(a, b)] || {a: a < b ? a : b, b: a < b ? b : a,
      contact: new Array(N).fill(false)};
    e.contact[i] = true;
  }));
}
useRule(0);
function pairScore(pair, m) {
  const s = pair && scoreName && pair.scores[scoreName];
  if (!s) return null;
  const v = s[m].filter(x => x != null);
  if (!v.length) return null;
  const how = job.filter.reduce || 'max';
  return how === 'min' ? Math.min(...v) : how === 'mean' ? v.reduce((x, y) => x + y, 0) / v.length : Math.max(...v);
}
// a contact with no score is kept, as in the tables
function passes(k, m) {
  if (!state.scoreOn) return true;
  const v = pairScore(pairByKey[k], m);
  return v == null || v >= state.cut;
}
const inModel = (k, m) => !!edges[k] && edges[k].contact[m] && passes(k, m);
const count = k => { let n = 0; for (let m = 0; m < N; m++) if (inModel(k, m)) n++; return n; };
// in how many models the pair is a contact at all, whatever its score
const contacts = k => edges[k] ? edges[k].contact.filter(Boolean).length : 0;
// as the command line prints it: 'contact in 5 of 5 (2 pass the score filter)'
const countText = k => contacts(k) + ' of ' + N + (state.scoreOn ? ' (' + count(k) + ' pass)' : '');
const shown = k => count(k) >= state.minK;

/* ---------- loading a model's data file on first use ---------- */
const pending = {};
function modelData(i) {
  if (BPV.models[i]) return Promise.resolve(BPV.models[i]);
  if (!pending[i]) pending[i] = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = 'model_' + i + '.js';
    s.onload = () => BPV.models[i] ? resolve(prepare(BPV.models[i])) : reject(new Error(s.src + ' has no data'));
    s.onerror = () => reject(new Error(s.src + ' could not be loaded'));
    document.head.appendChild(s);
  });
  return pending[i];
}
function prepare(m) {
  // PAE row -> residue, and residue -> row
  m.rowInfo = new Array(m.pae_rows);
  m.rowOf = {};
  for (const c in m.rows) {
    m.rowOf[c] = {};
    m.rows[c].forEach((row, k) => { m.rowInfo[row] = {chain: c, resi: m.residues[c][k]}; m.rowOf[c][m.residues[c][k]] = row; });
  }
  m.plddtOf = {};
  for (const c in m.plddt) { m.plddtOf[c] = {}; m.plddt[c].forEach((v, k) => { m.plddtOf[c][m.residues[c][k]] = v; }); }
  if (m.pae) {
    const raw = atob(m.pae.data), bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    m.pae.bytes = bytes;
  }
  return m;
}

/* ---------- the 3D viewer (Mol*) ---------- */
const PLDDT_COLORS = [[90, 0x0053d6], [70, 0x65cbf3], [50, 0xffdb13], [-Infinity, 0xff7d45]];  // Mol*'s own
const GREY = 0xb3b3b3;
const Mol = {
  ready: false, structures: {}, onRotate: null, onResidue: null,
  async init(element) {
    if (!window.molstar) throw new Error('Mol* could not be loaded (it comes from cdn.jsdelivr.net, so the page needs a connection). The networks, the heatmap and the table work without it.');
    const viewer = this.viewer = await molstar.Viewer.create(element, {
      layoutIsExpanded: false, layoutShowControls: false, layoutShowSequence: false, layoutShowLog: false,
      layoutShowLeftPanel: false, layoutShowRemoteState: false, viewportShowExpand: false,
      viewportShowAnimation: false, viewportShowSelectionMode: false, viewportShowTrajectoryControls: false,
      volumeStreamingDisabled: true});
    const plugin = this.plugin = viewer.plugin;
    this.S = molstar.lib.structure;
    // Mol*'s "illustrative" look: outlines and ambient occlusion on the canvas, and (in load())
    // representations that ignore the light
    plugin.canvas3d.setProps({camera: {mode: 'orthographic', manualReset: true},
      postprocessing: {outline: {name: 'on', params: {scale: 1, color: 0x000000, threshold: 0.33, includeTransparent: true}},
                       occlusion: {name: 'on', params: Object.assign({}, (plugin.canvas3d.props.postprocessing.occlusion.params || {}),
                         {samples: 32, radius: 5, bias: 0.8, blurKernelSize: 15})}}});
    const hex = Object.fromEntries(job.chains.map(c => [c.id, parseInt(c.color.slice(1), 16)]));
    this.addTheme('bpv-chain', (chain) => hex[chain] != null ? hex[chain] : GREY);
    const camera = plugin.canvas3d.camera;
    const emit = () => { const v = camera.view;
      if (this.onRotate) this.onRotate([[v[0], v[4], v[8]], [v[1], v[5], v[9]], [v[2], v[6], v[10]]]); };
    (camera.changed || camera.stateChanged).subscribe(emit);
    const residueOf = loci => {
      const L = this.S.StructureElement.Loci, P = this.S.StructureProperties;
      if (!L.is(loci)) return null;
      let first = null;
      L.forEachLocation(loci, loc => { if (!first) first = {chain: P.chain.auth_asym_id(loc), resi: P.residue.auth_seq_id(loc)}; });
      return first;
    };
    viewer.subscribe(plugin.behaviors.interaction.hover, e => this.onResidue && this.onResidue('hover', residueOf(e.current.loci)));
    viewer.subscribe(plugin.behaviors.interaction.click, e => { const r = residueOf(e.current.loci); if (r && this.onResidue) this.onResidue('click', r); });
    // the panel is sized by the page's layout, which can change after Mol* starts
    if (window.ResizeObserver) new ResizeObserver(() => plugin.canvas3d && plugin.canvas3d.handleResize()).observe(element);
    this.ready = true;
    emit();
  },
  // a color theme from a function of (chain ID, residue number)
  addTheme(name, colorOf) {
    const S = this.S, P = S.StructureProperties;
    const provider = {name, label: name, category: 'BioPlexPy',
      factory: (ctx, props) => ({factory: provider.factory, granularity: 'group', props, description: '',
        color: loc => S.StructureElement.Location.is(loc) ? colorOf(P.chain.auth_asym_id(loc), P.residue.auth_seq_id(loc)) : GREY}),
      getParams: () => ({}), defaultValues: {}, isApplicable: () => true};
    this.plugin.representation.structure.themes.colorThemeRegistry.add(provider);
  },
  async load(i, m) {
    if (this.structures[i]) return;
    const H = this.plugin.managers.structure.hierarchy;
    const before = new Set(H.current.structures.map(s => s.cell.transform.ref));
    // cartoon for the chains whatever the size of the complex (Mol*'s own preset changes with
    // size), ball-and-stick for ligands and ions
    const B = this.plugin.builders;
    const data = await B.data.rawData({data: m.structure, label: m.name});
    const trajectory = await B.structure.parseTrajectory(data, m.format === 'cif' ? 'mmcif' : 'pdb');
    const structure = await B.structure.createStructure(await B.structure.createModel(trajectory));
    for (const [part, type] of [['polymer', 'cartoon'], ['ligand', 'ball-and-stick'], ['ion', 'ball-and-stick']]) {
      const component = await B.structure.tryCreateComponentStatic(structure, part);
      if (component) await B.structure.representation.addRepresentation(component, {type, typeParams: {ignoreLight: true}, color: 'bpv-chain'});
    }
    this.structures[i] = H.current.structures.find(s => !before.has(s.cell.transform.ref));
    this.addTheme('bpv-plddt-' + i, (chain, resi) => {
      const v = m.plddtOf[chain] && m.plddtOf[chain][resi];
      return v == null ? GREY : PLDDT_COLORS.find(([floor]) => v > floor)[1];
    });
    await this.color(i, state.color);
    if (Object.keys(this.structures).length === 1) { this.plugin.canvas3d.handleResize(); this.plugin.managers.camera.reset(); }
  },
  async color(i, mode) {
    const s = this.structures[i];
    if (s) await this.plugin.managers.structure.component.updateRepresentationsTheme(
      s.components, {color: mode === 'plddt' ? 'bpv-plddt-' + i : 'bpv-chain'});
  },
  show(i) {
    const H = this.plugin.managers.structure.hierarchy;
    for (const j in this.structures) H.toggleVisibility([this.structures[j]], +j === i ? 'show' : 'hide');
  },
  // residues: [{chain, resi: [numbers]}], of model i
  loci(i, residues) {
    const chains = [], numbers = [];
    residues.forEach(r => r.resi.forEach(n => { chains.push(r.chain); numbers.push(n); }));
    return this.S.StructureElement.Loci.fromSchema(this.structures[i].cell.obj.data,
      {items: {auth_asym_id: chains, auth_seq_id: numbers}});
  },
  select(i, residues, zoom) {
    if (!this.ready || !this.structures[i]) return;
    const I = this.plugin.managers.interactivity;
    I.lociSelects.deselectAll();
    this.plugin.managers.structure.focus.clear();
    if (!residues || !residues.length) return;
    const loci = this.loci(i, residues);
    I.lociSelects.select({loci});
    if (zoom) this.plugin.managers.camera.focusLoci(loci, {extraRadius: 14});
  },
  highlightChains(i, chains) {
    if (!this.ready || !this.structures[i]) return;
    const I = this.plugin.managers.interactivity;
    if (!chains || !chains.length) { I.lociHighlights.clearHighlights(); return; }
    I.lociHighlights.highlightOnly({loci: this.S.StructureElement.Loci.fromSchema(
      this.structures[i].cell.obj.data, {items: {auth_asym_id: chains}})});
  },
};

/* ---------- network panels ---------- */
const centre = {}, scaleRadius = (() => {
  let worst = 1;
  job.models.forEach((m, i) => {
    const pts = Object.values(m.centroids);
    const c = centre[i] = [0, 1, 2].map(d => pts.reduce((s, p) => s + p[d], 0) / (pts.length || 1));
    pts.forEach(p => { worst = Math.max(worst, Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2])); });
  });
  return worst;
})();
function nodeFill(panel, n) {
  if (n.type !== 'protein') return [panel === 'model' ? '#ffffff' : '#ffffff', 1];
  if (panel === 'model') return [n.color, 1];
  if (panel === '293t') return [n.bait293 ? '#15b01a' : n.prey293 ? '#c7fdb5' : '#b2b2b2', 1];
  if (n.bait293 && n.baitHct) return ['#929591', 1];
  if (n.bait293) return ['#e50000', 1];
  if (n.baitHct) return ['#0343df', 1];
  if (n.prey293 && n.preyHct) return ['#929591', 0.4];
  if (n.prey293) return ['#e50000', 0.4];
  if (n.preyHct) return ['#0343df', 0.4];
  return ['#d9d9d9', 1];
}
function panelEdges(panel) {
  const out = [];
  if (panel === 'all') {
    job.bioplex_edges.forEach(e => out.push({a: e.a, b: e.b, k: key(e.a, e.b),
      color: e.bp293 && e.bpHct ? '#929591' : e.bp293 ? '#e50000' : '#0343df',
      width: shown(key(e.a, e.b)) ? 3.5 : 1.2, alpha: 1, dashed: false}));
    return out;
  }
  for (const k in edges) {
    const n = count(k);
    if (n < state.minK || n === 0) continue;
    const e = edges[k], here = inModel(k, state.model), pair = pairByKey[k];
    const dashed = nodeById[e.a].type !== 'protein' || nodeById[e.b].type !== 'protein';
    if (panel === 'model') out.push({a: e.a, b: e.b, k, dashed, color: here ? '#4c4c4c' : '#c8c8c8',
      width: 1.2 + 4.8 * n / N, alpha: 1});
    else out.push({a: e.a, b: e.b, k, dashed, color: pair && pair.bp293 ? '#15b01a' : '#929591',
      width: 3, alpha: here ? 1 : 0.3});
  }
  return out;
}
const panels = [['model', 'net-model'], ['293t', 'net-293t'], ['all', 'net-all']].map(([kind, id]) => ({kind, canvas: $(id), pos: {}, edges: []}));
function drawPanel(p) {
  const canvas = p.canvas, dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const radius = 13, R = state.R, c = centre[state.model], cents = job.models[state.model].centroids;
  // room at the sides for the labels
  const s = Math.max(10, Math.min(w / 2 - radius - 54, h / 2 - radius - 18)) / scaleRadius;
  const nodes = job.nodes.filter(n => cents[n.id] && (p.kind !== 'all' || n.type === 'protein'));
  p.pos = {};
  nodes.forEach(n => { const q = cents[n.id].map((v, d) => v - c[d]);
    p.pos[n.id] = {x: w / 2 + s * (R[0][0] * q[0] + R[0][1] * q[1] + R[0][2] * q[2]),
                   y: h / 2 - s * (R[1][0] * q[0] + R[1][1] * q[1] + R[1][2] * q[2]),
                   z: R[2][0] * q[0] + R[2][1] * q[1] + R[2][2] * q[2]}; });
  p.edges = panelEdges(p.kind).filter(e => p.pos[e.a] && p.pos[e.b]);
  const selected = state.sel && key(state.sel.a, state.sel.b);
  p.edges.forEach(e => {
    const a = p.pos[e.a], b = p.pos[e.b];
    g.globalAlpha = e.alpha; g.lineWidth = e.width; g.strokeStyle = e.color; g.lineCap = 'round';
    g.setLineDash(e.dashed ? [6, 5] : []);
    if (e.k === selected) { g.save(); g.globalAlpha = 1; g.lineWidth = e.width + 6; g.strokeStyle = '#ffd33d'; g.setLineDash([]);
      g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke(); g.restore(); }
    g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
  });
  g.globalAlpha = 1; g.setLineDash([]);
  // far nodes first, so that near ones are drawn over them
  nodes.slice().sort((m, n) => p.pos[m.id].z - p.pos[n.id].z).forEach(n => {
    const q = p.pos[n.id], [fill, alpha] = nodeFill(p.kind, n);
    const r = radius * (1 + 0.18 * q.z / scaleRadius);
    q.r = r;
    g.beginPath(); g.arc(q.x, q.y, r, 0, 2 * Math.PI);
    g.fillStyle = '#ffffff'; g.fill();
    g.globalAlpha = alpha; g.fillStyle = fill; g.fill(); g.globalAlpha = 1;
    const marked = state.hoverNode === n.id || (state.hoverChain && n.chains.includes(state.hoverChain))
      || (state.sel && (state.sel.a === n.id || state.sel.b === n.id));
    g.lineWidth = marked ? 3.5 : 1.5; g.strokeStyle = marked ? '#bf8700' : '#000000';
    g.setLineDash(n.type !== 'protein' ? [4, 3] : []); g.stroke(); g.setLineDash([]);
  });
  // labels last, so that no node covers one
  g.font = '600 12px system-ui, sans-serif'; g.lineJoin = 'round';
  // each label on the side of its node that faces away from the middle of the panel
  nodes.forEach(n => { const q = p.pos[n.id], dx = q.x - w / 2, dy = q.y - h / 2, d = Math.hypot(dx, dy) || 1;
    const ux = d < 2 ? 1 : dx / d, uy = d < 2 ? 0 : dy / d, tx = q.x + ux * (q.r + 5), ty = q.y + uy * (q.r + 5);
    g.textAlign = ux > 0.35 ? 'left' : ux < -0.35 ? 'right' : 'center';
    g.textBaseline = uy > 0.6 ? 'top' : uy < -0.6 ? 'bottom' : 'middle';
    g.lineWidth = 3; g.strokeStyle = 'rgba(255,255,255,0.9)'; g.strokeText(n.label, tx, ty);
    g.fillStyle = '#1f2328'; g.fillText(n.label, tx, ty); });
}
let drawQueued = false;
function drawNets() { if (drawQueued) return; drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; panels.forEach(drawPanel); }); }
function hit(p, ev) {
  const box = p.canvas.getBoundingClientRect(), x = ev.clientX - box.left, y = ev.clientY - box.top;
  let best = null;
  for (const id in p.pos) { const q = p.pos[id]; if (Math.hypot(x - q.x, y - q.y) <= q.r + 2 && (!best || q.z > p.pos[best].z)) best = id; }
  if (best) return {node: best};
  let edge = null, near = 7;
  p.edges.forEach(e => { const a = p.pos[e.a], b = p.pos[e.b], dx = b.x - a.x, dy = b.y - a.y;
    const t = Math.max(0, Math.min(1, ((x - a.x) * dx + (y - a.y) * dy) / (dx * dx + dy * dy || 1)));
    const d = Math.hypot(x - a.x - t * dx, y - a.y - t * dy);
    if (d < near) { near = d; edge = e; } });
  return edge ? {edge} : {};
}
const tip = $('tip');
function showTip(ev, text) { if (!text) { tip.hidden = true; return; }
  tip.textContent = text; tip.hidden = false; tip.style.left = ev.clientX + 12 + 'px'; tip.style.top = ev.clientY + 12 + 'px'; }
function edgeText(k) {
  const e = edges[k], pair = pairByKey[k];
  const [a, b] = k.split('|');
  let text = nodeById[a].label + ' – ' + nodeById[b].label;
  if (e) text += ': contact in ' + contacts(k) + ' of ' + N + ' models' + (state.scoreOn ? ' (' + count(k) + ' pass the score cutoff)' : '');
  else text += ': not a contact in any model';
  const v = pairScore(pair, state.model);
  if (v != null) text += '; ' + (SCORE_LABELS[scoreName] || scoreName) + ' ' + v.toFixed(2) + ' in the model shown';
  return text;
}
panels.forEach(p => {
  p.canvas.addEventListener('mousemove', ev => {
    const h = hit(p, ev), node = h.node || null;
    if (node !== state.hoverNode) { state.hoverNode = node; drawNets();
      Mol.highlightChains(state.model, node ? nodeById[node].chains : null); }
    showTip(ev, node ? nodeById[node].label + ' (' + node + '), chain ' + nodeById[node].chains.join(', ')
      : h.edge ? edgeText(h.edge.k) : null);
  });
  p.canvas.addEventListener('mouseleave', () => { tip.hidden = true;
    if (state.hoverNode) { state.hoverNode = null; drawNets(); Mol.highlightChains(state.model, null); } });
  p.canvas.addEventListener('click', ev => { const h = hit(p, ev);
    selectPair(h.edge ? {a: h.edge.a, b: h.edge.b} : null); });
});

/* ---------- selection ---------- */
function interfaceOf(m, sel) {
  // the residues of the selected pair's interface(s) in model m, and the chain pairs involved
  const residues = [], chainPairs = [];
  if (!m || !sel) return {residues, chainPairs};
  m.interfaces[rule.id].forEach(f => {
    const [ci, cj] = f.chains, ia = chainById[ci].ids, ib = chainById[cj].ids;
    if ((ia.includes(sel.a) && ib.includes(sel.b)) || (ia.includes(sel.b) && ib.includes(sel.a))) {
      residues.push({chain: ci, resi: f.residues[0]}, {chain: cj, resi: f.residues[1]});
      chainPairs.push([ci, cj]);
    }
  });
  return {residues, chainPairs};
}
function selectionNote() {
  const m = BPV.models[state.model], sel = state.sel;
  let text = '';
  if (sel && m) {
    const name = nodeById[sel.a].label + ' – ' + nodeById[sel.b].label, n = interfaceOf(m, sel).residues.reduce((s, r) => s + r.resi.length, 0);
    text = n ? name + ': ' + n + ' interface residues marked in model ' + state.model + '.'
      : name + ' is not a contact in model ' + state.model + ', so there is no interface to mark.';
  }
  $('sel-note').textContent = text;
}
function selectPair(sel) {
  state.sel = sel; state.cell = null;
  const m = BPV.models[state.model];
  Mol.select(state.model, interfaceOf(m, sel).residues, true);
  drawNets(); drawPaeOverlay(); drawTable(); selectionNote();
}

/* ---------- PAE heatmap ---------- */
const pae = $('pae'), over = $('pae-over');
const paeColor = v => { const t = Math.min(1, v / 255); return [Math.round(0 + 255 * t), Math.round(83 + 172 * t), Math.round(28 + 227 * t)]; };
$('pae-scale').style.background = 'linear-gradient(to right, rgb(' + paeColor(0) + '), rgb(' + paeColor(255) + '))';
function drawPae() {
  const m = BPV.models[state.model];
  $('pae-panel').style.display = m && !m.pae ? 'none' : '';
  if (!m || !m.pae) return;
  const n = m.pae.n, g = pae.getContext('2d');
  pae.width = pae.height = n;
  const image = g.createImageData(n, n), bytes = m.pae.bytes;
  for (let i = 0; i < n * n; i++) { const c = paeColor(bytes[i]);
    image.data[4 * i] = c[0]; image.data[4 * i + 1] = c[1]; image.data[4 * i + 2] = c[2]; image.data[4 * i + 3] = 255; }
  g.putImageData(image, 0, 0);
  $('pae-note').textContent = '(model ' + state.model + ', ' + m.pae_rows + ' rows' + (m.pae.factor > 1 ? ', averaged over blocks of ' + m.pae.factor : '') + ')';
  drawPaeOverlay();
}
function chainSpans(m) {  // chain -> [first row, last row + 1]
  const spans = {};
  for (const c in m.rows) if (m.rows[c].length) spans[c] = [Math.min(...m.rows[c]), Math.max(...m.rows[c]) + 1];
  return spans;
}
function drawPaeOverlay() {
  const m = BPV.models[state.model];
  if (!m || !m.pae) return;
  const dpr = window.devicePixelRatio || 1, size = over.clientWidth;
  over.width = over.height = Math.round(size * dpr);
  const g = over.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, size, size);
  const px = row => row / m.pae_rows * size, spans = chainSpans(m);
  for (const c in spans) {
    const [a, b] = spans[c];
    g.strokeStyle = 'rgba(0,0,0,0.55)'; g.lineWidth = 1;
    [a, b].forEach(r => { g.beginPath(); g.moveTo(px(r), 0); g.lineTo(px(r), size); g.moveTo(0, px(r)); g.lineTo(size, px(r)); g.stroke(); });
    g.fillStyle = chainById[c] ? chainById[c].color : '#b3b3b3';
    g.fillRect(px(a), 0, px(b) - px(a), 5); g.fillRect(0, px(a), 5, px(b) - px(a));
    g.fillStyle = '#000'; g.font = '600 11px system-ui, sans-serif'; g.textBaseline = 'top';
    if (px(b) - px(a) > 14) { g.fillText(c, px(a) + 3, 7); }
  }
  interfaceOf(m, state.sel).chainPairs.forEach(([ci, cj]) => {
    if (!spans[ci] || !spans[cj]) return;
    g.strokeStyle = '#d4a72c'; g.lineWidth = 3;
    [[ci, cj], [cj, ci]].forEach(([r, c]) => g.strokeRect(px(spans[c][0]), px(spans[r][0]), px(spans[c][1]) - px(spans[c][0]), px(spans[r][1]) - px(spans[r][0])));
  });
  if (state.cell) { g.strokeStyle = '#cf222e'; g.lineWidth = 2; g.beginPath();
    g.arc(px(state.cell[1] + 0.5), px(state.cell[0] + 0.5), 6, 0, 2 * Math.PI); g.stroke(); }
  if (state.hoverResidue && m.rowOf[state.hoverResidue.chain]) {
    const row = m.rowOf[state.hoverResidue.chain][state.hoverResidue.resi];
    if (row != null) { g.strokeStyle = 'rgba(207,34,46,0.8)'; g.lineWidth = 1; g.beginPath();
      g.moveTo(0, px(row + 0.5)); g.lineTo(size, px(row + 0.5)); g.moveTo(px(row + 0.5), 0); g.lineTo(px(row + 0.5), size); g.stroke(); }
  }
}
function paeCell(ev) {
  const m = BPV.models[state.model], box = over.getBoundingClientRect();
  if (!m || !m.pae) return null;
  const j = Math.floor((ev.clientX - box.left) / box.width * m.pae.n), i = Math.floor((ev.clientY - box.top) / box.height * m.pae.n);
  if (i < 0 || j < 0 || i >= m.pae.n || j >= m.pae.n) return null;
  const f = m.pae.factor;
  return {m, i, j, row: i * f, col: j * f, value: m.pae.bytes[i * m.pae.n + j] * m.pae.step, a: m.rowInfo[i * f], b: m.rowInfo[j * f]};
}
const residueText = r => r ? (chainById[r.chain] && nodeById[chainById[r.chain].ids[0]] ? nodeById[chainById[r.chain].ids[0]].label + ' ' : '') + r.chain + ':' + r.resi : 'not a protein residue';
over.addEventListener('mousemove', ev => { const c = paeCell(ev);
  $('pae-hover').textContent = c ? 'Aligned on ' + residueText(c.a) + ', error at ' + residueText(c.b) + ': ' + c.value.toFixed(1) + ' Å' : ''; });
over.addEventListener('mouseleave', () => { $('pae-hover').textContent = ''; });
over.addEventListener('click', ev => { const c = paeCell(ev);
  if (!c || !c.a || !c.b) return;
  state.sel = null; state.cell = [c.row, c.col]; $('sel-note').textContent = residueText(c.a) + ' and ' + residueText(c.b) + ' marked.';
  Mol.select(state.model, [{chain: c.a.chain, resi: [c.a.resi]}, {chain: c.b.chain, resi: [c.b.resi]}], false);
  drawNets(); drawPaeOverlay(); drawTable(); });

/* ---------- table ---------- */
function drawTable() {
  const scoreLabel = SCORE_LABELS[scoreName] || scoreName;
  const head = ['Pair', 'Contact in', 'Model ' + state.model].concat(scoreName ? [scoreLabel + ' (model ' + state.model + ')'] : [],
    ['293T', 'HCT116'], job.has_reference ? ['Reference'] : []);
  document.querySelector('#pairs thead').innerHTML = '<tr>' + head.map(h => '<th>' + h + '</th>').join('') + '</tr>';
  const body = document.querySelector('#pairs tbody'); body.textContent = '';
  if (!rule.pairs.length) { body.innerHTML = '<tr><td colspan="' + head.length + '">No pair of these proteins is a contact in a model or an interaction in BioPlex.</td></tr>'; return; }
  const yes = v => v ? 'yes' : '–';
  rule.pairs.map(p => ({p, k: key(p.a, p.b)})).map(x => Object.assign(x, {n: count(x.k)}))
    .sort((x, y) => y.n - x.n).forEach(({p, k, n}) => {
      const tr = document.createElement('tr'), v = pairScore(p, state.model);
      const cells = [nodeById[p.a].label + ' – ' + nodeById[p.b].label, countText(k), yes(inModel(k, state.model))]
        .concat(scoreName ? [v == null ? '' : v.toFixed(2)] : [], [yes(p.bp293), yes(p.bpHct)], job.has_reference ? [yes(p.reference)] : []);
      tr.innerHTML = cells.map(c => '<td>' + c + '</td>').join('');
      if (n < state.minK) tr.className = 'dim';
      if (state.sel && k === key(state.sel.a, state.sel.b)) tr.classList.add('on');
      tr.addEventListener('click', () => selectPair({a: p.a, b: p.b}));
      body.appendChild(tr);
    });
}

/* ---------- controls ---------- */
function modelText(i) {
  const m = job.models[i], s = Object.entries(m.scores).map(([name, v]) => name.replace(/_/g, ' ') + ' ' + v.toFixed(2));
  return m.name + (s.length ? ': ' + s.join(', ') : '') + (i ? '; CA RMSD to model 0 ' + m.rmsd_to_first.toFixed(1) + ' Å' : '');
}
async function showModel(i) {
  state.model = i; state.cell = null;
  document.querySelectorAll('#models button').forEach((b, j) => b.classList.toggle('on', j === i));
  $('model-note').textContent = '(' + modelText(i) + ')';
  drawNets(); drawTable();
  try {
    const m = await modelData(i);
    if (state.model !== i) return;
    drawPae();
    if (Mol.ready) { await Mol.load(i, m); if (state.model !== i) return; Mol.show(i);
      Mol.select(i, interfaceOf(m, state.sel).residues, false); }
    selectionNote();
  } catch (e) { $('mol-message').hidden = false; $('mol-message').textContent = String(e.message || e); }
}
function setup() {
  $('title').textContent = job.title;
  document.title = job.title + ' – BioPlexPy';
  $('settings').textContent = N + ' model' + (N === 1 ? '' : 's') + ' of one prediction' + (job.tool ? ' (' + ({boltz: 'Boltz', af3: 'AlphaFold3', colabfold: 'ColabFold'}[job.tool] || job.tool) + ')' : '') + '.';
  const ruleText = c => (c.contact_atoms === 'ca' ? 'C\u03b1 atoms' : 'any two atoms') + ' closer than ' + c.distance + ' \u00c5'
    + (c.min_plddt != null ? ', pLDDT at least ' + c.min_plddt : '') + (c.max_pae != null ? ', PAE at most ' + c.max_pae + ' \u00c5' : '');
  // one button per rule, like the model buttons; the rule itself is in the button's tooltip
  job.rules.forEach((r, i) => { const b = document.createElement('button');
    b.textContent = r.label; b.title = 'A contact: ' + ruleText(r.contact); b.classList.toggle('on', i === 0);
    b.addEventListener('click', () => { useRule(i);
      document.querySelectorAll('#rules button').forEach((x, j) => x.classList.toggle('on', j === i));
      Mol.select(state.model, interfaceOf(BPV.models[state.model], state.sel).residues, false);
      drawNets(); drawPaeOverlay(); drawTable(); selectionNote(); });
    $('rules').appendChild(b); });
  if (job.rules.length === 1) $('rules').closest('.group').style.display = 'none';
  job.models.forEach((m, i) => { const b = document.createElement('button'); b.textContent = i; b.title = modelText(i);
    b.addEventListener('click', () => showModel(i)); $('models').appendChild(b); });
  const k = $('min-k'); k.max = N; k.value = state.minK;
  const kText = () => { $('min-k-text').textContent = state.minK + ' of ' + N + ' models'; };
  k.addEventListener('input', () => { state.minK = +k.value; kText(); drawNets(); drawTable(); }); kText();
  if (N === 1) k.closest('.group').style.display = 'none';
  if (!scoreName) $('score-group').style.display = 'none';
  else {
    $('score-name').textContent = SCORE_LABELS[scoreName] || scoreName;
    const on = $('score-on'), cut = $('score-cut');
    on.checked = state.scoreOn; cut.value = state.cut;
    const text = () => { $('score-cut-text').textContent = (+cut.value).toFixed(2); cut.disabled = !on.checked; };
    const change = () => { state.scoreOn = on.checked; state.cut = +cut.value; text(); drawNets(); drawTable(); };
    on.addEventListener('change', change); cut.addEventListener('input', change); text();
  }
  document.querySelectorAll('input[name=color]').forEach(r => r.addEventListener('change', async () => {
    state.color = r.value; for (const i in Mol.structures) await Mol.color(+i, state.color); }));
  window.addEventListener('resize', () => { drawNets(); drawPaeOverlay(); });
  Mol.onRotate = R => { state.R = R; drawNets(); };
  Mol.onResidue = (kind, r) => {
    if (kind === 'hover') { const chain = r ? r.chain : null;
      const changed = chain !== state.hoverChain || (r && state.hoverResidue && r.resi !== state.hoverResidue.resi) || (!r !== !state.hoverResidue);
      state.hoverChain = chain; state.hoverResidue = r;
      if (changed) { drawNets(); drawPaeOverlay(); } }
  };
}
setup();
BPV.state = state; BPV.Mol = Mol; BPV.count = count;   // for inspection from the console
BPV.ready = Mol.init($('mol')).catch(e => { $('mol').style.display = 'none'; $('mol-message').hidden = false;
  $('mol-message').textContent = String(e.message || e); }).then(() => showModel(0));
})();
