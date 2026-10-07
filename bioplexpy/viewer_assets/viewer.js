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
  only: null,         // name of the overview count whose pairs alone are listed in the table
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
  ready: false, refs: {}, onRotate: null, onResidue: null,
  // Model i as Mol* holds it now. Mol* makes these objects anew whenever its state changes (a
  // representation added or changed in its own controls), so none is kept: only the reference.
  structure(i) {
    return this.plugin.managers.structure.hierarchy.current.structures.find(s => s.cell.transform.ref === this.refs[i]);
  },
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
    // a representation added or changed in Mol*'s own controls comes with Mol*'s coloring:
    // give it the page's
    plugin.managers.structure.hierarchy.behaviors.selection.subscribe(() => setTimeout(() => { this.hideOthers(); this.recolor(); }, 0));
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
    if (this.refs[i]) return;
    const H = this.plugin.managers.structure.hierarchy;
    const before = new Set(H.current.structures.map(s => s.cell.transform.ref));
    this.loading = true;
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
    this.refs[i] = H.current.structures.find(s => !before.has(s.cell.transform.ref)).cell.transform.ref;
    this.loading = false;
    this.addTheme('bpv-plddt-' + i, (chain, resi) => {
      const v = m.plddtOf[chain] && m.plddtOf[chain][resi];
      return v == null ? GREY : PLDDT_COLORS.find(([floor]) => v > floor)[1];
    });
    await this.color(i, state.color);
    if (Object.keys(this.refs).length === 1) { this.plugin.canvas3d.handleResize(); this.plugin.managers.camera.reset(); }
  },
  // the parts of model i that take the page's coloring: not the sticks Mol* draws around a
  // residue in focus, which keep Mol*'s colors by element
  colored(s) {
    return s.components.filter(c => !(c.cell.transform.tags || []).some(tag => tag.startsWith('structure-focus')));
  },
  themeName(i, mode) { return mode === 'plddt' ? 'bpv-plddt-' + i : 'bpv-chain'; },
  async color(i, mode) {
    const s = this.structure(i);
    if (s) await this.plugin.managers.structure.component.updateRepresentationsTheme(
      this.colored(s), {color: this.themeName(i, mode)});
  },
  // color again every model that has a representation not in the page's coloring
  async recolor() {
    if (this.recoloring || this.loading) return;
    this.recoloring = true;
    try {
      for (const i in this.refs) {
        const s = this.structure(+i), name = this.themeName(+i, state.color);
        if (s && this.colored(s).some(c => c.representations.some(r => r.cell.transform.params.colorTheme.name !== name)))
          await this.color(+i, state.color);
      }
    } finally { this.recoloring = false; }
  },
  show(i) {
    const H = this.plugin.managers.structure.hierarchy;
    this.shown = i;
    for (const j in this.refs) { const s = this.structure(+j); if (s) H.toggleVisibility([s], +j === i ? 'show' : 'hide'); }
  },
  // what Mol*'s controls make anew (a preset replaces every component, in every model) is
  // visible: hide it again in the models that are not shown
  hideOthers() {
    if (this.loading || this.shown == null) return;
    const H = this.plugin.managers.structure.hierarchy;
    for (const j in this.refs) { const s = this.structure(+j);
      if (s && +j !== this.shown && s.components.some(c => !c.cell.state.isHidden || c.representations.some(r => !r.cell.state.isHidden)))
        H.toggleVisibility([s], 'hide'); }
  },
  // The components and representations under model i, as Mol* holds them: (transformer,
  // parameters, tags), nested. The sticks around a residue in focus are left out.
  style(i) {
    const data = this.plugin.state.data;
    const walk = ref => data.tree.children.get(ref).toArray().map(r => data.cells.get(r))
      .filter(c => c && !(c.transform.tags || []).some(tag => tag.startsWith('structure-focus')))
      .map(c => ({ref: c.transform.ref, transformer: c.transform.transformer, params: c.transform.params,
                  tags: c.transform.tags, children: walk(c.transform.ref)}));
    return walk(this.refs[i]);
  },
  styleKey(nodes) {
    return JSON.stringify(nodes.map(n => [n.transformer.id, Object.assign({}, n.params, {colorTheme: null}), this.styleKey(n.children)]));
  },
  // Give model `to` the style of model `from`. A change made in Mol*'s controls reaches the
  // models that are loaded; one loaded later, or changed alone, would look different.
  async copyStyle(from, to) {
    if (from == null || from === to || !this.refs[from] || !this.refs[to]) return;
    let want, have;
    try { want = this.style(from); have = this.style(to); if (this.styleKey(want) === this.styleKey(have)) return; }
    catch (e) { return; }
    this.loading = true;
    try {
      const build = this.plugin.build();
      have.forEach(n => build.delete(n.ref));
      const add = (parent, nodes) => nodes.forEach(n => add(build.to(parent).apply(n.transformer, n.params, {tags: n.tags}).ref, n.children));
      add(this.refs[to], want);
      await build.commit();
    } catch (e) { console.warn('style of model ' + from + ' not copied to model ' + to + ': ' + e); }
    finally { this.loading = false; }
    await this.color(to, state.color);
  },
  // residues: [{chain, resi: [numbers]}], of model i
  loci(i, residues) {
    const chains = [], numbers = [];
    residues.forEach(r => r.resi.forEach(n => { chains.push(r.chain); numbers.push(n); }));
    return this.S.StructureElement.Loci.fromSchema(this.structure(i).cell.obj.data,
      {items: {auth_asym_id: chains, auth_seq_id: numbers}});
  },
  select(i, residues, zoom) {
    if (!this.ready || !this.structure(i)) return;
    const I = this.plugin.managers.interactivity;
    I.lociSelects.deselectAll();
    this.plugin.managers.structure.focus.clear();
    if (!residues || !residues.length) return;
    const loci = this.loci(i, residues);
    I.lociSelects.select({loci});
    if (zoom) this.plugin.managers.camera.focusLoci(loci, {extraRadius: 14});
  },
  // as a click on a residue in Mol*'s own sequence does: the residue and what surrounds it
  // drawn as sticks, and the camera on it
  focusResidue(i, chain, resi) {
    if (!this.ready || !this.structure(i)) return;
    const loci = this.loci(i, [{chain, resi: [resi]}]);
    this.plugin.managers.interactivity.lociSelects.deselectAll();
    this.plugin.managers.structure.focus.setFromLoci(loci);
    this.plugin.managers.camera.focusLoci(loci);
  },
  highlightResidue(i, chain, resi) {
    if (!this.ready || !this.structure(i)) return;
    const I = this.plugin.managers.interactivity;
    if (chain == null) I.lociHighlights.clearHighlights();
    else I.lociHighlights.highlightOnly({loci: this.loci(i, [{chain, resi: [resi]}])});
  },
  highlightChains(i, chains) {
    if (!this.ready || !this.structure(i)) return;
    const I = this.plugin.managers.interactivity;
    if (!chains || !chains.length) { I.lociHighlights.clearHighlights(); return; }
    I.lociHighlights.highlightOnly({loci: this.S.StructureElement.Loci.fromSchema(
      this.structure(i).cell.obj.data, {items: {auth_asym_id: chains}})});
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
  if (n.type !== 'protein') return [n.color, 1];   // white or a light grey, as the chain in the structure
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
  drawNets(); drawPaeOverlay(); drawTable(); selectionNote(); drawSeq();
}

/* ---------- sequences ---------- */
const hexColor = v => '#' + v.toString(16).padStart(6, '0');
// the color of a residue's letter: the structure's coloring, lighter where it is the chain's
function residueColor(m, chain, resi) {
  if (state.color === 'plddt') { const v = m.plddtOf[chain.id] && m.plddtOf[chain.id][resi];
    return v == null ? null : hexColor(PLDDT_COLORS.find(([floor]) => v > floor)[1]); }
  return chain.color + '59';
}
const chainLabel = c => (nodeById[c.ids[0]] ? nodeById[c.ids[0]].label + ' ' : '') + c.id;
// one line of letters per chain of the model shown; the residues of the selected pair's
// interface are bold and underlined
function drawSeq() {
  const m = BPV.models[state.model], box = $('seq');
  if (!m || !m.sequence) { box.textContent = ''; return; }
  const marked = {};
  interfaceOf(m, state.sel).residues.forEach(r => { const s = marked[r.chain] = marked[r.chain] || new Set(); r.resi.forEach(n => s.add(n)); });
  let html = '';
  job.chains.forEach(c => { const letters = m.sequence[c.id];
    if (!letters) return;
    html += '<div class="seq-row"><span class="seq-name">' + chainLabel(c) + '</span>';
    m.residues[c.id].forEach((n, k) => { const color = residueColor(m, c, n);
      html += '<span class="r' + (marked[c.id] && marked[c.id].has(n) ? ' if' : '') + '" data-c="' + c.id + '" data-n="' + n + '"'
        + (color ? ' style="background:' + color + '"' : '') + '>' + letters[k] + '</span>'; });
    html += '</div>'; });
  box.innerHTML = html;
}
const seqResidue = ev => { const el = ev.target.closest && ev.target.closest('.r');
  return el ? {chain: el.dataset.c, resi: +el.dataset.n, letter: el.textContent} : null; };
$('seq').addEventListener('mousemove', ev => { const r = seqResidue(ev), m = BPV.models[state.model];
  if (!r) { showTip(ev, null); return; }
  const v = m && m.plddtOf[r.chain] && m.plddtOf[r.chain][r.resi];
  showTip(ev, chainLabel(chainById[r.chain]) + ': ' + r.letter + r.resi + (v != null ? ', pLDDT ' + v.toFixed(0) : ''));
  Mol.highlightResidue(state.model, r.chain, r.resi); });
$('seq').addEventListener('mouseleave', () => { tip.hidden = true; Mol.highlightResidue(state.model, null); });
$('seq').addEventListener('click', ev => { const r = seqResidue(ev);
  if (!r) return;
  state.sel = null; state.cell = null;
  $('sel-note').textContent = chainLabel(chainById[r.chain]) + ': ' + r.letter + r.resi + ' in focus.';
  Mol.focusResidue(state.model, r.chain, r.resi);
  drawNets(); drawPaeOverlay(); drawTable(); drawSeq(); });
// the residue under the pointer in the structure, marked in the sequences and brought into view
function markSeq(r) {
  const box = $('seq'), old = box.querySelector('.r.hov');
  if (old) old.classList.remove('hov');
  const el = r && box.querySelector('.r[data-c="' + r.chain + '"][data-n="' + r.resi + '"]');
  if (!el) return;
  el.classList.add('hov');
  const top = el.offsetTop - box.offsetTop;
  if (top < box.scrollTop || top > box.scrollTop + box.clientHeight - 20) box.scrollTop = top - box.clientHeight / 2;
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
// what job.js has on a pair's interface in model i under the rule in use (bioplexpy/interfaces.py),
// with the two sides in the order of the pair's name
function interfaceNumbers(p, i) {
  const all = job.models[i].interfaces, o = all && all[rule.id][key(p.a, p.b)];
  if (!o) return null;
  const turn = v => v && (p.a < p.b ? v : [v[1], v[0]]);
  return Object.assign({}, o, {residues: turn(o.residues), plddt: turn(o.plddt)});
}
/* ---------- overview ---------- */
// The job in a few numbers, under the contact rule (and the score cutoff) in use. Each count of
// protein pairs is a set of pair keys: a click lists those pairs alone in the table.
function overviewSets() {
  const N1 = N > 1, sets = {any: [], all: [], bp293: [], bpHct: [], missed: [], failing: []};
  rule.pairs.forEach(p => { const k = key(p.a, p.b), n = count(k);
    if (n > 0) { sets.any.push(k); if (n === N) sets.all.push(k); if (p.bp293) sets.bp293.push(k); if (p.bpHct) sets.bpHct.push(k); }
    else {
      if (p.bp293 || p.bpHct) sets.missed.push(k);
      // residues close enough in some model, and a contact in none
      if (job.models.some(m => { const o = m.interfaces && m.interfaces[rule.id][k]; return o && o.n_close > 0; })) sets.failing.push(k);
    } });
  return {sets, N1};
}
function drawOverview() {
  const box = $('overview'), {sets, N1} = overviewSets();
  const proteins = job.chains.filter(c => c.type === 'protein'), others = job.chains.length - proteins.length;
  const residues = job.chains.reduce((n, c) => n + c.length, 0);
  const range = name => { const v = job.models.map(m => m.scores[name]).filter(x => x != null);
    if (!v.length) return null;
    const lo = Math.min(...v), hi = Math.max(...v);
    return lo.toFixed(2) + (hi.toFixed(2) !== lo.toFixed(2) ? '\u2013' + hi.toFixed(2) : ''); };
  const scores = [['iptm', 'ipTM'], ['ptm', 'pTM']].filter(([name]) => proteins.length > 1 || name !== 'iptm')
    .map(([name, label]) => range(name) ? label + ' ' + range(name) : null).filter(Boolean);
  const pairs = n => n + ' pair' + (n === 1 ? '' : 's');
  const button = (name, text, title) => '<button data-only="' + name + '" title="' + title + '"' + (state.only === name ? ' class="on"' : '') + '>' + text + '</button>';
  const item = (label, html) => '<span class="item"><span class="label">' + label + '</span>' + html + '</span>';
  const items = [
    item('Size', proteins.length + ' protein chain' + (proteins.length === 1 ? '' : 's') + (others ? ' and ' + others + ' other' : '') + ', '
      + residues.toLocaleString('en-US') + ' residues, ' + N + ' model' + (N1 ? 's' : '')),
    scores.length ? item('Confidence', scores.join(', ') + (N1 ? ' over the models' : '')) : '',
    item('Contacts (' + rule.label + ')', button('any', pairs(sets.any.length), 'Protein pairs in contact' + (N1 ? ' in at least one model' : ''))
      + (N1 ? button('all', sets.all.length + ' in all ' + N + ' models', 'Protein pairs in contact in every model') : '')),
    item('Detected by BioPlex', button('bp293', sets.bp293.length + ' in 293T', 'Pairs in contact that BioPlex detected in 293T')
      + button('bpHct', sets.bpHct.length + ' in HCT116', 'Pairs in contact that BioPlex detected in HCT116') + '<span>of ' + sets.any.length + ';</span>'
      + button('missed', sets.missed.length + ' not in contact', 'BioPlex interactions between these proteins that are not a contact in any model')),
    job.models[0].interfaces ? item('Close, not a contact', button('failing', pairs(sets.failing.length),
      'Pairs with residues close enough in some model that fail the rule in every model: see the columns Residue pairs and Failing')) : '',
  ];
  box.innerHTML = items.join('');
  const names = {any: 'in contact', all: 'in contact in all models', bp293: 'in contact and detected in 293T', bpHct: 'in contact and detected in HCT116',
    missed: 'BioPlex interactions that are not a contact', failing: 'close but not a contact'};
  $('only-note').innerHTML = state.only ? '(' + names[state.only] + ' only) <button>show all</button>' : '';
  return state.only ? new Set(sets[state.only]) : null;
}
$('overview').addEventListener('click', ev => { const b = ev.target.closest('button[data-only]');
  if (!b) return;
  state.only = state.only === b.dataset.only ? null : b.dataset.only; drawTable(); });
$('only-note').addEventListener('click', ev => { if (ev.target.closest('button')) { state.only = null; drawTable(); } });

const INTERFACE_HEAD = [
  ['Residue pairs', 'Pairs of residues that pass the contact rule, of those that are close enough (within the rule\'s distance)'],
  ['Failing', 'Close residue pairs that fail the rule\'s pLDDT condition, and its PAE condition; a pair can fail both'],
  ['PAE', 'Median and lowest PAE over the residue pairs that pass (over the close pairs where none passes), the smaller of the two directions'],
  ['pLDDT', 'Mean pLDDT of the interface residues of either protein, in the order of the pair\'s name']];
function interfaceCells(p) {
  const o = interfaceNumbers(p, state.model);
  if (!o) return ['', '', '', ''];
  const pae = o.pae_pass || o.pae_close;
  return [o.n_pass + ' of ' + o.n_close,
    o.n_low_plddt == null && o.n_high_pae == null ? '–' : !o.n_low_plddt && !o.n_high_pae ? 'none'
      : [o.n_low_plddt != null ? o.n_low_plddt + ' pLDDT' : null, o.n_high_pae != null ? o.n_high_pae + ' PAE' : null].filter(Boolean).join(', '),
    pae ? pae[0].toFixed(1) + ' / ' + pae[1].toFixed(1) + (o.pae_pass ? '' : ' (close)') : '–',
    o.plddt ? o.plddt.map(v => v == null ? '–' : v.toFixed(0)).join(' / ') : '–'];
}
// the residues of the selected pair's interface in the model shown: one line each, either side
function interfaceDetail(p, m, columns) {
  const detail = document.createElement('tr'); detail.className = 'detail';
  const lines = [];
  ((m && m.contacts && m.contacts[rule.id]) || []).forEach(f => {
    const ids = f.chains.map(c => chainById[c].ids);
    const sides = ids[0].includes(p.a) && ids[1].includes(p.b) ? [0, 1] : ids[0].includes(p.b) && ids[1].includes(p.a) ? [1, 0] : null;
    if (!sides) return;
    sides.forEach(side => { const c = f.chains[side], other = f.chains[1 - side], letters = m.sequence && m.sequence[c];
      const at = letters ? Object.fromEntries(m.residues[c].map((r, k) => [r, letters[k]])) : {};
      const atOther = m.sequence && m.sequence[other] ? Object.fromEntries(m.residues[other].map((r, k) => [r, m.sequence[other][k]])) : {};
      f.residues[side].forEach(([n, nPass, nClose, partner, distance, pae, plddt]) => lines.push(
        '<tr class="' + (nPass ? '' : 'dim') + '" data-c="' + c + '" data-n="' + n + '"><td>' + chainLabel(chainById[c]) + '</td><td>' + (at[n] || '') + n + '</td><td>'
        + (plddt == null ? '–' : plddt.toFixed(0)) + '</td><td>' + nPass + ' of ' + nClose + '</td><td>' + chainLabel(chainById[other]) + ' ' + (atOther[partner] || '') + partner
        + '</td><td>' + distance.toFixed(1) + '</td><td>' + (pae == null ? '–' : pae.toFixed(1)) + '</td></tr>')); });
  });
  detail.innerHTML = '<td colspan="' + columns + '">' + (!m ? 'Loading the model...' : !lines.length ? 'No residues of this pair are close in model ' + state.model + '.'
    : '<div class="detail-box"><table><thead><tr><th>Chain</th><th>Residue</th><th>pLDDT</th><th title="Partner residues that pass the contact rule, of those that are close">Partners</th>'
      + '<th title="The nearest partner residue: one that passes, if there is any">Nearest partner</th><th>Distance (Å)</th><th title="PAE to the nearest partner, the smaller of the two directions">PAE (Å)</th></tr></thead><tbody>'
      + lines.join('') + '</tbody></table></div><div class="legend">Every residue with a partner close enough, in model ' + state.model
      + '; grey: no partner passes the rule. Click a line to bring the residue into focus.</div>') + '</td>';
  detail.addEventListener('click', ev => { const tr = ev.target.closest('tr[data-c]');
    if (tr) Mol.focusResidue(state.model, tr.dataset.c, +tr.dataset.n); });
  return detail;
}
function drawTable() {
  const scoreLabel = SCORE_LABELS[scoreName] || scoreName;
  const head = [['Pair'], ['Contact in'], ['Model ' + state.model]].concat(scoreName ? [[scoreLabel + ' (model ' + state.model + ')']] : [],
    INTERFACE_HEAD, [['293T'], ['HCT116']], job.has_reference ? [['Reference']] : []);
  document.querySelector('#pairs thead').innerHTML = '<tr>' + head.map(([h, title]) => '<th' + (title ? ' title="' + title + '"' : '') + '>' + h + '</th>').join('') + '</tr>';
  const body = document.querySelector('#pairs tbody'); body.textContent = '';
  const only = drawOverview();
  if (!rule.pairs.length) { body.innerHTML = '<tr><td colspan="' + head.length + '">No pair of these proteins is a contact in a model or an interaction in BioPlex.</td></tr>'; return; }
  if (only && !only.size) body.innerHTML = '<tr><td colspan="' + head.length + '">No such pair.</td></tr>';
  const yes = v => v ? 'yes' : '–';
  rule.pairs.map(p => ({p, k: key(p.a, p.b)})).filter(x => !only || only.has(x.k)).map(x => Object.assign(x, {n: count(x.k)}))
    .sort((x, y) => y.n - x.n).forEach(({p, k, n}) => {
      const tr = document.createElement('tr'), v = pairScore(p, state.model);
      const cells = [nodeById[p.a].label + ' – ' + nodeById[p.b].label, countText(k), yes(inModel(k, state.model))]
        .concat(scoreName ? [v == null ? '' : v.toFixed(2)] : [], interfaceCells(p), [yes(p.bp293), yes(p.bpHct)], job.has_reference ? [yes(p.reference)] : []);
      tr.innerHTML = cells.map(c => '<td>' + c + '</td>').join('');
      if (n < state.minK) tr.className = 'dim';
      const selected = state.sel && k === key(state.sel.a, state.sel.b);
      if (selected) tr.classList.add('on');
      // a second click on the selected pair closes it
      tr.addEventListener('click', () => selectPair(selected ? null : {a: p.a, b: p.b}));
      body.appendChild(tr);
      if (selected) body.appendChild(interfaceDetail(p, BPV.models[state.model], head.length));
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
    drawPae(); drawSeq(); drawTable();
    if (Mol.ready) { const before = Mol.shown;
      await Mol.load(i, m); if (state.model !== i) return;
      await Mol.copyStyle(before, i); if (state.model !== i) return;
      Mol.show(i);
      Mol.select(i, interfaceOf(m, state.sel).residues, false); }
    selectionNote();
  } catch (e) { $('mol-message').hidden = false; $('mol-message').textContent = String(e.message || e); }
}
function setup() {
  $('title').textContent = job.title;
  document.title = job.title + ' – BioPlexPy';
  // structures without confidence files (an experimental structure): no pLDDT to color by,
  // and the contact rule's pLDDT and PAE conditions were not applied
  if (!job.confidence) $('color-group').style.display = 'none';
  $('settings').textContent = !job.confidence
    ? N + ' structure' + (N === 1 ? '' : 's') + ' without confidence data: contacts are judged on distance alone.'
    : N + ' model' + (N === 1 ? '' : 's') + ' of one prediction' + (job.tool ? ' (' + ({boltz: 'Boltz', af3: 'AlphaFold3', colabfold: 'ColabFold'}[job.tool] || job.tool) + ')' : '') + '.';
  const ruleText = c => (c.contact_atoms === 'ca' ? 'C\u03b1 atoms' : 'any two atoms') + ' closer than ' + c.distance + ' \u00c5'
    + (c.min_plddt != null ? ', pLDDT at least ' + c.min_plddt : '') + (c.max_pae != null ? ', PAE at most ' + c.max_pae + ' \u00c5' : '');
  // one button per rule, like the model buttons, with the number of protein pairs that are a
  // contact in some model under it; the rule itself is in the button's tooltip
  const called = r => r.pairs.filter(p => p.n_contact > 0).length;
  const pairsText = n => n + ' protein pair' + (n === 1 ? '' : 's');
  // said outright when the rule in use finds no contact and another rule does
  const ruleNote = () => { const more = called(rule) ? [] : job.rules.filter(r => called(r));
    $('rule-note').hidden = !more.length;
    $('rule-note').textContent = !more.length ? '' : 'No protein pair is a contact under the rule ' + rule.label + '. '
      + more.map(r => 'Under ' + r.label + ', ' + pairsText(called(r)) + (called(r) === 1 ? ' is' : ' are')).join('; ')
      + ': choose the contact rule above to see ' + (more.length === 1 && called(more[0]) === 1 ? 'it' : 'them') + '.'; };
  ruleNote();
  job.rules.forEach((r, i) => { const b = document.createElement('button');
    b.textContent = r.label + ' (' + called(r) + ')';
    b.title = 'A contact: ' + ruleText(r.contact) + '. ' + pairsText(called(r)) + ' in contact in at least one model.'; b.classList.toggle('on', i === 0);
    b.addEventListener('click', () => { useRule(i); ruleNote();
      document.querySelectorAll('#rules button').forEach((x, j) => x.classList.toggle('on', j === i));
      Mol.select(state.model, interfaceOf(BPV.models[state.model], state.sel).residues, false);
      drawNets(); drawPaeOverlay(); drawTable(); selectionNote(); drawSeq(); });
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
    state.color = r.value; drawSeq(); for (const i in Mol.refs) await Mol.color(+i, state.color); }));
  window.addEventListener('resize', () => { drawNets(); drawPaeOverlay(); });
  // the two layouts: everything side by side, or the structure across the page; the choice is kept
  const wide = on => { document.body.classList.toggle('wide', on); $('wide').textContent = on ? 'Side by side' : 'Wide structure';
    try { localStorage.setItem('bpv-wide', on ? '1' : ''); } catch (e) { /* a page opened from a file may have no storage */ }
    drawNets(); drawPaeOverlay(); };
  let startWide = false;
  try { startWide = localStorage.getItem('bpv-wide') === '1'; } catch (e) { /* as above */ }
  wide(startWide);
  $('wide').addEventListener('click', () => wide(!document.body.classList.contains('wide')));
  Mol.onRotate = R => { state.R = R; drawNets(); };
  Mol.onResidue = (kind, r) => {
    if (kind === 'hover') { const chain = r ? r.chain : null;
      const changed = chain !== state.hoverChain || (r && state.hoverResidue && r.resi !== state.hoverResidue.resi) || (!r !== !state.hoverResidue);
      state.hoverChain = chain; state.hoverResidue = r;
      if (changed) { drawNets(); drawPaeOverlay(); markSeq(r); } }
  };
}
setup();
BPV.state = state; BPV.Mol = Mol; BPV.count = count;   // for inspection from the console
BPV.ready = Mol.init($('mol')).catch(e => { $('mol').style.display = 'none'; $('mol-message').hidden = false;
  $('mol-message').textContent = String(e.message || e); }).then(() => showModel(0));
})();
