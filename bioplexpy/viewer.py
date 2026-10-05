'''
Interactive web page for the models of one prediction job
(`bioplexpy-structure --viewer`): the 3D structure, the three Figure 2-style
network panels drawn in 3D so that they turn with the structure, the PAE as a
heatmap, and in how many of the models each protein pair is a contact.

Everything is computed here, with the same functions the tables and the
static figure use; the page only displays it. write_viewer() writes a folder:

  viewer/index.html, viewer.js, viewer.css   the page (copied from viewer_assets/)
  viewer/job.js                              chains, nodes, pairs, per-model summaries
  viewer/model_<n>.js                        one model: coordinates, PAE, pLDDT, interfaces

The data files are JavaScript, not JSON, because a page opened from a local
folder may load scripts but may not fetch files. A model's file is loaded
when that model is first shown.
'''

import base64
import json
import os
import re
import shutil
import tempfile
import warnings

import numpy as np
from scipy.spatial.distance import cdist

ASSET_DIR = os.path.join(os.path.dirname(__file__), 'viewer_assets')
ASSET_FILES = ('index.html', 'viewer.js', 'viewer.css')

# PAE is stored one byte per cell, in steps of PAE_STEP angstrom
PAE_STEP = 0.125
# a larger PAE matrix is block-averaged down to at most this many rows
MAX_PAE_ROWS = 3000

# the model-level scores each predictor writes, most informative first
MODEL_SCORE_KEYS = {
    'boltz': ('confidence_score', 'iptm', 'ptm', 'complex_plddt'),
    'af3': ('ranking_score', 'iptm', 'ptm'),
    'colabfold': ('iptm', 'ptm'),
}


def kabsch(mobile, target):
    '''
    The rotation R (3, 3) and translation t (3,) that best place the points
    `mobile` (N, 3) on `target` (N, 3) in the least-squares sense:
    R @ x + t for a point x of `mobile`.
    '''
    mobile, target = np.asarray(mobile, dtype=float), np.asarray(target, dtype=float)
    mobile_mean, target_mean = mobile.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - mobile_mean).T @ (target - target_mean))
    # no reflection: the determinant of the rotation must be +1
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, target_mean - rotation @ mobile_mean


def _protein_ca(model):
    '''CA coordinates of a model's protein residues, keyed by (chain ID, residue number).'''
    from bioplexpy.analysis_funcs import classify_chain

    return {(chain.get_id(), residue.id[1]): residue['CA'].coord.astype(float)
            for chain in model if classify_chain(chain) == 'protein'
            for residue in chain if residue.id[0] == ' ' and 'CA' in residue}


def _chain_sequences(model):
    '''Residue names of every chain, in file order: what two models of one job must share.'''
    return {chain.get_id(): [residue.get_resname() for residue in chain] for chain in model}


def common_frame(model, reference_ca, frame_rotation, frame_center):
    '''
    The transform that puts `model` in the page's common frame: fitted onto
    the reference model by the CA atoms the two share, then turned into the
    reference's own display orientation (frame_rotation, frame_center, from
    _pca_rotation_matrix()).

    Returns (rotation, center, rmsd): a point x of the model goes to
    rotation @ (x - center), the form _write_rotated_structure() takes;
    rmsd is over the shared CA atoms after the fit.
    '''
    ca = _protein_ca(model)
    shared = sorted(set(ca) & set(reference_ca))
    if len(shared) < 3:
        raise ValueError('fewer than three CA atoms in common with the first model')
    mobile = np.array([ca[key] for key in shared])
    target = np.array([reference_ca[key] for key in shared])
    fit_rotation, fit_translation = kabsch(mobile, target)
    rmsd = float(np.sqrt(((mobile @ fit_rotation.T + fit_translation - target) ** 2)
                         .sum(axis=1).mean()))
    # frame_rotation @ (fit_rotation @ x + fit_translation - frame_center)
    rotation = frame_rotation @ fit_rotation
    center = fit_rotation.T @ (frame_center - fit_translation)
    return rotation, center, rmsd


def interface_residues(model, chain_pairs, contact, pae_data=None):
    '''
    For each contacting chain pair, the residues on either side that make
    the contact, found the way _direct_interaction_chain_pairs() finds the
    pair itself: by _ca_interface() for two protein chains under a CA rule,
    otherwise by any two polymer atoms within the distance.

    Parameters
    ----------
    model: Bio.PDB model
    chain_pairs: list of [chain_i, chain_j] (from _direct_interaction_chain_pairs())
    contact: dict from resolve_contact_settings()
    pae_data: dict from read_pae() (optional)

    Returns
    -------
    list of dict: chains [i, j] and residues [[numbers of i], [numbers of j]]
    '''
    from bioplexpy.analysis_funcs import (NUCLEIC_ACID_CONTACT_DISTANCE, _ca_interface,
                                          _plddt_scale_factor, classify_chain)

    by_ca = contact['contact_atoms'] == 'ca'
    min_plddt = contact['min_plddt']
    kinds = {chain.get_id(): classify_chain(chain) for chain in model}
    plddt_scale = (_plddt_scale_factor(model)
                   if min_plddt is not None and not by_ca else None)

    def polymer_atoms(chain):
        return [atom for residue in chain if residue.id[0] == ' ' for atom in residue
                if plddt_scale is None or atom.get_bfactor() * plddt_scale >= min_plddt]

    interfaces = []
    for chain_i, chain_j in chain_pairs:
        if by_ca and kinds[chain_i] == 'protein' and kinds[chain_j] == 'protein':
            pair_pae = (pae_data if pae_data is not None and chain_i in pae_data['rows']
                        and chain_j in pae_data['rows'] else None)
            mask = _ca_interface(model, chain_i, chain_j, pair_pae, contact['distance'],
                                 min_plddt, contact['max_pae'])
            numbers = [[residue.id[1] for residue in model[chain]]
                       for chain in (chain_i, chain_j)]
            residues = [sorted({numbers[0][k] for k in np.flatnonzero(mask.any(axis=1))}),
                        sorted({numbers[1][k] for k in np.flatnonzero(mask.any(axis=0))})]
        else:
            atoms_i, atoms_j = polymer_atoms(model[chain_i]), polymer_atoms(model[chain_j])
            distance = NUCLEIC_ACID_CONTACT_DISTANCE if by_ca else contact['distance']
            close = cdist(np.array([atom.coord for atom in atoms_i]),
                          np.array([atom.coord for atom in atoms_j])) < distance
            residues = [sorted({atoms_i[k].get_parent().id[1]
                                for k in np.flatnonzero(close.any(axis=1))}),
                        sorted({atoms_j[k].get_parent().id[1]
                                for k in np.flatnonzero(close.any(axis=0))})]
        interfaces.append({'chains': [chain_i, chain_j],
                           'residues': [[int(n) for n in side] for side in residues]})
    return interfaces


def pack_pae(pae, max_rows=MAX_PAE_ROWS):
    '''
    A PAE matrix as base64 text, one byte per cell (value / PAE_STEP,
    rounded; values above 255 * PAE_STEP are capped). A matrix with more
    than max_rows rows is first averaged over square blocks of `factor`
    rows, so row r of the original is row r // factor of the packed one.

    Returns dict: data (base64), n (rows of the packed matrix), factor, step.
    '''
    pae = np.asarray(pae, dtype=np.float32)
    factor = int(np.ceil(pae.shape[0] / max_rows))
    if factor > 1:
        n = int(np.ceil(pae.shape[0] / factor))
        padded = np.full((n * factor, n * factor), np.nan, dtype=np.float32)
        padded[:pae.shape[0], :pae.shape[1]] = pae
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            pae = np.nanmean(padded.reshape(n, factor, n, factor), axis=(1, 3))
    packed = np.clip(np.rint(pae / PAE_STEP), 0, 255).astype(np.uint8)
    return {'data': base64.b64encode(packed.tobytes()).decode('ascii'),
            'n': int(packed.shape[0]), 'factor': factor, 'step': PAE_STEP}


def unpack_pae(packed):
    '''The matrix pack_pae() stored, in angstrom (the inverse, up to the rounding).'''
    values = np.frombuffer(base64.b64decode(packed['data']), dtype=np.uint8)
    return values.reshape(packed['n'], packed['n']).astype(np.float32) * packed['step']


def read_model_scores(structure_file):
    '''
    The whole-model scores the predictor wrote next to a model (Boltz:
    confidence_score, iptm, ptm, complex_plddt; AlphaFold3: ranking_score,
    iptm, ptm; ColabFold: iptm, ptm), as {name: value}; empty without a
    confidence file.
    '''
    from bioplexpy.analysis_funcs import find_interface_confidence_file

    source, tool = find_interface_confidence_file(structure_file)
    if source is None:
        return {}
    with open(source) as handle:
        data = json.load(handle)
    return {key: float(data[key]) for key in MODEL_SCORE_KEYS.get(tool, ())
            if isinstance(data.get(key), (int, float))}


def _clean(value):
    '''Make a value JSON-safe: numpy types to Python's, NaN and infinity to None.'''
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_clean(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _job_title(names):
    """What the models' names share, without a trailing model or rank number."""
    prefix = os.path.commonprefix(names) if len(names) > 1 else names[0]
    return re.sub(r'[_-]*(model|rank|unrelaxed_rank|relaxed_rank)?[_-]*\d*$', '', prefix) or names[0]


def _write_js(path, target, data):
    '''Write `window.BPV.<target> = <data>;` (script-tag data file).'''
    with open(path, 'w') as out:
        out.write('window.BPV = window.BPV || {models: {}};\n')
        out.write(f'window.BPV.{target} = ')
        json.dump(_clean(data), out, separators=(',', ':'))
        out.write(';\n')


def build_viewer_data(structure_files, names, rules, chain_to_uniprot, bp_293t_df,
                      bp_hct116_df, min_score=None, filter_score='ipsae_calc',
                      filter_reduce='max', title=None):
    '''
    Everything the page shows, for the models of one job.

    Parameters
    ----------
    structure_files: list of str
        The job's model files, best-ranked first; the first is the
        reference the others are superposed on.
    names: dict, structure file -> name (as in the per-model tables)
    rules: list of dict, one per contact rule the page can switch between,
        the first being the one it opens with. Each has: id, label, contact
        (dict from resolve_contact_settings()), tables (name -> table from
        compare_structure_contacts_to_BioPlex() under that rule) and
        summary (summarize_contacts() of those tables).
    chain_to_uniprot: dict, chain ID -> UniProt ID(s); the same for every model
    bp_293t_df, bp_hct116_df: BioPlex tables
    min_score, filter_score, filter_reduce: the score filter in use, if any
    title: str (optional)

    Returns
    -------
    (job, models): job is a dict for job.js, models a list of dicts, one
    per model, for model_<n>.js
    '''
    from bioplexpy.analysis_funcs import (SUGGESTED_MIN_SCORE, PDB_chains_to_uniprot,
                                          _bioplex_edges_and_roles,
                                          _direct_interaction_chain_pairs,
                                          _load_pdb_model, classify_chain,
                                          fetch_pdb_structure_file, read_pae)
    from bioplexpy.visualization_funcs import (_prepare_figure2_inputs,
                                               _protein_centroids,
                                               _write_rotated_structure)

    reference_file = structure_files[0]
    contact = rules[0]['contact']
    # chain map with the stand-in IDs for nucleic acid and unmapped chains,
    # colors, labels and the reference's display orientation: exactly what
    # the static figure uses
    figure = _prepare_figure2_inputs(
        reference_file, None, bp_293t_df, bp_hct116_df,
        interact_dist_threshold=contact['distance'], chain_to_uniprot=chain_to_uniprot,
        min_plddt=contact['min_plddt'], contact_atoms=contact['contact_atoms'],
        max_pae=contact['max_pae'])
    chain_ids_map, chain_types = figure['chain_to_uniprot'], figure['chain_types']

    reference_model = _load_pdb_model(reference_file, None)
    reference_ca = _protein_ca(reference_model)
    reference_sequences = _chain_sequences(reference_model)

    models, model_summaries, tools = [], [], set()
    with tempfile.TemporaryDirectory(prefix='bioplexpy_viewer_') as tmpdir:
        for structure_file in structure_files:
            name = names[structure_file]
            model = (reference_model if structure_file == reference_file
                     else _load_pdb_model(structure_file, None))
            if _chain_sequences(model) != reference_sequences:
                raise ValueError(f'{name} does not have the same chains and residues as '
                                 f'{names[reference_file]}: the viewer shows the models '
                                 'of one prediction')
            pae_data = None
            try:
                pae_data = read_pae(structure_file, model=model)
            except (ValueError, KeyError) as e:
                warnings.warn(f'PAE for {structure_file} not shown: {e}')
            if pae_data is not None:
                tools.add(pae_data['tool'])
            interfaces, edges = {}, {}
            for rule in rules:
                settings = rule['contact']
                rule_pae = pae_data if (settings['contact_atoms'] == 'ca'
                                        and (settings['min_plddt'] is not None
                                             or settings['max_pae'] is not None)) else None
                chain_pairs = _direct_interaction_chain_pairs(
                    model, settings['distance'], min_plddt=settings['min_plddt'],
                    contact_atoms=settings['contact_atoms'], max_pae=settings['max_pae'],
                    pae_data=rule_pae)
                interfaces[rule['id']] = interface_residues(model, chain_pairs, settings,
                                                            rule_pae)
                # every contact, nucleic acid and unmapped chains included
                edges[rule['id']] = PDB_chains_to_uniprot(chain_pairs, chain_ids_map)

            rotation, center, rmsd = common_frame(model, reference_ca,
                                                  figure['structure_rotation'],
                                                  figure['structure_center'])
            file_path, file_format = fetch_pdb_structure_file(structure_file, None)
            moved = os.path.join(tmpdir, 'moved')
            _write_rotated_structure(file_path, file_format, rotation, center, moved)
            with open(moved) as handle:
                structure_text = handle.read()

            # node positions: chain centroids as get_chain_centroids() takes
            # them (mean of the polymer atoms), in the common frame
            centroids = {}
            for chain in model:
                coords = [atom.coord for residue in chain if residue.id[0] == ' '
                          for atom in residue]
                if coords and classify_chain(chain) != 'other':
                    centroids[chain.get_id()] = rotation @ (np.mean(coords, axis=0) - center)
            node_ids, node_points = _protein_centroids(chain_ids_map, centroids)

            residues = {chain.get_id(): [residue.id[1] for residue in chain]
                        for chain in model if chain.get_id() in chain_types}
            rows, plddt = {}, {}
            if pae_data is not None:
                rows = {chain: pae_data['rows'][chain].tolist() for chain in pae_data['rows']}
                plddt = {chain: np.round(pae_data['plddt'][pae_data['rows'][chain]], 1)
                         for chain in pae_data['rows']}
            models.append({
                'name': name,
                'format': 'cif' if file_format == 'mmCif' else 'pdb',
                'structure': structure_text,
                'pae': pack_pae(pae_data['pae']) if pae_data is not None else None,
                'pae_rows': int(pae_data['pae'].shape[0]) if pae_data is not None else 0,
                'rows': rows, 'residues': residues, 'plddt': plddt,
                'interfaces': interfaces,
            })
            model_summaries.append({
                'name': name,
                'scores': read_model_scores(structure_file),
                'rmsd_to_first': round(rmsd, 2),
                'centroids': {node: np.round(point, 2)
                              for node, point in zip(node_ids, node_points)},
                'edges': edges,
            })

    names_in_order = [names[f] for f in structure_files]
    base = {'UniprotA', 'UniprotB', 'SymbolA', 'SymbolB', 'structure_contact',
            'passes_filter', 'bioplex_293T', 'bioplex_HCT116'}
    score_names = []
    for rule in rules:
        for table in rule['tables'].values():
            score_names += [c[:-3] for c in table.columns if c not in base
                            and c.endswith('_AB') and c[:-3] not in score_names]

    def pairs_of(summary):
        filtered = 'n_structures_pass' in summary
        pairs = []
        for row in summary.to_dict('records'):
            pair = {'a': row['UniprotA'], 'b': row['UniprotB'],
                    'n_contact': row['n_structures_contact'],
                    'bp293': bool(row['bioplex_293T']), 'bpHct': bool(row['bioplex_HCT116']),
                    'contact': [bool(row[name]) for name in names_in_order],
                    'scores': {score: [[row.get(f'{name}:{score}_AB'),
                                        row.get(f'{name}:{score}_BA')]
                                       for name in names_in_order] for score in score_names}}
            if filtered:
                pair['n_pass'] = row['n_structures_pass']
                pair['pass'] = [bool(row[f'{name}:pass']) for name in names_in_order]
            if 'reference_contact' in row:
                pair['reference'] = bool(row['reference_contact'])
            pairs.append(pair)
        return pairs

    protein_ids = [id_i for id_i in figure['all_ids'] if figure['id_type'][id_i] == 'protein']
    edges_293t, baits_293t, preys_293t = _bioplex_edges_and_roles(bp_293t_df)
    edges_hct116, baits_hct116, preys_hct116 = _bioplex_edges_and_roles(bp_hct116_df)
    nodes = [{'id': id_i, 'label': figure['labels'][id_i], 'type': figure['id_type'][id_i],
              'color': figure['node_color_palette'].get(id_i, '#b3b3b3'),
              'chains': sorted(c for c, ids in chain_ids_map.items() if id_i in ids),
              'bait293': id_i in baits_293t, 'prey293': id_i in preys_293t,
              'baitHct': id_i in baits_hct116, 'preyHct': id_i in preys_hct116}
             for id_i in figure['all_ids']]
    in_complex = set(protein_ids)
    bioplex_edges = [{'a': a, 'b': b, 'bp293': pair in edges_293t, 'bpHct': pair in edges_hct116}
                     for pair in sorted((edges_293t | edges_hct116), key=sorted)
                     if pair <= in_complex and len(pair) == 2
                     for a, b in [sorted(pair)]]
    job = {
        'title': title or _job_title(names_in_order),
        'tool': sorted(tools)[0] if len(tools) == 1 else None,
        'rules': [{'id': rule['id'], 'label': rule['label'], 'contact': rule['contact'],
                   'pairs': pairs_of(rule['summary'])} for rule in rules],
        'filter': {'min_score': min_score, 'score': filter_score, 'reduce': filter_reduce,
                   # where the page's cutoff slider starts when no --min-score was given
                   'suggested': SUGGESTED_MIN_SCORE.get(sorted(tools)[0]) if len(tools) == 1 else None},
        'score_names': score_names,
        'chains': [{'id': chain, 'type': chain_types[chain],
                    'color': figure['chain_color_palette'].get(chain, '#b3b3b3'),
                    'ids': chain_ids_map.get(chain, []),
                    'length': len(models[0]['residues'].get(chain, []))}
                   for chain in chain_types],
        'nodes': nodes,
        'bioplex_edges': bioplex_edges,
        'has_reference': 'reference_contact' in rules[0]['summary'],
        'models': model_summaries,
    }
    return job, models


# the contact rules the page offers besides the command line's: the presets, by name
RULE_LABELS = {'bioplex3d': 'BioPlex3D', 'bioplex2021': 'BioPlex 3.0 paper'}


def contact_rules(contact):
    """
    The contact rules the page can switch between: the one in use (first),
    then every preset of resolve_contact_settings() that differs from it.
    Returns a list of dict: id, label, contact.
    """
    from bioplexpy.analysis_funcs import CONTACT_PRESETS, resolve_contact_settings

    presets = {name: resolve_contact_settings(name) for name in CONTACT_PRESETS}
    same = [name for name, settings in presets.items() if settings == contact]
    rules = [{'id': same[0] if same else 'command_line',
              'label': RULE_LABELS.get(same[0], same[0]) if same else 'Command line',
              'contact': contact}]
    rules += [{'id': name, 'label': RULE_LABELS.get(name, name), 'contact': settings}
              for name, settings in presets.items() if name not in same]
    return rules


def write_viewer(structure_files, names, contacts_by_name, args, chain_to_uniprot,
                 bp_293t_df, bp_hct116_df, out_dir, reference=None, title=None,
                 reference_for_rule=None):
    """
    Write the viewer folder for the models of one job (see the module
    docstring). Called by `bioplexpy-structure --viewer` after the
    per-model tables are made.

    The page opens with the contact rule the tables were made with, and can
    switch to the other preset(s) (see contact_rules()); the contacts under
    those are worked out here, with the same functions.

    Parameters
    ----------
    structure_files: list of str (the models that were processed without error)
    names: dict, structure file -> name
    contacts_by_name: dict, name -> per-model contact table
    args: the CLI's parsed arguments (contact settings and score filter)
    chain_to_uniprot: dict, chain ID -> UniProt ID(s), the same for every model
    bp_293t_df, bp_hct116_df: BioPlex tables
    out_dir: str; the page goes to <out_dir>/viewer/
    reference: set of frozenset pairs (optional), an experimental structure's
        contacts under the rule in use
    title: str (optional)
    reference_for_rule: function (optional), contact settings -> set of
        frozenset pairs: the experimental structure's contacts under another rule

    Returns
    -------
    str: path of index.html
    """
    from bioplexpy.analysis_funcs import (PDB_to_interacting_chains_uniprot_maps,
                                          _read_interface_confidence_or_warn,
                                          compare_structure_contacts_to_BioPlex,
                                          resolve_min_score)
    from bioplexpy.cli import summarize_contacts

    contact = {'contact_atoms': args.contact_atoms, 'distance': args.distance,
               'min_plddt': args.min_plddt, 'max_pae': args.max_pae}
    rules = contact_rules(contact)
    rules[0]['tables'] = {names[f]: contacts_by_name[names[f]] for f in structure_files}
    rules[0]['summary'] = summarize_contacts(rules[0]['tables'], bp_293t_df, bp_hct116_df,
                                             reference)
    # the scores do not depend on the contact rule: read (or compute) them once per model
    confidence = {}
    for rule in rules[1:]:
        settings, rule['tables'] = rule['contact'], {}
        for structure_file in structure_files:
            maps = PDB_to_interacting_chains_uniprot_maps(
                structure_file, None, settings['distance'], chain_to_uniprot=chain_to_uniprot,
                min_plddt=settings['min_plddt'], contact_atoms=settings['contact_atoms'],
                max_pae=settings['max_pae'])
            if structure_file not in confidence:
                confidence[structure_file] = _read_interface_confidence_or_warn(
                    structure_file, chain_ids=list(maps[2]),
                    compute_scores=args.compute_scores)
            rule['tables'][names[structure_file]] = compare_structure_contacts_to_BioPlex(
                *maps, bp_293t_df, bp_hct116_df,
                interface_confidence=confidence[structure_file])
        rule_reference = (reference_for_rule(settings)
                          if reference is not None and reference_for_rule else None)
        rule['summary'] = summarize_contacts(rule['tables'], bp_293t_df, bp_hct116_df,
                                             rule_reference)
    min_score = None
    if args.min_score is not None:
        # 'suggested' depends on the predictor, which the first model's scores name
        min_score = resolve_min_score(
            args.min_score,
            _read_interface_confidence_or_warn(structure_files[0], compute_scores=False),
            args.filter_score)
    job, models = build_viewer_data(
        structure_files, names, rules, chain_to_uniprot, bp_293t_df, bp_hct116_df,
        min_score=min_score, filter_score=args.filter_score,
        filter_reduce=args.filter_reduce, title=title)

    viewer_dir = os.path.join(out_dir, 'viewer')
    os.makedirs(viewer_dir, exist_ok=True)
    for asset in ASSET_FILES:
        shutil.copyfile(os.path.join(ASSET_DIR, asset), os.path.join(viewer_dir, asset))
    _write_js(os.path.join(viewer_dir, 'job.js'), 'job', job)
    for index, model in enumerate(models):
        _write_js(os.path.join(viewer_dir, f'model_{index}.js'), f'models[{index}]', model)
    return os.path.join(viewer_dir, 'index.html')
