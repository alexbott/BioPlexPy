#!/usr/bin/env python
'''
Command-line entry point for comparing a user's own structure files
(experimental models, or AlphaFold3/Boltz/ColabFold predictions) with
BioPlex AP-MS data. Installed as `bioplexpy-structure`; also runnable as
`python -m bioplexpy.cli`.

For each structure file this writes, into --out-dir:
  <name>_contacts.tsv   structure contacts vs BioPlex 293T/HCT116 edges, plus
                        the predictor's own chain-pair scores for predicted
                        models (AF3/Boltz pair ipTM; ipSAE/pDockQ/pDockQ2 if
                        a ColabFold scores file holds them); with
                        --min-score, a passes_filter column
  <name>_chain_map.tsv  which UniProt protein each chain was assigned
  <name>_figure.png     the Figure 2-style panels (unless --no-render); a
                        predicted model's chain-pair scores are shown as edge
                        opacity by default; with several --confidence-style
                        values, one figure each: <name>_figure_<style>.png
  <name>_structure.html interactive py3Dmol view (--interactive only)
and, when two or more structures are processed:
  summary_contacts.tsv  every protein pair, how many of the models have it
                        as a contact, per-model columns, BioPlex columns,
                        and (with --reference) the experimental structure;
                        per-model score columns named <model>:<score>_AB/_BA;
                        with --min-score, n_structures_pass and <model>:pass
'''

import argparse
import json
import os
import re
import sys
import tempfile
import traceback

import pandas as pd



def _collect_structure_files(paths, zip_extract_root, include_pae=False):
    '''
    Expand each path into its model files with find_structure_files()
    (which recognizes AF3-server, ColabFold and Boltz output layouts),
    printing what was detected. Returns the files and the set of distinct
    prediction inputs seen (Boltz predictions/<input> folder names).
    '''
    from bioplexpy.analysis_funcs import find_structure_files
    files, inputs, display = [], set(), {}
    for i, path in enumerate(paths):
        # each .zip gets its own folder under this run's temporary directory
        extract_dir = os.path.join(zip_extract_root, f'zip{i}')
        found, notes, groups = find_structure_files(path, extract_dir=extract_dir,
                                                    include_pae=include_pae)
        for f in found:
            # files extracted from a .zip are shown as <zip>:<member>, since
            # the temporary copy is gone once the run ends
            display[os.path.abspath(f)] = (
                f'{path}:{os.path.relpath(f, extract_dir)}'
                if os.path.abspath(f).startswith(os.path.abspath(extract_dir) + os.sep)
                else f)
        for note in notes:
            print(note, file=sys.stderr)
        files.extend(found)
        inputs.update(os.path.basename(g) for g in groups)
    # the same file given twice (e.g. a folder and a file inside it)
    files = list(dict.fromkeys(os.path.abspath(f) for f in files))
    return files, inputs, display


def _read_chain_map(pairs, map_file):
    '''
    Build a chain -> UniProt dict from `A=P61158` pairs and/or a file:
    JSON ({"A": "P61158", ...}) or a two-column CSV/TSV (chain, uniprot;
    a header row is optional).
    '''
    chain_map = {}
    if map_file:
        if map_file.lower().endswith('.json'):
            with open(map_file) as f:
                chain_map.update(json.load(f))
        else:
            sep = '\t' if map_file.lower().endswith(('.tsv', '.txt')) else ','
            df = pd.read_csv(map_file, sep=sep, header=None, dtype=str,
                             comment='#').dropna(how='all')
            if df.iloc[0, 0].strip().lower() == 'chain':
                df = df.iloc[1:]
            for chain_id, uniprot_id in zip(df.iloc[:, 0], df.iloc[:, 1]):
                chain_map[chain_id.strip()] = uniprot_id.strip()
    for pair in pairs or []:
        if '=' not in pair:
            raise ValueError(f"--chain-map entries look like A=P61158, got '{pair}'")
        chain_id, uniprot_id = pair.split('=', 1)
        chain_map[chain_id.strip()] = uniprot_id.strip()
    return chain_map or None


def _read_uniprots(ids, ids_file):
    uniprots = list(ids or [])
    if ids_file:
        with open(ids_file) as f:
            uniprots += [tok for line in f for tok in line.replace(',', ' ').split()
                         if not line.startswith('#')]
    return uniprots or None


def _unique_names(structure_files):
    '''
    Output name per file: its stem, with just enough parent folder names
    prepended to tell apart files that share a stem (e.g. two Boltz runs
    of the same input both produce arp23_model_0.cif), so no file's
    outputs overwrite another's.
    '''
    from bioplexpy.analysis_funcs import structure_label
    parts = {f: os.path.normpath(os.path.abspath(f)).split(os.sep)[:-1]
             for f in structure_files}
    depth = {f: 0 for f in structure_files}

    def name(f):
        prefix = parts[f][len(parts[f]) - depth[f]:] if depth[f] else []
        return '_'.join(prefix + [structure_label(f)])

    while True:
        by_name = {}
        for f in structure_files:
            by_name.setdefault(name(f), []).append(f)
        clashes = [fs for fs in by_name.values() if len(fs) > 1]
        if not clashes:
            return {f: re.sub(r'[^A-Za-z0-9_.-]', '_', name(f)) for f in structure_files}
        grew = False
        for fs in clashes:
            for f in fs:
                if depth[f] < len(parts[f]):
                    depth[f] += 1
                    grew = True
        if not grew:
            raise ValueError(f'Cannot give distinct names to {clashes[0]}')


def process_structure(structure_file, name, args, chain_map, uniprots,
                      bp_293t_df, bp_hct116_df):
    '''Run one structure file through mapping, contact table, and rendering.'''
    import matplotlib.pyplot as plt

    from bioplexpy.analysis_funcs import (PDB_to_interacting_chains_uniprot_maps,
                                          compare_structure_contacts_to_BioPlex,
                                          _read_interface_confidence_or_warn,
                                          filter_contacts_by_score,
                                          map_chains_to_uniprot,
                                          resolve_min_score)
    from bioplexpy.visualization_funcs import (get_edge_confidence_scores,
                                               render_figure2_panels,
                                               render_figure2_panels_static)

    # chain assignment: explicit mapping wins; otherwise match by sequence
    # (per file, since different tools can name the same chains differently)
    if chain_map is not None:
        file_chain_map = chain_map
        map_report = pd.DataFrame({'chain': list(chain_map),
                                   'uniprot': [v if isinstance(v, str) else ';'.join(v)
                                               for v in chain_map.values()],
                                   'method': 'user', 'accepted': True})
    else:
        file_chain_map, map_report = map_chains_to_uniprot(
            structure_file, uniprots, min_identity=args.min_identity,
            min_coverage=args.min_coverage)
    map_report.to_csv(os.path.join(args.out_dir, f'{name}_chain_map.tsv'),
                      sep='\t', index=False)

    maps = PDB_to_interacting_chains_uniprot_maps(
        structure_file, None, args.distance, chain_to_uniprot=file_chain_map,
        min_plddt=args.min_plddt, contact_atoms=args.contact_atoms, max_pae=args.max_pae)
    # the predictor's own chain-pair scores, if it wrote any next to the model
    # (with --compute-scores, plus scores calculated here from the PAE)
    interface_confidence = _read_interface_confidence_or_warn(
        structure_file, chain_ids=list(maps[2]), compute_scores=args.compute_scores)
    # opt-in (--min-score): the table keeps every contact and says which
    # ones the filter rejected; the figure leaves those out
    # ('suggested' becomes the cutoff for this model's predictor)
    filter_status = None
    min_score = resolve_min_score(args.min_score, interface_confidence, args.filter_score)
    if min_score is not None:
        _, filter_status = filter_contacts_by_score(
            maps[1], maps[0], interface_confidence, score=args.filter_score,
            min_score=min_score, reduce=args.filter_reduce)
    contacts_df = compare_structure_contacts_to_BioPlex(
        *maps, bp_293t_df, bp_hct116_df, interface_confidence=interface_confidence,
        filter_status=filter_status)
    contacts_df.to_csv(os.path.join(args.out_dir, f'{name}_contacts.tsv'),
                       sep='\t', index=False)

    if not args.no_render:
        # one figure per confidence style, but only if there's a score to show
        styles = [None if style == 'none' else style for style in args.confidence_style]
        edge_scores, _ = get_edge_confidence_scores(
            interface_confidence, maps[0], score=args.confidence_score,
            reduce=args.confidence_reduce)
        if edge_scores is None:
            styles = [None]
        for style in dict.fromkeys(styles):
            render_kwargs = dict(interact_dist_threshold=args.distance,
                                 chain_to_uniprot=file_chain_map,
                                 min_plddt=args.min_plddt, confidence_style=style,
                                 confidence_score=args.confidence_score,
                                 confidence_reduce=args.confidence_reduce,
                                 interface_confidence=interface_confidence,
                                 contact_atoms=args.contact_atoms,
                                 max_pae=args.max_pae,
                                 filter_score=args.filter_score,
                                 min_score=min_score,
                                 filter_reduce=args.filter_reduce)
            if args.interactive:
                fig, view = render_figure2_panels(structure_file, None, bp_293t_df,
                                                  bp_hct116_df, **render_kwargs)
                view.write_html(os.path.join(args.out_dir, f'{name}_structure.html'))
            else:
                fig = render_figure2_panels_static(structure_file, None, bp_293t_df,
                                                   bp_hct116_df, **render_kwargs)
            suffix = f'_{style or "plain"}' if len(set(styles)) > 1 else ''
            fig.savefig(os.path.join(args.out_dir, f'{name}_figure{suffix}.png'),
                        dpi=args.dpi, bbox_inches='tight')
            plt.close(fig)

    # .astype(bool): a table with no rows has an untyped structure_contact
    # column, and indexing with it would select columns, not rows
    contacts = contacts_df[contacts_df.structure_contact.astype(bool)]
    n_unmapped = (len(map_report) - int(map_report.accepted.sum())
                  if 'accepted' in map_report else 0)
    summary = (f'{len(file_chain_map)} chains mapped'
               + (f' ({n_unmapped} unmapped)' if n_unmapped else '')
               + f'; {len(contacts)} protein contacts, '
               f'{int(contacts.bioplex_293T.sum())} seen in 293T, '
               f'{int(contacts.bioplex_HCT116.sum())} in HCT116')
    if min_score is not None:
        n_hidden = int((contacts.passes_filter == False).sum())  # noqa: E712
        summary += (f'; {n_hidden} hidden by {args.filter_score} < {min_score:g}, '
                    f'{int(contacts.passes_filter.isna().sum())} unscored (kept)')
    if interface_confidence is not None:
        summary += (f"; {interface_confidence['tool']} scores "
                    f"({', '.join(interface_confidence['scores'])}) from "
                    f"{os.path.basename(interface_confidence['source'])}")
    return summary, contacts_df


def reference_contacts(reference, args, chain_map, uniprots):
    '''
    Direct protein contacts (as frozensets of UniProt IDs) in a reference
    structure -- an RCSB PDB ID (mapped via SIFTS) or a local file (mapped
    with the same --chain-map/--uniprots as the models).
    '''
    from bioplexpy.analysis_funcs import (PDB_to_interacting_chains_uniprot_maps,
                                          is_local_structure_file,
                                          map_chains_to_uniprot)
    ref_chain_map = None
    if is_local_structure_file(reference):
        ref_chain_map = (chain_map if chain_map is not None
                         else map_chains_to_uniprot(reference, uniprots)[0])
    # a PDB ID is downloaded to a throwaway directory, not into --out-dir
    with tempfile.TemporaryDirectory() as download_dir:
        chain_to_uniprot, interacting, chain_types = PDB_to_interacting_chains_uniprot_maps(
            reference, download_dir, args.distance, chain_to_uniprot=ref_chain_map,
            min_plddt=args.min_plddt, contact_atoms=args.contact_atoms,
            max_pae=args.max_pae)
    protein_ids = {id_i for chain_id, ids in chain_to_uniprot.items()
                   if chain_types.get(chain_id) == 'protein' for id_i in ids}
    return {frozenset(pair) for pair in interacting if set(pair) <= protein_ids}


def summarize_contacts(contacts_by_name, bp_293t_df, bp_hct116_df, reference=None):
    '''
    Combine per-structure contact tables (from
    compare_structure_contacts_to_BioPlex()) into one table: a row per
    protein pair seen in any of them, with the number and fraction of
    structures in which it is a direct contact, one True/False column per
    structure, the BioPlex columns, and -- if `reference` (a set of
    frozenset pairs) is given -- whether the reference structure has it.
    Any chain-pair score columns in the per-structure tables are carried
    over per structure, as <name>:<score>_AB/_BA (oriented to this
    table's UniprotA/UniprotB).

    If the tables have a passes_filter column (--min-score), the contact
    counts stay as they are, and n_structures_pass /
    fraction_structures_pass count the structures where the pair is a
    contact the filter did not reject (an unscored contact counts), with
    one <name>:pass column per structure.
    '''
    base_columns = {'UniprotA', 'UniprotB', 'SymbolA', 'SymbolB', 'structure_contact',
                    'passes_filter', 'bioplex_293T', 'bioplex_HCT116'}
    filtered = any('passes_filter' in df for df in contacts_by_name.values())
    rows, score_columns = {}, []
    for name, df in contacts_by_name.items():
        scores = [c for c in df.columns if c not in base_columns]
        for rec in df.to_dict('records'):
            key = frozenset((rec['UniprotA'], rec['UniprotB']))
            row = rows.setdefault(key, dict(
                UniprotA=rec['UniprotA'], UniprotB=rec['UniprotB'],
                SymbolA=rec['SymbolA'], SymbolB=rec['SymbolB'],
                bioplex_293T=rec['bioplex_293T'], bioplex_HCT116=rec['bioplex_HCT116']))
            row[name] = bool(rec['structure_contact'])
            # passes_filter is True, False or empty: only False is a rejection
            row[f'{name}:pass'] = row[name] and not rec.get('passes_filter') == False  # noqa: E712
            flipped = rec['UniprotA'] != row['UniprotA']
            for column in scores:
                target = column
                if flipped and column.endswith(('_AB', '_BA')):
                    target = column[:-2] + column[-2:][::-1]
                row[f'{name}:{target}'] = rec[column]
        score_columns += [f'{name}:{c}' for c in scores]
    if reference is not None and reference - set(rows):
        # pairs only the reference has as a contact (and no model has as a
        # contact or BioPlex edge) still need symbols and BioPlex status
        from bioplexpy.analysis_funcs import (_bioplex_edges_and_roles,
                                              _bioplex_symbol_lookup)
        symbols = _bioplex_symbol_lookup(bp_293t_df, bp_hct116_df)
        edges_293t = _bioplex_edges_and_roles(bp_293t_df)[0]
        edges_hct116 = _bioplex_edges_and_roles(bp_hct116_df)[0]
        for key in reference - set(rows):
            uniprot_A, uniprot_B = sorted(key)
            rows[key] = dict(UniprotA=uniprot_A, UniprotB=uniprot_B,
                             SymbolA=symbols.get(uniprot_A, uniprot_A),
                             SymbolB=symbols.get(uniprot_B, uniprot_B),
                             bioplex_293T=key in edges_293t,
                             bioplex_HCT116=key in edges_hct116)

    names = list(contacts_by_name)
    # the columns are named so that a table with no rows (no structure has
    # a contact or a BioPlex interaction) still has them
    summary = pd.DataFrame(list(rows.values())) if rows else pd.DataFrame(
        columns=['UniprotA', 'UniprotB', 'SymbolA', 'SymbolB', 'bioplex_293T',
                 'bioplex_HCT116'])
    for name in names:
        # pairs missing from a structure's table are not contacts there
        summary[name] = summary[name].eq(True) if name in summary else False
    summary['n_structures_contact'] = summary[names].sum(axis=1).astype(int)
    summary['fraction_structures_contact'] = (summary.n_structures_contact
                                              / len(names)).round(3)
    columns = ['UniprotA', 'UniprotB', 'SymbolA', 'SymbolB', 'n_structures_contact',
               'fraction_structures_contact']
    if reference is not None:
        summary['reference_contact'] = [frozenset((a, b)) in reference for a, b in
                                        zip(summary.UniprotA, summary.UniprotB)]
        columns.append('reference_contact')
    pass_columns = [f'{name}:pass' for name in names] if filtered else []
    if filtered:
        for column in pass_columns:
            summary[column] = summary[column].eq(True) if column in summary else False
        summary['n_structures_pass'] = summary[pass_columns].sum(axis=1).astype(int)
        summary['fraction_structures_pass'] = (summary.n_structures_pass
                                               / len(names)).round(3)
        columns += ['n_structures_pass', 'fraction_structures_pass']
    columns += ['bioplex_293T', 'bioplex_HCT116'] + names + pass_columns
    for column in score_columns:
        if column not in summary:
            summary[column] = float('nan')
    columns += score_columns
    return (summary[columns]
            .sort_values(['n_structures_contact', 'SymbolA', 'SymbolB'],
                         ascending=[False, True, True])
            .reset_index(drop=True))


# marks a contact option left to --contact-preset (see resolve_contact_settings())
FROM_PRESET = 'preset'


def _number_or_none(text):
    '''argparse type: a number, or 'none'/'off' to switch a condition off.'''
    if text == FROM_PRESET:     # argparse passes a string default through here
        return text
    if text.lower() in ('none', 'off'):
        return None
    return float(text)


def build_parser():
    parser = argparse.ArgumentParser(
        prog='bioplexpy-structure',
        description='Compare your own structure files (experimental or predicted) '
                    'with BioPlex AP-MS interactions in 293T and HCT116 cells.',
        epilog='Put the structure paths first: options that take a list '
               '(--uniprots, --chain-map) would otherwise read them as list items. '
               'Examples: bioplexpy-structure models/ --uniprots P61158 P61160 '
               '--contact-preset bioplex2021 | bioplexpy-structure fold_my_complex.zip --uniprots '
               'P61158 P61160 (an AlphaFold Server download, used as it is)')
    parser.add_argument('structures', nargs='+',
                        help='.pdb/.ent/.cif/.mmcif files, directories of them, or '
                             '.zip archives. The .zip the AlphaFold Server downloads '
                             'can be given directly, without unpacking (its models and '
                             'confidence files are extracted to a temporary folder; '
                             'the zip itself is not changed). '
                             'Raw predictor output folders can be given as-is: '
                             'AlphaFold3 server and ColabFold models are read from '
                             'the top of the folder (templates are ignored), Boltz '
                             'models from boltz_results_*/predictions/*/; other '
                             'folders are not searched below their top level')

    mapping = parser.add_argument_group(
        'chain assignment (give an explicit mapping, or UniProt IDs to match by sequence)')
    mapping.add_argument('--chain-map', nargs='+', metavar='CHAIN=UNIPROT',
                         help='e.g. --chain-map A=P61158 B=P61160')
    mapping.add_argument('--chain-map-file', metavar='FILE',
                         help='JSON dict, or two-column CSV/TSV of chain, uniprot')
    mapping.add_argument('--uniprots', nargs='+', metavar='ID',
                         help='UniProt IDs of the complex members; each chain is '
                              'matched to one by sequence')
    mapping.add_argument('--uniprots-file', metavar='FILE',
                         help='UniProt IDs, whitespace/comma separated')
    mapping.add_argument('--min-identity', type=float, default=0.9,
                         help='sequence matching: minimum identity (default 0.9)')
    mapping.add_argument('--min-coverage', type=float, default=0.5,
                         help='sequence matching: minimum chain coverage (default 0.5)')

    contact = parser.add_argument_group(
        'direct contacts (what counts as two chains touching)',
        'By default the parameters of the BioPlex3D pipeline: CA atoms closer than '
        '8 A, both residues with pLDDT >= 50, and PAE <= 10 in at least one '
        'direction. The pLDDT and PAE conditions need the PAE file next to the '
        'model; a structure without one (e.g. an experimental structure) is judged '
        'on the distance alone. With CA atoms, contacts with a nucleic acid chain use '
        'any atom closer than 6 A.')
    contact.add_argument('--contact-preset', choices=['bioplex3d', 'bioplex2021'],
                         default='bioplex3d',
                         help='bioplex3d (default): the parameters above. bioplex2021: '
                              'the rule of the BioPlex 3.0 paper (Huttlin et al. 2021) '
                              'that BioPlexPy used before -- any two atoms closer than '
                              '6 A, no pLDDT or PAE condition. The options below replace '
                              'single parameters of the preset')
    contact.add_argument('--contact-atoms', choices=['ca', 'any'], default=None,
                         help='atoms compared: ca (one CA per residue) or any (every atom)')
    contact.add_argument('--distance', type=float, default=None,
                         help='distance cutoff in Angstroms (preset: 8, or 6 for '
                              'bioplex2021)')
    contact.add_argument('--min-plddt', type=_number_or_none, default=FROM_PRESET,
                         metavar='VALUE',
                         help="minimum pLDDT, 0-100 (preset: 50, or none for bioplex2021); "
                              "'none' switches it off. With --contact-atoms ca both "
                              'residues of a pair must reach it; with any, atoms below it '
                              'are ignored')
    contact.add_argument('--max-pae', type=_number_or_none, default=FROM_PRESET,
                         metavar='VALUE',
                         help="maximum PAE in at least one direction (preset: 10, or none "
                              "for bioplex2021); 'none' switches it off. Only used with "
                              '--contact-atoms ca')

    analysis = parser.add_argument_group('analysis')
    analysis.add_argument('--bioplex-293t-version', default='3.0')
    analysis.add_argument('--bioplex-hct116-version', default='1.0')

    confidence = parser.add_argument_group(
        'interface confidence (predicted models; how scores are shown)')
    confidence.add_argument('--confidence-style', nargs='+', default=['alpha'],
                            choices=['width', 'alpha', 'color', 'none'],
                            help="how the model network's edges show the predictor's "
                                 'chain-pair score: opacity (default), line width, or '
                                 'viridis color; several give one figure each, named '
                                 '<name>_figure_<style>.png (e.g. --confidence-style '
                                 'alpha width color to compare); none = plain edges')
    confidence.add_argument('--confidence-score', default='pair_iptm',
                            help='score to show: pair_iptm (AlphaFold3/Boltz; default), '
                                 'or ipsae/pdockq/pdockq2 if a ColabFold scores file '
                                 'has them, or a computed one (see --compute-scores). '
                                 'All scores found go in the TSVs regardless')
    confidence.add_argument('--compute-scores', action='store_true',
                            help='also calculate interface scores from each model and '
                                 'its PAE file, as the BioPlex3D pipeline defines them: '
                                 'pdockq_calc, pdockq2_calc, lis, clis, ilis, plus '
                                 'ipsae_calc, lia/clia (pair counts) and clashes. '
                                 'Protein chain pairs only; needs the PAE file next to '
                                 'the model (AlphaFold3 full_data, Boltz pae_*.npz, '
                                 'ColabFold scores JSON)')
    confidence.add_argument('--confidence-reduce', default='mean',
                            choices=['mean', 'min', 'max'],
                            help='for scores that differ by direction (Boltz ipTM, '
                                 'ipSAE, pDockQ2): combine A->B and B->A by mean '
                                 '(default), min or max for the figure; the TSVs '
                                 'keep both')

    filtering = parser.add_argument_group(
        'confidence filter (predicted models; off unless --min-score is given)')
    filtering.add_argument('--min-score', type=_min_score, default=None, metavar='VALUE',
                           help='hide contacts whose --filter-score is below VALUE: they '
                                'are left out of the figure, and marked passes_filter = '
                                'False in the tables (structure_contact is unchanged). A '
                                'contact with no score is kept. VALUE is a number, or '
                                '"suggested" for the suggested ipsae_calc cutoff of the '
                                "model's predictor: 0.3 for Boltz, 0.2 for AlphaFold3 "
                                '(none for ColabFold). In our calibration these remove '
                                'most, not all, contacts called with an unrelated protein, '
                                'and about 2 real contacts in 25')
    filtering.add_argument('--filter-score', default=None, metavar='NAME',
                           help='score the filter tests (default ipsae_calc, which '
                                'switches --compute-scores on); any score column works, '
                                'e.g. pair_iptm')
    filtering.add_argument('--filter-reduce', choices=['max', 'mean', 'min'], default=None,
                           help='for a score that differs by direction, the value tested: '
                                'max (default; either direction is enough), mean or min. '
                                'Provisional -- to be confirmed with the BioPlex3D authors')

    output = parser.add_argument_group('output')
    output.add_argument('--out-dir', default='bioplexpy_structure_out',
                        help='output directory (default ./bioplexpy_structure_out)')
    output.add_argument('--no-render', action='store_true',
                        help='write the TSV tables only, no figure')
    output.add_argument('--interactive', action='store_true',
                        help='write an interactive py3Dmol HTML view plus a 3-panel '
                             'network PNG, instead of the static 4-panel PNG '
                             '(which needs pymol-open-source)')
    output.add_argument('--dpi', type=int, default=150)
    output.add_argument('--reference', metavar='PDB_ID_OR_FILE',
                        help='experimental structure to compare against in '
                             'summary_contacts.tsv, e.g. --reference 6YW7')
    return parser


def _min_score(value):
    '''--min-score: a number, or the word "suggested".'''
    if value == 'suggested':
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'a number or "suggested", not {value!r}')


def _resolve_filter_args(parser, args):
    '''Fill in the filter defaults, and switch on what the options need.'''
    from bioplexpy.analysis_funcs import COMPUTED_SCORES, resolve_contact_settings

    if args.min_score is None and (args.filter_score or args.filter_reduce):
        parser.error('--filter-score/--filter-reduce only apply with --min-score')
    args.filter_score = args.filter_score or 'ipsae_calc'
    args.filter_reduce = args.filter_reduce or 'max'
    if args.min_score == 'suggested' and args.filter_score != 'ipsae_calc':
        parser.error('--min-score suggested is defined for ipsae_calc only; give a number '
                     f'with --filter-score {args.filter_score}')
    if args.min_score is not None and args.filter_score in COMPUTED_SCORES:
        args.compute_scores = True
    # contact parameters: the preset's, unless given
    max_pae_given = args.max_pae != FROM_PRESET
    contact = resolve_contact_settings(
        args.contact_preset, contact_atoms=args.contact_atoms or FROM_PRESET,
        distance=FROM_PRESET if args.distance is None else args.distance,
        min_plddt=args.min_plddt, max_pae=args.max_pae)
    args.contact_atoms, args.distance = contact['contact_atoms'], contact['distance']
    args.min_plddt, args.max_pae = contact['min_plddt'], contact['max_pae']
    if args.contact_atoms == 'any' and args.max_pae is not None:
        if max_pae_given:
            parser.error('--max-pae needs --contact-atoms ca')
        args.max_pae = None     # a preset's PAE condition only goes with CA atoms


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _resolve_filter_args(parser, args)

    chain_map = _read_chain_map(args.chain_map, args.chain_map_file)
    uniprots = _read_uniprots(args.uniprots, args.uniprots_file)
    if chain_map is None and uniprots is None:
        parser.error('give a chain mapping (--chain-map/--chain-map-file) or the '
                     "complex's UniProt IDs (--uniprots/--uniprots-file)")

    # .zip archives are extracted here for the run and removed afterwards
    with tempfile.TemporaryDirectory(prefix='bioplexpy_zip_') as zip_extract_root:
        return _run(parser, args, chain_map, uniprots, zip_extract_root)


def _run(parser, args, chain_map, uniprots, zip_extract_root):
    structure_files, prediction_inputs, display = _collect_structure_files(
        args.structures, zip_extract_root, include_pae=(args.compute_scores
                     or (args.contact_atoms == 'ca'
                         and (args.min_plddt is not None or args.max_pae is not None))))
    if not structure_files:
        parser.error('no structure files found')
    # one Boltz run over several input files holds several different
    # complexes; a single across-structure summary of them is meaningless
    summarize = len(prediction_inputs) <= 1
    if not summarize:
        print(f'warning: these outputs hold {len(prediction_inputs)} different '
              f'prediction inputs ({", ".join(sorted(prediction_inputs))}); each is '
              'processed, but no combined summary_contacts.tsv is written -- run '
              'each input folder separately for one.', file=sys.stderr)

    import matplotlib
    matplotlib.use('Agg')

    from bioplexpy.data_import_funcs import getBioPlex

    os.makedirs(args.out_dir, exist_ok=True)
    print('Loading BioPlex data...', file=sys.stderr)
    bp_293t_df = getBioPlex('293T', args.bioplex_293t_version)
    bp_hct116_df = getBioPlex('HCT116', args.bioplex_hct116_version)

    failures = 0
    names = _unique_names(structure_files)
    contacts_by_name = {}
    for structure_file in structure_files:
        name = names[structure_file]
        try:
            summary, contacts_df = process_structure(structure_file, name, args, chain_map,
                                        uniprots, bp_293t_df, bp_hct116_df)
            print(f'{display[structure_file]}: {summary}')
            contacts_by_name[name] = contacts_df
        except Exception as e:
            failures += 1
            print(f'{display[structure_file]}: FAILED -- {e}', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    if summarize and (len(contacts_by_name) >= 2 or (contacts_by_name and args.reference)):
        reference = None
        if args.reference:
            try:
                reference = reference_contacts(args.reference, args, chain_map, uniprots)
            except Exception as e:
                failures += 1
                print(f'reference {args.reference}: FAILED -- {e}', file=sys.stderr)
        summarize_contacts(contacts_by_name, bp_293t_df, bp_hct116_df, reference).to_csv(
            os.path.join(args.out_dir, 'summary_contacts.tsv'), sep='\t', index=False)

    print(f'Wrote results for {len(structure_files) - failures}/{len(structure_files)} '
          f'structure(s) to {args.out_dir}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
