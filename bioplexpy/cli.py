#!/usr/bin/env python
'''
Command-line entry point for comparing a user's own structure files
(experimental models, or AlphaFold3/Boltz/ColabFold predictions) with
BioPlex AP-MS data. Installed as `bioplexpy-structure`; also runnable as
`python -m bioplexpy.cli`.

For each structure file this writes, into --out-dir:
  <name>_contacts.tsv   structure contacts vs BioPlex 293T/HCT116 edges
  <name>_chain_map.tsv  which UniProt protein each chain was assigned
  <name>_figure.png     the Figure 2-style panels (unless --no-render)
  <name>_structure.html interactive py3Dmol view (--interactive only)
'''

import argparse
import glob
import json
import os
import sys
import traceback

import pandas as pd

_STRUCTURE_EXTS = ('.pdb', '.ent', '.cif', '.mmcif')


def _collect_structure_files(paths):
    '''Expand directories (non-recursively) into their structure files.'''
    files = []
    for path in paths:
        if os.path.isdir(path):
            found = sorted(f for f in glob.glob(os.path.join(path, '*'))
                           if f.lower().endswith(_STRUCTURE_EXTS))
            if not found:
                print(f'warning: no {"/".join(_STRUCTURE_EXTS)} files in {path}',
                      file=sys.stderr)
            files.extend(found)
        else:
            files.append(path)
    return files


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


def _unique_name(structure_file, used):
    '''File stem, prefixed with its directory name if the stem repeats.'''
    from bioplexpy.analysis_funcs import structure_label
    name = structure_label(structure_file)
    if name in used:
        parent = os.path.basename(os.path.dirname(os.path.abspath(structure_file)))
        name = f'{parent}_{name}'
    used.add(name)
    return name


def process_structure(structure_file, name, args, chain_map, uniprots,
                      bp_293t_df, bp_hct116_df):
    '''Run one structure file through mapping, contact table, and rendering.'''
    import matplotlib.pyplot as plt

    from bioplexpy.analysis_funcs import (PDB_to_interacting_chains_uniprot_maps,
                                          compare_structure_contacts_to_BioPlex,
                                          map_chains_to_uniprot)
    from bioplexpy.visualization_funcs import (render_figure2_panels,
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
        min_plddt=args.min_plddt)
    contacts_df = compare_structure_contacts_to_BioPlex(*maps, bp_293t_df, bp_hct116_df)
    contacts_df.to_csv(os.path.join(args.out_dir, f'{name}_contacts.tsv'),
                       sep='\t', index=False)

    if not args.no_render:
        render_kwargs = dict(interact_dist_threshold=args.distance,
                             chain_to_uniprot=file_chain_map, min_plddt=args.min_plddt)
        if args.interactive:
            fig, view = render_figure2_panels(structure_file, None, bp_293t_df,
                                              bp_hct116_df, **render_kwargs)
            view.write_html(os.path.join(args.out_dir, f'{name}_structure.html'))
        else:
            fig = render_figure2_panels_static(structure_file, None, bp_293t_df,
                                               bp_hct116_df, **render_kwargs)
        fig.savefig(os.path.join(args.out_dir, f'{name}_figure.png'), dpi=args.dpi,
                    bbox_inches='tight')
        plt.close(fig)

    contacts = contacts_df[contacts_df.structure_contact]
    n_unmapped = (len(map_report) - int(map_report.accepted.sum())
                  if 'accepted' in map_report else 0)
    return (f'{len(file_chain_map)} chains mapped'
            + (f' ({n_unmapped} unmapped)' if n_unmapped else '')
            + f'; {len(contacts)} protein contacts, '
            f'{int(contacts.bioplex_293T.sum())} seen in 293T, '
            f'{int(contacts.bioplex_HCT116.sum())} in HCT116')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='bioplexpy-structure',
        description='Compare your own structure files (experimental or predicted) '
                    'with BioPlex AP-MS interactions in 293T and HCT116 cells.',
        epilog='Put the structure paths first: options that take a list '
               '(--uniprots, --chain-map) would otherwise read them as list items. '
               'Example: bioplexpy-structure models/ --uniprots P61158 P61160 '
               '--min-plddt 70')
    parser.add_argument('structures', nargs='+',
                        help='.pdb/.ent/.cif/.mmcif files, or directories of them '
                             '(not searched recursively)')

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

    analysis = parser.add_argument_group('analysis')
    analysis.add_argument('--min-plddt', type=float, default=None,
                          help='predicted structures only: ignore atoms below this '
                               'pLDDT (0-100) when finding contacts, e.g. 70')
    analysis.add_argument('--distance', type=float, default=6,
                          help='direct-contact distance cutoff in Angstroms (default 6)')
    analysis.add_argument('--bioplex-293t-version', default='3.0')
    analysis.add_argument('--bioplex-hct116-version', default='1.0')

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
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    chain_map = _read_chain_map(args.chain_map, args.chain_map_file)
    uniprots = _read_uniprots(args.uniprots, args.uniprots_file)
    if chain_map is None and uniprots is None:
        parser.error('give a chain mapping (--chain-map/--chain-map-file) or the '
                     "complex's UniProt IDs (--uniprots/--uniprots-file)")

    structure_files = _collect_structure_files(args.structures)
    if not structure_files:
        parser.error('no structure files found')

    import matplotlib
    matplotlib.use('Agg')

    from bioplexpy.data_import_funcs import getBioPlex

    os.makedirs(args.out_dir, exist_ok=True)
    print('Loading BioPlex data...', file=sys.stderr)
    bp_293t_df = getBioPlex('293T', args.bioplex_293t_version)
    bp_hct116_df = getBioPlex('HCT116', args.bioplex_hct116_version)

    failures = 0
    used_names = set()
    for structure_file in structure_files:
        name = _unique_name(structure_file, used_names)
        try:
            summary = process_structure(structure_file, name, args, chain_map,
                                        uniprots, bp_293t_df, bp_hct116_df)
            print(f'{structure_file}: {summary}')
        except Exception as e:
            failures += 1
            print(f'{structure_file}: FAILED -- {e}', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(f'Wrote results for {len(structure_files) - failures}/{len(structure_files)} '
          f'structure(s) to {args.out_dir}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
