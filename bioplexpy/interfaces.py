'''
What a contact between two chains is made of: the residue pairs that are
close, which of them pass the contact rule and which condition the others
fail, the PAE and pLDDT at the interface, and each residue's partners.

`bioplexpy-structure` writes this per structure as <name>_interfaces.tsv
(one row per chain pair) and <name>_interface_residues.tsv (one row per
residue), and the --viewer page shows it in the pairs table.

A pair of residues is *close* when it meets the rule's distance, and *passes*
when it meets the whole rule. For two protein chains under a CA rule
(resolve_contact_settings(); the default) the rule also asks for a pLDDT and
a PAE, so close pairs can fail; under the any-atom rule, and for a pair with
a nucleic acid chain, close and passing are the same thing.
'''

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def _residue_distances(model, chain_i, chain_j, by_ca, plddt_scale, min_plddt):
    '''
    Distance between every residue of chain i and every residue of chain j
    (all residues, in file order): between CA atoms, or the smallest
    between any two polymer atoms (atoms under min_plddt left out, as the
    any-atom rule does). Infinite where a residue has no such atom.
    '''
    if by_ca:
        def ca(chain):
            return np.array([residue['CA'].coord if residue.id[0] == ' ' and 'CA' in residue
                             else [np.nan] * 3 for residue in chain], dtype=float)
        distances = cdist(ca(model[chain_i]), ca(model[chain_j]))
        distances[np.isnan(distances)] = np.inf
        return distances

    def atoms(chain):
        coords, index = [], []
        for k, residue in enumerate(chain):
            if residue.id[0] != ' ':
                continue
            for atom in residue:
                if plddt_scale is None or atom.get_bfactor() * plddt_scale >= min_plddt:
                    coords.append(atom.coord)
                    index.append(k)
        return np.array(coords, dtype=float).reshape(-1, 3), np.array(index, dtype=int)

    coords_i, index_i = atoms(model[chain_i])
    coords_j, index_j = atoms(model[chain_j])
    distances = np.full((len(model[chain_i]), len(model[chain_j])), np.inf)
    if len(coords_i) and len(coords_j):
        between = cdist(coords_i, coords_j)
        # smallest over the atoms of each residue: the atoms are in residue order
        present_i, start_i = np.unique(index_i, return_index=True)
        present_j, start_j = np.unique(index_j, return_index=True)
        smallest = np.minimum.reduceat(np.minimum.reduceat(between, start_i, axis=0),
                                       start_j, axis=1)
        distances[np.ix_(present_i, present_j)] = smallest
    return distances


def interface_details(model, contact, pae_data=None, chain_pairs=None):
    '''
    The interface of every pair of chains that has close residues.

    Parameters
    ----------
    model: Bio.PDB model
    contact: dict from resolve_contact_settings() (contact_atoms, distance,
        min_plddt, max_pae)
    pae_data: dict from read_pae() (optional). Without it the pLDDT and PAE
        conditions are not applied, as in the contact call itself, and no
        PAE or pLDDT is reported.
    chain_pairs: list of [chain_i, chain_j] (optional), the pairs in contact
        from _direct_interaction_chain_pairs(). Under a CA rule every pair of
        protein chains is looked at as well, so that a pair with close
        residues that all fail the rule is reported too.

    Returns
    -------
    list of dict, one per chain pair, with `chains` [i, j], `numbers` and
    `names` (residue numbers and names of either chain, file order) and
    arrays [residue of i][residue of j]: `distance`, `close`, `passes`,
    and, where the rule has those conditions, `low_plddt` and `high_pae`
    (close pairs failing each; a pair can fail both). `pae` is the smaller
    of the two directions (the one the rule tests), `plddt` the per-residue
    values of either chain; both None without pae_data.
    '''
    from bioplexpy.analysis_funcs import (NUCLEIC_ACID_CONTACT_DISTANCE, _ca_interface,
                                          _plddt_scale_factor, classify_chain)

    ca_rule = contact['contact_atoms'] == 'ca'
    min_plddt, max_pae = contact['min_plddt'], contact['max_pae']
    order = [chain.get_id() for chain in model]
    kinds = {chain.get_id(): classify_chain(chain) for chain in model}
    proteins = [c for c in order if kinds[c] == 'protein']
    pairs = {tuple(sorted(pair, key=order.index)) for pair in (chain_pairs or [])}
    if ca_rule:
        pairs |= {(a, b) for k, a in enumerate(proteins) for b in proteins[k + 1:]}
    plddt_scale = (_plddt_scale_factor(model)
                   if min_plddt is not None and not ca_rule else None)

    details = []
    for chain_i, chain_j in sorted(pairs, key=lambda pair: [order.index(c) for c in pair]):
        by_ca = ca_rule and kinds[chain_i] == 'protein' and kinds[chain_j] == 'protein'
        limit = (contact['distance'] if by_ca or not ca_rule
                 else NUCLEIC_ACID_CONTACT_DISTANCE)
        distance = _residue_distances(model, chain_i, chain_j, by_ca, plddt_scale, min_plddt)
        close = distance < limit
        if not close.any():
            continue
        scored = (pae_data is not None and chain_i in pae_data['rows']
                  and chain_j in pae_data['rows'])
        detail = {'chains': [chain_i, chain_j], 'distance': distance, 'close': close,
                  'passes': close, 'low_plddt': None, 'high_pae': None, 'pae': None,
                  'plddt': None,
                  'numbers': [[residue.id[1] for residue in model[c]] for c in (chain_i, chain_j)],
                  'names': [[residue.get_resname() for residue in model[c]]
                            for c in (chain_i, chain_j)]}
        if scored:
            rows_i, rows_j = pae_data['rows'][chain_i], pae_data['rows'][chain_j]
            pae = pae_data['pae']
            detail['pae'] = np.minimum(pae[np.ix_(rows_i, rows_j)], pae[np.ix_(rows_j, rows_i)].T)
            detail['plddt'] = [pae_data['plddt'][rows_i], pae_data['plddt'][rows_j]]
        if by_ca:
            detail['passes'] = _ca_interface(model, chain_i, chain_j,
                                             pae_data if scored else None,
                                             contact['distance'], min_plddt, max_pae)
            if scored and min_plddt is not None:
                detail['low_plddt'] = close & ~np.outer(detail['plddt'][0] >= min_plddt,
                                                        detail['plddt'][1] >= min_plddt)
            if scored and max_pae is not None:
                detail['high_pae'] = close & (detail['pae'] > max_pae)
        details.append(detail)
    return details


def summarize_interfaces(details, flip=None):
    '''
    The numbers of one interface, over one or several chain pairs (several
    where a protein is present in more than one copy).

    Parameters
    ----------
    details: list of dict from interface_details()
    flip: list of bool (optional), per chain pair: True to count its second
        chain as side A (so that side A is one protein throughout)

    Returns
    -------
    dict: n_close, n_pass (residue pairs); n_low_plddt, n_high_pae (close
    pairs failing each condition, None where the rule has none);
    residues [side A, side B] (residues with a passing partner);
    pae_pass and pae_close ([median, lowest] over the passing and over the
    close pairs, None without a PAE); plddt [side A, side B] (mean over the
    residues with a passing partner, or with a close one where none passes)
    '''
    flip = flip or [False] * len(details)

    def total(key):
        values = [d[key] for d in details]
        return None if any(v is None for v in values) else int(sum(v.sum() for v in values))

    def pae_over(key):
        values = [d['pae'][d[key]] for d in details if d['pae'] is not None]
        values = np.concatenate(values) if values else np.array([])
        return [round(float(np.median(values)), 1), round(float(values.min()), 1)] \
            if len(values) else None

    n_pass = total('passes')
    of = 'passes' if n_pass else 'close'
    residues, plddt = [0, 0], [[], []]
    for detail, turned in zip(details, flip):
        for side in (0, 1):
            to = 1 - side if turned else side
            residues[to] += int(detail['passes'].any(axis=1 - side).sum())
            if detail['plddt'] is not None:
                plddt[to].append(detail['plddt'][side][detail[of].any(axis=1 - side)])
    means = [np.concatenate(side) if side else np.array([]) for side in plddt]
    return {'n_close': total('close'), 'n_pass': n_pass,
            'n_low_plddt': total('low_plddt'), 'n_high_pae': total('high_pae'),
            'residues': residues,
            'pae_pass': pae_over('passes'), 'pae_close': pae_over('close'),
            'plddt': ([round(float(side.mean()), 1) if len(side) else None for side in means]
                      if any(len(side) for side in means) else None)}


def interface_residue_rows(detail):
    '''
    The residues of one chain pair that have a close partner, either side.

    Returns
    -------
    [rows of chain i, rows of chain j]; a row is a dict: number, name,
    plddt (None without a PAE file), n_pass and n_close (partner residues
    that pass the rule / are close), and the nearest partner (a passing one
    if there is any): partner (its number), distance and pae to it.
    '''
    sides = []
    for side in (0, 1):
        def per_residue(array):
            return array if side == 0 else array.T
        close, passes = per_residue(detail['close']), per_residue(detail['passes'])
        distance = per_residue(detail['distance'])
        pae = per_residue(detail['pae']) if detail['pae'] is not None else None
        rows = []
        for k in np.flatnonzero(close.any(axis=1)):
            among = passes[k] if passes[k].any() else close[k]
            partner = int(np.flatnonzero(among)[np.argmin(distance[k][among])])
            rows.append({
                'number': int(detail['numbers'][side][k]), 'name': detail['names'][side][k],
                'plddt': (round(float(detail['plddt'][side][k]), 1)
                          if detail['plddt'] is not None else None),
                'n_pass': int(passes[k].sum()), 'n_close': int(close[k].sum()),
                'partner': int(detail['numbers'][1 - side][partner]),
                'distance': round(float(distance[k][partner]), 1),
                'pae': round(float(pae[k][partner]), 1) if pae is not None else None})
        sides.append(rows)
    return sides


def interface_tables(details, chain_ids_map, symbols=None):
    '''
    The interfaces of one structure as two tables.

    Parameters
    ----------
    details: list of dict from interface_details()
    chain_ids_map: dict, chain ID -> list of protein IDs (UniProt)
    symbols: dict (optional), protein ID -> gene symbol

    Returns
    -------
    (interfaces, residues): pandas DataFrames, one row per chain pair and
    one row per residue with a close partner
    '''
    symbols = symbols or {}

    def protein(chain):
        ids = chain_ids_map.get(chain, [])
        return ';'.join(ids), ';'.join(symbols.get(i, '') for i in ids)

    pair_rows, residue_rows = [], []
    for detail in details:
        chain_a, chain_b = detail['chains']
        s = summarize_interfaces([detail])
        (uniprot_a, symbol_a), (uniprot_b, symbol_b) = protein(chain_a), protein(chain_b)
        pair_rows.append({
            'chain_A': chain_a, 'chain_B': chain_b, 'UniprotA': uniprot_a, 'UniprotB': uniprot_b,
            'SymbolA': symbol_a, 'SymbolB': symbol_b, 'contact': s['n_pass'] > 0,
            'residue_pairs_close': s['n_close'], 'residue_pairs_pass': s['n_pass'],
            'close_pairs_low_plddt': s['n_low_plddt'], 'close_pairs_high_pae': s['n_high_pae'],
            'residues_A': s['residues'][0], 'residues_B': s['residues'][1],
            'pae_pass_median': s['pae_pass'][0] if s['pae_pass'] else None,
            'pae_pass_lowest': s['pae_pass'][1] if s['pae_pass'] else None,
            'pae_close_median': s['pae_close'][0] if s['pae_close'] else None,
            'pae_close_lowest': s['pae_close'][1] if s['pae_close'] else None,
            'plddt_A': s['plddt'][0] if s['plddt'] else None,
            'plddt_B': s['plddt'][1] if s['plddt'] else None})
        for side, rows in enumerate(interface_residue_rows(detail)):
            chain, other = detail['chains'][side], detail['chains'][1 - side]
            for row in rows:
                residue_rows.append({
                    'chain': chain, 'Uniprot': protein(chain)[0], 'Symbol': protein(chain)[1],
                    'residue': row['number'], 'residue_name': row['name'], 'plddt': row['plddt'],
                    'partner_chain': other, 'partner_Uniprot': protein(other)[0],
                    'partner_Symbol': protein(other)[1],
                    'passes': row['n_pass'] > 0, 'partners_pass': row['n_pass'],
                    'partners_close': row['n_close'], 'nearest_partner_residue': row['partner'],
                    'distance': row['distance'], 'pae': row['pae']})
    pair_columns = ['chain_A', 'chain_B', 'UniprotA', 'UniprotB', 'SymbolA', 'SymbolB', 'contact',
                    'residue_pairs_close', 'residue_pairs_pass', 'close_pairs_low_plddt',
                    'close_pairs_high_pae', 'residues_A', 'residues_B', 'pae_pass_median',
                    'pae_pass_lowest', 'pae_close_median', 'pae_close_lowest', 'plddt_A',
                    'plddt_B']
    residue_columns = ['chain', 'Uniprot', 'Symbol', 'residue', 'residue_name', 'plddt',
                       'partner_chain', 'partner_Uniprot', 'partner_Symbol', 'passes',
                       'partners_pass', 'partners_close', 'nearest_partner_residue', 'distance',
                       'pae']
    return (pd.DataFrame(pair_rows, columns=pair_columns),
            pd.DataFrame(residue_rows, columns=residue_columns))


def structure_interfaces(structure_file, contact, chain_ids_map, symbols=None):
    '''
    interface_tables() for a structure file: its model and PAE are read,
    and the pairs in contact found, as the contact table does.

    Returns
    -------
    (interfaces, residues): pandas DataFrames (see interface_tables())
    '''
    from bioplexpy.analysis_funcs import (_direct_interaction_chain_pairs, _load_pdb_model,
                                          read_pae)

    model = _load_pdb_model(structure_file, None)
    try:
        pae_data = read_pae(structure_file, model=model)
    except (ValueError, KeyError):
        pae_data = None
    # the rule takes the PAE file only where it has a pLDDT or PAE condition
    rule_pae = pae_data if (contact['contact_atoms'] == 'ca'
                            and (contact['min_plddt'] is not None
                                 or contact['max_pae'] is not None)) else None
    chain_pairs = _direct_interaction_chain_pairs(
        model, contact['distance'], min_plddt=contact['min_plddt'],
        contact_atoms=contact['contact_atoms'], max_pae=contact['max_pae'], pae_data=rule_pae)
    return interface_tables(interface_details(model, contact, pae_data, chain_pairs),
                            chain_ids_map, symbols)
