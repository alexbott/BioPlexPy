'''
Checks for the opt-in confidence filter (filter_contacts_by_score(),
--min-score) and the BioPlex3D contact definition (contact_definition=
'bioplex3d'), on real predictor output kept outside the repo in
../fold_outputs/. Tests whose data isn't there are skipped.

The cutoff used here, 0.05, is only an illustration: it sits inside the gap
seen so far between decoy contacts (ipsae_calc 0) and the lowest true
contact (0.11). It is not a recommended default.

Run with pytest, or directly: python tests/test_filtering.py
'''

import glob
import os
import tempfile
import warnings

import pandas as pd

from bioplexpy.analysis_funcs import (PDB_to_interacting_chains_uniprot_maps,
                                      _bioplex3d_interface, _load_pdb_model,
                                      _read_interface_confidence_or_warn,
                                      compare_structure_contacts_to_BioPlex,
                                      filter_contacts_by_score, find_structure_files,
                                      interface_score_by_pair, read_pae)
from bioplexpy.cli import _resolve_filter_args, build_parser, summarize_contacts

HERE = os.path.dirname(__file__)
FOLDS = os.environ.get('BIOPLEXPY_FOLD_OUTPUTS', os.path.join(HERE, '..', '..', 'fold_outputs'))
ARP23 = os.path.join(FOLDS, 'arp23_6YW7')
DECOY_ZIP = os.path.join(FOLDS, 'arp23_decoy_HSD17B14', 'af3', 'folds_2026_09_23_18_59.zip')
DECOY_BOLTZ = os.path.join(FOLDS, 'arp23_decoy_HSD17B14', 'boltz')
TFIIH = os.path.join(FOLDS, 'tfiih_core_6NMI', 'af3')
REFERENCE = os.path.join(HERE, 'data', 'bioplex3d_reference_interfaces.tsv')
CUTOFF = 0.05

# chain -> UniProt for the Arp2/3 + HSD17B14 decoy models (chain H is the decoy)
DECOY = 'Q9BPX1'
DECOY_CHAINS = dict(zip('ABCDEFGH', ['P61158', 'P61160', 'Q92747', 'O15144', 'O15145',
                                     'P59998', 'O15511', DECOY]))
TFIIH_CHAINS = dict(zip('ABCDEFGH', ['P19447', 'P18074', 'P32780', 'Q92759', 'Q13888',
                                     'Q13889', 'Q6ZYL4', 'P51948']))


class Skip(Exception):
    pass


def _need(path):
    if not os.path.exists(path):
        try:
            import pytest
            pytest.skip(f'missing {path}')
        except ImportError:
            raise Skip(path)


def _filtered(structure_file, chain_map, min_score=CUTOFF, **kwargs):
    maps = PDB_to_interacting_chains_uniprot_maps(structure_file, None, 6,
                                                  chain_to_uniprot=chain_map)
    confidence = _read_interface_confidence_or_warn(structure_file, chain_ids=list(maps[2]),
                                                    compute_scores=True)
    kept, status = filter_contacts_by_score(maps[1], maps[0], confidence,
                                            min_score=min_score, **kwargs)
    return maps, confidence, kept, status


def test_pass_fail_unscored_and_reduce():
    '''Hand-built scores: no files needed.'''
    chain_map = {'A': ['P1'], 'B': ['P2'], 'C': ['P3'], 'R': ['RNA:R']}
    contacts = [['P1', 'P2'], ['P1', 'P3'], ['P2', 'P3'], ['P1', 'RNA:R']]
    confidence = dict(tool='boltz', source='x', scores={'ipsae_calc': {
        ('A', 'B'): 0.6, ('B', 'A'): 0.02, ('A', 'C'): 0.0, ('C', 'A'): 0.0,
        ('B', 'C'): float('nan'), ('C', 'B'): float('nan')}})

    # off by default: nothing changes, nothing is reported
    assert filter_contacts_by_score(contacts, chain_map, confidence) == (contacts, {})

    kept, status = filter_contacts_by_score(contacts, chain_map, confidence, min_score=0.05)
    assert kept == [['P1', 'P2'], ['P2', 'P3'], ['P1', 'RNA:R']]
    assert status == {frozenset(('P1', 'P2')): 'pass', frozenset(('P1', 'P3')): 'fail',
                      frozenset(('P2', 'P3')): 'unscored',
                      frozenset(('P1', 'RNA:R')): 'unscored'}
    # the two directions of P1-P2 are 0.6 and 0.02
    for reduce, expected in (('max', 'pass'), ('mean', 'pass'), ('min', 'fail')):
        status = filter_contacts_by_score(contacts, chain_map, confidence, min_score=0.05,
                                          reduce=reduce)[1]
        assert status[frozenset(('P1', 'P2'))] == expected, reduce
    assert interface_score_by_pair(confidence, chain_map, 'ipsae_calc') == {
        frozenset(('P1', 'P2')): 0.6, frozenset(('P1', 'P3')): 0.0}

    # a score the model doesn't have: every contact kept, with a warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for scores in (confidence, None):
            kept, status = filter_contacts_by_score(contacts, chain_map, scores,
                                                    score='pair_iptm', min_score=0.5)
            assert kept == contacts and set(status.values()) == {'unscored'}
    assert len(caught) == 2 and "No 'pair_iptm' score" in str(caught[0].message)


def test_homo_oligomer_uses_best_chain_pair():
    chain_map = {'A': ['P1'], 'B': ['P1'], 'C': ['P2']}
    confidence = dict(tool='af3', source='x', scores={'ipsae_calc': {
        ('A', 'C'): 0.0, ('C', 'A'): 0.0, ('B', 'C'): 0.4, ('C', 'B'): 0.3}})
    kept, status = filter_contacts_by_score([['P1', 'P2']], chain_map, confidence,
                                            min_score=0.35)
    assert kept == [['P1', 'P2']] and list(status.values()) == ['pass']


def _check_decoy(files):
    assert len(files) == 5
    for structure_file in files:
        maps, _, kept, status = _filtered(structure_file, DECOY_CHAINS)
        decoy = [pair for pair in maps[1] if DECOY in pair]
        for pair in maps[1]:
            expected = 'fail' if DECOY in pair else 'pass'
            assert status[frozenset(pair)] == expected, (structure_file, pair)
        assert kept == [pair for pair in maps[1] if pair not in decoy]
        assert len(kept) >= 12


def test_af3_decoy_contacts_are_removed_and_real_ones_kept():
    _need(DECOY_ZIP)
    with tempfile.TemporaryDirectory() as tmp:
        files = find_structure_files(DECOY_ZIP, extract_dir=tmp, include_pae=True)[0]
        _check_decoy(files)


def test_boltz_decoy_contacts_are_removed_and_real_ones_kept():
    _need(DECOY_BOLTZ)
    _check_decoy(sorted(glob.glob(os.path.join(DECOY_BOLTZ, '*', 'predictions', '*',
                                               '*_model_?.cif'))))


def test_tfiih_contacts_all_survive():
    '''Includes the lowest-scoring true contacts seen so far (0.11-0.16).'''
    _need(TFIIH)
    files = sorted(glob.glob(os.path.join(TFIIH, '*_model_?.cif')))
    assert len(files) == 5
    lowest = 1.0
    for structure_file in files:
        maps, confidence, kept, status = _filtered(structure_file, TFIIH_CHAINS)
        assert kept == maps[1] and set(status.values()) == {'pass'}, structure_file
        values = interface_score_by_pair(confidence, maps[0], 'ipsae_calc')
        lowest = min([lowest] + [values[frozenset(pair)] for pair in maps[1]])
    assert CUTOFF < lowest < 0.2, lowest


def test_table_column_and_no_change_without_filter():
    _need(DECOY_BOLTZ)
    structure_file = sorted(glob.glob(os.path.join(DECOY_BOLTZ, '*', 'predictions', '*',
                                                   '*_model_0.cif')))[0]
    maps, confidence, _, status = _filtered(structure_file, DECOY_CHAINS)
    empty = pd.DataFrame({'UniprotA': [], 'UniprotB': [], 'SymbolA': [], 'SymbolB': []})
    plain = compare_structure_contacts_to_BioPlex(*maps, empty, empty,
                                                  interface_confidence=confidence)
    table = compare_structure_contacts_to_BioPlex(*maps, empty, empty,
                                                  interface_confidence=confidence,
                                                  filter_status=status)
    assert 'passes_filter' not in plain
    assert list(table.columns) == (list(plain.columns[:7]) + ['passes_filter']
                                   + list(plain.columns[7:]))
    pd.testing.assert_frame_equal(table.drop(columns='passes_filter'), plain)
    # the raw contact call is untouched; only the decoy rows are rejected
    rejected = table[table.passes_filter == False]  # noqa: E712
    assert len(rejected) == 2 and rejected.structure_contact.all()
    assert ((rejected.UniprotA == DECOY) | (rejected.UniprotB == DECOY)).all()

    # no PAE file -> no ipsae_calc: contacts kept, column empty
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        kept, status = filter_contacts_by_score(maps[1], maps[0], None, min_score=CUTOFF)
    table = compare_structure_contacts_to_BioPlex(*maps, empty, empty, filter_status=status)
    assert kept == maps[1] and table.passes_filter.isna().all()

    # summary: raw counts stay, pass counts exclude only rejected contacts
    summary = summarize_contacts(
        {'m0': compare_structure_contacts_to_BioPlex(
            *maps, empty, empty, filter_status=_filtered(structure_file, DECOY_CHAINS)[3]),
         'm1': table}, empty, empty)
    decoy_rows = summary[(summary.UniprotA == DECOY) | (summary.UniprotB == DECOY)]
    assert (decoy_rows.n_structures_contact == 2).all()
    assert (decoy_rows.n_structures_pass == 1).all()      # m1 is unscored, so kept
    others = summary.drop(decoy_rows.index)
    assert (others.n_structures_pass == others.n_structures_contact).all()


def test_cli_options():
    parser = build_parser()

    def parse(*argv):
        args = parser.parse_args(['x.cif', *argv])
        _resolve_filter_args(parser, args)
        return args

    args = parse()
    assert args.min_score is None and not args.compute_scores
    assert args.contact_definition == 'any_atom'
    args = parse('--min-score', '0.1')
    assert (args.filter_score, args.filter_reduce, args.compute_scores) == (
        'ipsae_calc', 'max', True)
    args = parse('--min-score', '0.3', '--filter-score', 'pair_iptm')
    assert not args.compute_scores
    assert parse('--contact-definition', 'bioplex3d').contact_definition == 'bioplex3d'
    try:
        parse('--filter-score', 'pair_iptm')
    except SystemExit:
        pass
    else:
        raise AssertionError('--filter-score without --min-score should be an error')


def test_bioplex3d_interface_matches_bioplex3d():
    '''Interface sizes for every chain pair, against BioPlex3D's own function.'''
    _need(ARP23)
    reference = pd.read_csv(REFERENCE, sep='\t', comment='#')
    files = {'colabfold': glob.glob(os.path.join(ARP23, 'af2_multimer',
                                                 '*_unrelaxed_rank_001_*.pdb')),
             'boltz': glob.glob(os.path.join(ARP23, 'boltz', '*', 'predictions', '*',
                                             '*_model_0.cif'))}
    for tool, found in files.items():
        model = _load_pdb_model(found[0], None)
        pae_data = read_pae(found[0], model=model)
        rows = reference[reference.tool == tool]
        assert len(rows) == 21
        in_contact = set()
        for _, row in rows.iterrows():
            contact = _bioplex3d_interface(model, row.chain1, row.chain2, pae_data)
            assert (contact.any(axis=1).sum(), contact.any(axis=0).sum(),
                    contact.sum()) == (row.residues1, row.residues2, row.pairs), (tool, row)
            if row.pairs:
                in_contact.add(frozenset((row.chain1, row.chain2)))
        # and the chain pairs the contact search reports are the non-empty ones
        chain_map = {chain.get_id(): chain.get_id() for chain in model}
        pairs = PDB_to_interacting_chains_uniprot_maps(
            found[0], None, 6, chain_to_uniprot=chain_map,
            contact_definition='bioplex3d')[1]
        assert {frozenset(pair) for pair in pairs} == in_contact


def test_bioplex3d_definition_without_pae_uses_distance_only():
    pdb = os.path.join(HERE, '..', 'TESTING', 'pdb6nmi.ent')
    _need(pdb)
    chain_map = {chain.get_id(): chain.get_id() for chain in _load_pdb_model(pdb, None)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pairs = PDB_to_interacting_chains_uniprot_maps(
            pdb, None, 6, chain_to_uniprot=chain_map, contact_definition='bioplex3d')[1]
    assert any('CA distance only' in str(w.message) for w in caught)
    any_atom = PDB_to_interacting_chains_uniprot_maps(pdb, None, 6,
                                                      chain_to_uniprot=chain_map)[1]
    # CA-CA < 8 A is the stricter rule
    assert pairs and {frozenset(p) for p in pairs} <= {frozenset(p) for p in any_atom}


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'PASS {name}')
            except Skip as e:
                print(f'SKIP {name} ({e})')
