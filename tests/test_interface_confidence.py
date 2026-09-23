'''
Checks for the interface-confidence readers against real predictor output
(AlphaFold3 server, Boltz, ColabFold) kept outside the repo in
../fold_outputs/. Tests whose data isn't there are skipped.

Run with pytest, or directly: python tests/test_interface_confidence.py
'''

import glob
import json
import os
import tempfile

from bioplexpy.analysis_funcs import (find_structure_files,
                                      interface_confidence_by_uniprot,
                                      read_interface_confidence)
from bioplexpy.visualization_funcs import get_edge_confidence_scores

FOLDS = os.environ.get('BIOPLEXPY_FOLD_OUTPUTS', os.path.join(
    os.path.dirname(__file__), '..', '..', 'fold_outputs'))
DECOY_ZIP = os.path.join(FOLDS, 'arp23_decoy_HSD17B14', 'af3', 'folds_2026_09_23_18_59.zip')
ARP23 = os.path.join(FOLDS, 'arp23_6YW7')
TFIIH_AF3 = os.path.join(FOLDS, 'tfiih_core_6NMI', 'af3')


class Skip(Exception):
    pass


def _need(path):
    if not os.path.exists(path):
        try:
            import pytest
            pytest.skip(f'missing {path}')
        except ImportError:
            raise Skip(path)


def test_af3_decoy_zip_separates_decoy():
    '''
    HSD17B14 (chain H) is a known non-binder folded with Arp2/3: every
    pair with it must come out ~0.1 and every Arp2/3 pair well above, in
    all 5 models -- a scrambled chain order would break this.
    '''
    _need(DECOY_ZIP)
    with tempfile.TemporaryDirectory() as tmp:
        files, _, _ = find_structure_files(DECOY_ZIP, extract_dir=tmp)
        assert len(files) == 5
        # only the small summary files come along, never full_data
        extracted = [f for _, _, fs in os.walk(tmp) for f in fs]
        assert not any('full_data' in f for f in extracted)
        for f in files:
            confidence = read_interface_confidence(f)
            assert confidence['tool'] == 'af3'
            scores = confidence['scores']['pair_iptm']
            with open(confidence['source']) as fh:
                raw = json.load(fh)['chain_pair_iptm']
            assert scores[('D', 'H')] == raw[3][7]
            assert max(v for k, v in scores.items() if 'H' in k) <= 0.15
            assert min(v for k, v in scores.items() if 'H' not in k) >= 0.3


def test_af3_chain_order_matches_model_files():
    for folder in (os.path.join(ARP23, 'af3'), TFIIH_AF3):
        _need(folder)
        files, _, _ = find_structure_files(folder)
        assert len(files) == 5
        for f in files:
            # raises if summary chain_ids disagree with the CIF chain order
            assert read_interface_confidence(f)['tool'] == 'af3'


def test_boltz_reads_asymmetric_pair_iptm():
    folder = os.path.join(ARP23, 'boltz')
    _need(folder)
    files, _, _ = find_structure_files(folder)
    confidence = read_interface_confidence(files[0])
    assert confidence['tool'] == 'boltz'
    scores = confidence['scores']['pair_iptm']
    with open(confidence['source']) as fh:
        raw = json.load(fh)['pair_chains_iptm']
    assert scores[('A', 'E')] == raw['0']['4']
    assert scores[('E', 'A')] == raw['4']['0']

    chain_map = {c: [c] for c in 'ABCDEFG'}
    for reduce, expected in (('mean', (raw['0']['4'] + raw['4']['0']) / 2),
                             ('min', min(raw['0']['4'], raw['4']['0'])),
                             ('max', max(raw['0']['4'], raw['4']['0']))):
        edge_scores, label = get_edge_confidence_scores(confidence, chain_map,
                                                        reduce=reduce)
        assert abs(edge_scores[frozenset('AE')] - expected) < 1e-9
        assert f'{reduce} of both directions' in label


def test_colabfold_precomputed_scores():
    folder = os.path.join(ARP23, 'af2_multimer')
    _need(folder)
    files, _, _ = find_structure_files(folder)
    confidence = read_interface_confidence(files[0])
    assert confidence['tool'] == 'colabfold'
    assert set(confidence['scores']) == {'ipsae', 'pdockq', 'pdockq2'}
    # pdockq is stored once per pair; both orders are filled in
    assert confidence['scores']['pdockq'][('A', 'B')] == confidence['scores']['pdockq'][('B', 'A')]
    # no chain-pair ipTM in ColabFold output: nothing to draw by default
    assert get_edge_confidence_scores(confidence, {c: [c] for c in 'ABCDEFG'}) == (None, None)


def test_homo_oligomer_keeps_highest():
    confidence = dict(tool='af3', source='x', scores={'pair_iptm': {
        ('A', 'C'): 0.2, ('C', 'A'): 0.2, ('B', 'C'): 0.8, ('C', 'B'): 0.8,
        ('A', 'B'): 0.9, ('B', 'A'): 0.9}})
    by_uniprot = interface_confidence_by_uniprot(
        confidence, {'A': ['P1'], 'B': ['P1'], 'C': ['P2']})['pair_iptm']
    assert by_uniprot == {('P1', 'P2'): 0.8, ('P2', 'P1'): 0.8}


def test_experimental_structure_has_none():
    pdb = os.path.join(os.path.dirname(__file__), '..', 'TESTING', 'pdb6nmi.ent')
    _need(pdb)
    assert read_interface_confidence(pdb) is None


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'PASS {name}')
            except Skip as e:
                print(f'SKIP {name} ({e})')
