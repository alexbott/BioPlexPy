'''
Checks for compute_interface_scores() against the BioPlex3D pipeline's own
values (tests/data/bioplex3d_reference_scores.tsv) on real predictor
output (AlphaFold3 server, Boltz, ColabFold) kept outside the repo in
../fold_outputs/. Tests whose data isn't there are skipped.

Run with pytest, or directly: python tests/test_interface_scores.py
'''

import glob
import json
import os
import shutil
import tempfile
import warnings

import pandas as pd

from bioplexpy.analysis_funcs import (COMPUTED_SCORES,
                                      _read_interface_confidence_or_warn,
                                      compute_interface_scores, find_pae_file,
                                      find_structure_files,
                                      read_interface_confidence)

HERE = os.path.dirname(__file__)
FOLDS = os.environ.get('BIOPLEXPY_FOLD_OUTPUTS', os.path.join(HERE, '..', '..', 'fold_outputs'))
ARP23 = os.path.join(FOLDS, 'arp23_6YW7')
DECOY_ZIP = os.path.join(FOLDS, 'arp23_decoy_HSD17B14', 'af3', 'folds_2026_09_23_18_59.zip')
RNASEP_ZIP = os.path.join(FOLDS, 'rnasep_6AHR', 'af3', 'fold_rnasep_6ahr.zip')
REFERENCE = os.path.join(HERE, 'data', 'bioplex3d_reference_scores.tsv')


class Skip(Exception):
    pass


def _need(path):
    if not os.path.exists(path):
        try:
            import pytest
            pytest.skip(f'missing {path}')
        except ImportError:
            raise Skip(path)


def _arp23_models():
    return {'af3': glob.glob(os.path.join(ARP23, 'af3', '*_model_0.cif')),
            'boltz': glob.glob(os.path.join(ARP23, 'boltz', '*', 'predictions', '*',
                                            '*_model_0.cif')),
            'colabfold': glob.glob(os.path.join(ARP23, 'af2_multimer',
                                                '*_unrelaxed_rank_001_*.pdb'))}


def test_scores_match_bioplex3d():
    '''Every score, every chain pair, for one model of each tool.'''
    _need(ARP23)
    reference = pd.read_csv(REFERENCE, sep='\t', comment='#')
    columns = {'pDockQ': ('pdockq_calc', False), 'pDockQ2': ('pdockq2_calc', False),
               'pDockQ2rev': ('pdockq2_calc', True), 'LIAfwd': ('lia', False),
               'LISfwd': ('lis', False), 'LIArev': ('lia', True), 'LISrev': ('lis', True),
               'cLIAfwd': ('clia', False), 'cLISfwd': ('clis', False),
               'cLIArev': ('clia', True), 'cLISrev': ('clis', True),
               'iLIS': ('ilis', False), 'totalClashes': ('clashes', False)}
    for tool, files in _arp23_models().items():
        scores = compute_interface_scores(files[0])['scores']
        rows = reference[reference.tool == tool]
        assert len(rows) == 21 and set(scores) == set(COMPUTED_SCORES)
        for _, row in rows.iterrows():
            for column, (name, reverse) in columns.items():
                pair = (row.chain2, row.chain1) if reverse else (row.chain1, row.chain2)
                assert abs(scores[name][pair] - row[column]) < 1e-6, (tool, column, pair)


def test_ipsae_matches_colabfold_and_keeps_predictor_scores():
    '''
    ipsae_calc reproduces the ipSAE ColabFold 1.6 wrote, and computed
    scores sit next to the predictor's own instead of replacing them.
    '''
    _need(ARP23)
    model = _arp23_models()['colabfold'][0]
    predictor = read_interface_confidence(model)['scores']
    merged = _read_interface_confidence_or_warn(model, compute_scores=True)['scores']
    for name in ('ipsae', 'pdockq', 'pdockq2'):
        assert merged[name] == predictor[name]
    assert set(COMPUTED_SCORES) <= set(merged)
    assert len(predictor['ipsae']) == 42
    for pair, value in predictor['ipsae'].items():
        assert abs(merged['ipsae_calc'][pair] - value) < 1e-3, pair


def test_stock_colabfold_still_gets_computed_scores():
    '''A scores file without the precomputed chain-pair scores has no predictor scores.'''
    _need(ARP23)
    model = _arp23_models()['colabfold'][0]
    scores_file = find_pae_file(model)[0]
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(model, tmp)
        with open(scores_file) as fh:
            data = json.load(fh)
        for name in ('ipsae', 'pdockq', 'pdockq2'):
            del data[name]
        with open(os.path.join(tmp, os.path.basename(scores_file)), 'w') as fh:
            json.dump(data, fh)
        copied = os.path.join(tmp, os.path.basename(model))
        assert read_interface_confidence(copied) is None
        merged = _read_interface_confidence_or_warn(copied, compute_scores=True)
    assert merged['tool'] == 'colabfold' and set(merged['scores']) == set(COMPUTED_SCORES)


def test_zip_extracts_pae_only_on_request():
    _need(DECOY_ZIP)
    with tempfile.TemporaryDirectory() as tmp:
        files, _, _ = find_structure_files(DECOY_ZIP, extract_dir=os.path.join(tmp, 'a'))
        assert not glob.glob(os.path.join(tmp, 'a', '**', '*full_data*'), recursive=True)
        assert compute_interface_scores(files[0]) is None
        files, _, _ = find_structure_files(DECOY_ZIP, extract_dir=os.path.join(tmp, 'b'),
                                           include_pae=True)
        assert len(files) == 5
        scores = compute_interface_scores(files[0])['scores']
    # the decoy (chain H, a known non-binder) scores below every real Arp2/3 contact
    decoy = max(v for (a, b), v in scores['ilis'].items() if 'H' in (a, b))
    assert decoy < scores['ilis'][('A', 'B')] and decoy < scores['ilis'][('D', 'F')]


def test_pae_that_does_not_fit_only_drops_computed_scores():
    _need(ARP23)
    model = _arp23_models()['colabfold'][0]
    scores_file = find_pae_file(model)[0]
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(model, tmp)
        with open(scores_file) as fh:
            data = json.load(fh)
        data['pae'] = [row[:-1] for row in data['pae'][:-1]]
        with open(os.path.join(tmp, os.path.basename(scores_file)), 'w') as fh:
            json.dump(data, fh)
        copied = os.path.join(tmp, os.path.basename(model))
        try:
            compute_interface_scores(copied)
            raise AssertionError('PAE of the wrong size was not caught')
        except ValueError:
            pass
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            merged = _read_interface_confidence_or_warn(copied, compute_scores=True)
        assert any('not computed' in str(w.message) for w in caught)
    assert set(merged['scores']) == {'ipsae', 'pdockq', 'pdockq2'}


def test_non_protein_chains_are_left_out():
    '''AlphaFold3 RNase P + H1 RNA + Zn/Mg: 11 protein chains, 3 ions, 1 RNA chain.'''
    _need(RNASEP_ZIP)
    with tempfile.TemporaryDirectory() as tmp:
        files, _, _ = find_structure_files(RNASEP_ZIP, extract_dir=tmp, include_pae=True)
        model = [f for f in files if f.endswith('_model_0.cif')][0]
        merged = _read_interface_confidence_or_warn(model, compute_scores=True)['scores']
    chains = {chain for pair in merged['ilis'] for chain in pair}
    assert chains == set('ABCDEFGHIJK') and len(merged['ilis']) == 11 * 10
    assert ('A', 'O') in merged['pair_iptm'] and ('L', 'M') not in merged['pair_iptm']


def test_experimental_structure_has_none():
    pdb = os.path.join(HERE, '..', 'TESTING', 'pdb6nmi.ent')
    _need(pdb)
    assert find_pae_file(pdb) == (None, None)
    assert compute_interface_scores(pdb) is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert _read_interface_confidence_or_warn(pdb, compute_scores=True) is None
    assert any('no PAE file' in str(w.message) for w in caught)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'PASS {name}')
            except Skip as e:
                print(f'SKIP {name} ({e})')
