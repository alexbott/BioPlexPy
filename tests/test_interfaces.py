'''
Checks for what an interface is made of (bioplexpy/interfaces.py): the
residue pairs that pass are the ones the contact rule finds, the close pairs
that fail are accounted for by the pLDDT and PAE conditions, and the numbers
in the tables follow from the arrays. Uses the GINS Boltz model shipped in
TESTING/.

Run with pytest, or directly: python tests/test_interfaces.py
'''

import os

import numpy as np

from bioplexpy.analysis_funcs import (_ca_interface, _direct_interaction_chain_pairs,
                                      _load_pdb_model, read_pae, resolve_contact_settings)
from bioplexpy.interfaces import (interface_details, interface_residue_rows, interface_tables,
                                  structure_interfaces, summarize_interfaces)
from bioplexpy.viewer import interface_residues

HERE = os.path.dirname(__file__)
GINS = os.path.join(HERE, '..', 'TESTING', 'example_boltz_gins', 'predictions', 'gins_2E9X',
                    'gins_2E9X_model_0.cif')
CHAINS = dict(zip('ABCD', (['Q14691'], ['Q9Y248'], ['Q9BRX5'], ['Q9BRT9'])))


def _gins(preset, **changes):
    contact = resolve_contact_settings(preset, **changes)
    model = _load_pdb_model(GINS, None)
    pae = read_pae(GINS, model=model)
    rule_pae = pae if contact['contact_atoms'] == 'ca' else None
    pairs = _direct_interaction_chain_pairs(
        model, contact['distance'], min_plddt=contact['min_plddt'],
        contact_atoms=contact['contact_atoms'], max_pae=contact['max_pae'], pae_data=rule_pae)
    return contact, model, pae, pairs, interface_details(model, contact, pae, pairs)


def test_passing_pairs_are_the_contact_rule_s():
    contact, model, pae, pairs, details = _gins('bioplex3d')
    assert len(details) == 6 and {tuple(d['chains']) for d in details} == {tuple(p) for p in pairs}
    for d in details:
        a, b = d['chains']
        assert np.array_equal(d['passes'], _ca_interface(model, a, b, pae, 8, 50, 10))
        # passing pairs are close, and every close pair that fails does so on pLDDT or PAE
        assert not (d['passes'] & ~d['close']).any()
        assert np.array_equal(d['close'] & ~d['passes'], d['low_plddt'] | d['high_pae'])
        assert (d['pae'][d['passes']] <= 10).all()
    # the residues with a passing partner are the ones the page marks
    marked = {tuple(f['chains']): f['residues'] for f in interface_residues(model, pairs, contact, pae)}
    for d in details:
        rows = interface_residue_rows(d)
        assert [[r['number'] for r in side if r['n_pass']] for side in rows] == marked[tuple(d['chains'])]


def test_a_pair_that_fails_the_rule_is_still_described():
    # a PAE no pair can meet: nothing passes, and every close pair is counted as failing on it
    contact, model, pae, pairs, details = _gins('bioplex3d', max_pae=0.01)
    assert pairs == [] and len(details) == 6
    for d in details:
        s = summarize_interfaces([d])
        assert s['n_pass'] == 0 and s['n_close'] > 0 and s['n_high_pae'] == s['n_close']
        assert s['pae_pass'] is None and s['pae_close'][1] > 0.01 and s['residues'] == [0, 0]
        rows = interface_residue_rows(d)
        assert rows[0] and all(r['n_pass'] == 0 and r['n_close'] > 0 for r in rows[0] + rows[1])


def test_any_atom_rule_has_no_failing_pairs():
    contact, model, pae, pairs, details = _gins('bioplex2021')
    assert {tuple(d['chains']) for d in details} == {tuple(p) for p in pairs}
    for d in details:
        s = summarize_interfaces([d])
        assert s['n_pass'] == s['n_close'] > 0 and s['n_low_plddt'] is None and s['n_high_pae'] is None
        assert (d['distance'][d['close']] < 6).all()
        # the PAE is reported all the same
        assert s['pae_pass'] is not None and s['plddt'][0] > 50


def test_summary_numbers_follow_from_the_arrays():
    contact, model, pae, pairs, details = _gins('bioplex3d')
    d = details[0]
    s = summarize_interfaces([d])
    assert s['n_pass'] == d['passes'].sum() and s['n_close'] == d['close'].sum()
    assert s['residues'] == [d['passes'].any(axis=1).sum(), d['passes'].any(axis=0).sum()]
    values = d['pae'][d['passes']]
    assert s['pae_pass'] == [round(float(np.median(values)), 1), round(float(values.min()), 1)]
    # the other chain as side A: the two sides change places
    turned = summarize_interfaces([d], [True])
    assert turned['residues'] == s['residues'][::-1] and turned['plddt'] == s['plddt'][::-1]
    row = interface_residue_rows(d)[0][0]
    k = d['numbers'][0].index(row['number'])
    partner = d['numbers'][1].index(row['partner'])
    assert d['passes'][k, partner] and row['distance'] == round(float(d['distance'][k, partner]), 1)
    assert row['distance'] == round(float(d['distance'][k][d['passes'][k]].min()), 1)


def test_tables():
    contact, model, pae, pairs, details = _gins('bioplex3d')
    interfaces, residues = interface_tables(details, CHAINS, {'Q14691': 'GINS1'})
    assert len(interfaces) == 6 and interfaces.contact.all()
    assert list(interfaces.UniprotA[:1]) == ['Q14691'] and list(interfaces.SymbolA[:1]) == ['GINS1']
    assert (interfaces.residue_pairs_pass <= interfaces.residue_pairs_close).all()
    assert len(residues) == sum(len(side) for d in details for side in interface_residue_rows(d))
    assert residues.passes.sum() == interfaces.residues_A.sum() + interfaces.residues_B.sum()
    # from the file, as the command line does
    again, _ = structure_interfaces(GINS, contact, CHAINS, {'Q14691': 'GINS1'})
    assert again.equals(interfaces)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS {name}')
