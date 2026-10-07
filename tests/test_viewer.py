'''
Checks for the data behind the --viewer page (bioplexpy/viewer.py): the
models of one job are put in one frame without being distorted, the PAE
survives its packing, the interface residues are the ones the contact rule
found, and the page's numbers are the tables' numbers. Uses the five Boltz
models of Arp2/3 kept outside the repo in ../fold_outputs/; tests whose data
isn't there are skipped. No network: the chain map is given and the BioPlex
tables are small stand-ins.

Run with pytest, or directly: python tests/test_viewer.py
'''

import glob
import io
import json
import os
import tempfile

import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser

from bioplexpy.analysis_funcs import (_ca_interface, _direct_interaction_chain_pairs,
                                      _load_pdb_model, compare_structure_contacts_to_BioPlex,
                                      PDB_to_interacting_chains_uniprot_maps, read_pae,
                                      resolve_contact_settings)
from bioplexpy.cli import _resolve_filter_args, build_parser, summarize_contacts
from bioplexpy.viewer import (ASSET_FILES, NUCLEIC_ACID_COLORS, PAE_STEP, _protein_ca, build_viewer_data,
                              contact_rules, interface_residues, kabsch, one_copy_per_protein,
                              pack_pae,
                              unpack_pae, write_viewer)

HERE = os.path.dirname(__file__)
FOLDS = os.environ.get('BIOPLEXPY_FOLD_OUTPUTS', os.path.join(HERE, '..', '..', 'fold_outputs'))
ARP23_BOLTZ = os.path.join(FOLDS, 'arp23_6YW7', 'boltz')
ARP23_CHAINS = dict(zip('ABCDEFG', ['P61158', 'P61160', 'Q92747', 'O15144', 'O15145',
                                    'P59998', 'O15511']))
SYMBOLS = dict(zip(ARP23_CHAINS.values(), ['ACTR3', 'ACTR2', 'ARPC1A', 'ARPC2', 'ARPC3',
                                           'ARPC4', 'ARPC5']))
CONTACT = resolve_contact_settings('bioplex3d')


class Skip(Exception):
    pass


def _need(path):
    if not os.path.exists(path):
        try:
            import pytest
            pytest.skip(f'missing {path}')
        except ImportError:
            raise Skip(path)


def _bioplex(pairs):
    # a stand-in BioPlex table that names every Arp2/3 protein, so that no
    # gene symbol has to be looked up in UniProt
    ids = list(SYMBOLS)
    rows = [(a, b) for a, b in pairs] + [(i, i) for i in ids]
    return pd.DataFrame({'UniprotA': [a for a, _ in rows], 'UniprotB': [b for _, b in rows],
                         'SymbolA': [SYMBOLS[a] for a, _ in rows],
                         'SymbolB': [SYMBOLS[b] for _, b in rows]})


BP_293T = _bioplex([('P61158', 'P61160'), ('O15144', 'P61158')])
BP_HCT116 = _bioplex([('P61158', 'P61160')])


def _arp23(n_models=3, preset='bioplex3d'):
    _need(ARP23_BOLTZ)
    files = sorted(glob.glob(os.path.join(ARP23_BOLTZ, '*', 'predictions', '*',
                                          '*_model_?.cif')))[:n_models]
    names = {f: os.path.splitext(os.path.basename(f))[0] for f in files}
    tables = {}
    for f in files:
        maps = PDB_to_interacting_chains_uniprot_maps(f, None, chain_to_uniprot=ARP23_CHAINS,
                                                      contact_preset=preset)
        tables[names[f]] = compare_structure_contacts_to_BioPlex(*maps, BP_293T, BP_HCT116)
    return files, names, tables


def _built(n_models=3):
    # the page's data under one contact rule (BioPlex3D's, the default)
    files, names, tables = _arp23(n_models)
    summary = summarize_contacts(tables, BP_293T, BP_HCT116)
    rules = [dict(id='bioplex3d', label='BioPlex3D', contact=CONTACT, tables=tables,
                  summary=summary)]
    job, models = build_viewer_data(files, names, rules, ARP23_CHAINS, BP_293T, BP_HCT116)
    return files, names, summary, job, models


def test_kabsch_recovers_a_rotation_and_shift():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(40, 3)) * 20
    axis = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(0.7) * k + (1 - np.cos(0.7)) * k @ k
    shift = np.array([5.0, -12.0, 3.0])
    found_rotation, found_shift = kabsch(points, points @ rotation.T + shift)
    assert np.allclose(found_rotation, rotation, atol=1e-8)
    assert np.allclose(found_shift, shift, atol=1e-8)
    # a mirror image cannot be reached by a rotation: the fit stays a rotation
    mirrored = points * [1, 1, -1]
    assert np.isclose(np.linalg.det(kabsch(points, mirrored)[0]), 1.0)


def test_pae_packing_round_trip():
    rng = np.random.default_rng(1)
    pae = rng.uniform(0.25, 31.5, size=(50, 50)).astype(np.float32)
    packed = pack_pae(pae)
    assert packed['n'] == 50 and packed['factor'] == 1
    assert np.abs(unpack_pae(packed) - pae).max() <= PAE_STEP / 2 + 1e-6
    # a matrix over the limit is averaged over blocks; row r is then row r // factor
    small = pack_pae(pae, max_rows=20)
    assert small['factor'] == 3 and small['n'] == 17
    assert abs(unpack_pae(small)[0, 0] - pae[:3, :3].mean()) <= PAE_STEP / 2 + 1e-6
    assert abs(unpack_pae(small)[16, 16] - pae[48:, 48:].mean()) <= PAE_STEP / 2 + 1e-6


def test_models_are_superposed_without_distortion():
    files, names, summary, job, models = _built()
    parser = MMCIFParser(QUIET=True)

    def moved_ca(index):
        return _protein_ca(parser.get_structure('m', io.StringIO(models[index]['structure']))[0])

    reference = moved_ca(0)
    keys = sorted(reference)
    for index in (1, 2):
        moved, original = moved_ca(index), _protein_ca(_load_pdb_model(files[index], None))
        after = np.sqrt(np.mean([np.sum((moved[k] - reference[k]) ** 2) for k in keys]))
        first = _protein_ca(_load_pdb_model(files[0], None))
        before = np.sqrt(np.mean([np.sum((original[k] - first[k]) ** 2) for k in keys]))
        # the page's frame is the fit: no worse than the files as they came,
        # and the RMSD the page reports (coordinates are written to 0.001 A)
        assert after <= before + 1e-6
        assert abs(after - job['models'][index]['rmsd_to_first']) < 0.02
        # moving a model must not change its shape
        sample = keys[::25]
        for a, b in zip(sample, sample[1:]):
            assert abs(np.linalg.norm(moved[a] - moved[b])
                       - np.linalg.norm(original[a] - original[b])) < 0.01
    assert job['models'][0]['rmsd_to_first'] == 0


def test_node_positions_are_in_the_same_frame_as_the_coordinates():
    files, names, summary, job, models = _built(2)
    parser = MMCIFParser(QUIET=True)
    for index in (0, 1):
        model = parser.get_structure('m', io.StringIO(models[index]['structure']))[0]
        for chain, uniprot in ARP23_CHAINS.items():
            centroid = np.mean([atom.coord for residue in model[chain] for atom in residue], axis=0)
            assert np.allclose(job['models'][index]['centroids'][uniprot], centroid, atol=0.02)


def test_a_protein_in_two_copies_is_placed_on_one_of_them():
    # two copies of P1 and of P2 around a twofold axis: the means of the copies would
    # coincide on the axis
    centroids = {'A': np.array([10., 0, 0]), 'B': np.array([12., 5, 0]),
                 'C': np.array([-10., 0, 0]), 'D': np.array([-12., -5, 0])}
    chains = {'A': ['P1'], 'B': ['P2'], 'C': ['P1'], 'D': ['P2']}
    kept = one_copy_per_protein(chains, centroids)
    assert sorted(kept) == ['A', 'B']        # the copies next to the first chain
    assert one_copy_per_protein({'A': ['P1'], 'B': ['P2']}, centroids).keys() == {'A', 'B'}


def test_interface_residues_are_the_contact_rule_s():
    files, names, tables = _arp23(1)
    model = _load_pdb_model(files[0], None)
    pae_data = read_pae(files[0], model=model)
    pairs = _direct_interaction_chain_pairs(model, CONTACT['distance'],
                                            min_plddt=CONTACT['min_plddt'],
                                            contact_atoms='ca', max_pae=CONTACT['max_pae'],
                                            pae_data=pae_data)
    interfaces = interface_residues(model, pairs, CONTACT, pae_data)
    assert [i['chains'] for i in interfaces] == pairs and pairs
    for interface in interfaces:
        chain_i, chain_j = interface['chains']
        mask = _ca_interface(model, chain_i, chain_j, pae_data, CONTACT['distance'],
                             CONTACT['min_plddt'], CONTACT['max_pae'])
        numbers_i = np.array([r.id[1] for r in model[chain_i]])
        numbers_j = np.array([r.id[1] for r in model[chain_j]])
        assert interface['residues'][0] == sorted(numbers_i[mask.any(axis=1)].tolist())
        assert interface['residues'][1] == sorted(numbers_j[mask.any(axis=0)].tolist())
        assert interface['residues'][0] and interface['residues'][1]


def test_page_numbers_equal_the_tables():
    files, names, summary, job, models = _built()
    order = [names[f] for f in files]
    assert [m['name'] for m in job['models']] == order
    assert job['confidence'] is True          # Boltz models with their PAE files
    by_pair = {frozenset((p['a'], p['b'])): p for p in job['rules'][0]['pairs']}
    assert len(by_pair) == len(summary)
    for row in summary.to_dict('records'):
        pair = by_pair[frozenset((row['UniprotA'], row['UniprotB']))]
        assert pair['n_contact'] == row['n_structures_contact'] == sum(pair['contact'])
        assert pair['contact'] == [bool(row[name]) for name in order]
        assert pair['bp293'] == bool(row['bioplex_293T'])
        assert pair['bpHct'] == bool(row['bioplex_HCT116'])
    # the edges the network panel draws are the same contacts, model by model
    for index, model in enumerate(job['models']):
        drawn = {frozenset(edge) for edge in model['edges']['bioplex3d']}
        assert drawn == {pair for pair, p in by_pair.items() if p['contact'][index]}
    assert {frozenset((e['a'], e['b'])) for e in job['bioplex_edges']} == {
        frozenset(('P61158', 'P61160')), frozenset(('O15144', 'P61158'))}
    assert {c['id']: c['ids'] for c in job['chains']} == {c: [u] for c, u in ARP23_CHAINS.items()}
    # white and the light greys are reserved for DNA and RNA chains
    assert not {c['color'].lower() for c in job['chains']} & set(NUCLEIC_ACID_COLORS)
    # PAE rows: every protein residue has one, and the matrix is the file's
    pae_data = read_pae(files[0])
    assert models[0]['pae_rows'] == pae_data['pae'].shape[0]
    assert np.abs(unpack_pae(models[0]['pae']) - np.minimum(pae_data['pae'], 255 * PAE_STEP)
                  ).max() <= PAE_STEP / 2 + 1e-6
    for chain in ARP23_CHAINS:
        assert len(models[0]['rows'][chain]) == len(models[0]['residues'][chain]) \
            == len(models[0]['plddt'][chain]) == len(models[0]['sequence'][chain])
    # one letter per residue, the amino acid's
    assert models[0]['sequence']['A'].startswith('M') and 'X' not in models[0]['sequence']['A']


def test_viewer_folder_is_written():
    files, names, tables = _arp23(2)
    args = build_parser().parse_args(['x', '--uniprots', 'P61158'])
    _resolve_filter_args(build_parser(), args)
    with tempfile.TemporaryDirectory() as tmp:
        page = write_viewer(files, names, tables, args, ARP23_CHAINS, BP_293T, BP_HCT116, tmp)
        folder = os.path.dirname(page)
        assert sorted(os.listdir(folder)) == sorted(ASSET_FILES + ('job.js', 'model_0.js',
                                                                 'model_1.js'))

        def data(name, target):
            text = open(os.path.join(folder, name)).read()
            head = f'window.BPV.{target} = '
            return json.loads(text[text.index(head) + len(head):].rstrip().rstrip(';'))

        job = data('job.js', 'job')
        assert job['title'] == 'arp23' and len(job['models']) == 2
        # the rule the tables were made with first, then the other preset
        assert [r['id'] for r in job['rules']] == ['bioplex3d', 'bioplex2021']
        assert job['rules'][0]['contact'] == {'contact_atoms': 'ca', 'distance': 8,
                                              'min_plddt': 50, 'max_pae': 10}
        model = data('model_1.js', 'models[1]')
        assert model['format'] == 'cif' and model['structure'].startswith('data_')
        assert model['interfaces']['bioplex3d'] and model['pae']['n'] == model['pae_rows']

        # the other rule's numbers are what the tables give under that rule
        other_tables = _arp23(2, preset='bioplex2021')[2]
        other = summarize_contacts(other_tables, BP_293T, BP_HCT116)
        order = [names[f] for f in files]
        pairs = {frozenset((p['a'], p['b'])): p for p in job['rules'][1]['pairs']}
        assert len(pairs) == len(other)
        for row in other.to_dict('records'):
            pair = pairs[frozenset((row['UniprotA'], row['UniprotB']))]
            assert pair['contact'] == [bool(row[name]) for name in order]
        for index in (0, 1):
            drawn = {frozenset(e) for e in job['models'][index]['edges']['bioplex2021']}
            assert drawn == {k for k, p in pairs.items() if p['contact'][index]}
            made_by = {frozenset(ARP23_CHAINS[c] for c in i['chains'])
                       for i in data(f'model_{index}.js', f'models[{index}]')
                       ['interfaces']['bioplex2021']}
            assert made_by == drawn
        # any atom within 6 A reaches further than CA atoms within 8 A with the
        # confidence conditions: the rules must not give the same contacts here
        assert sum(p['n_contact'] for p in pairs.values()) > sum(
            p['n_contact'] for p in job['rules'][0]['pairs'])


def test_contact_rules_offered():
    # the rule in use comes first; a setting that is no preset is offered with both presets
    assert [r['id'] for r in contact_rules(resolve_contact_settings('bioplex3d'))] \
        == ['bioplex3d', 'bioplex2021']
    assert [r['id'] for r in contact_rules(resolve_contact_settings('bioplex2021'))] \
        == ['bioplex2021', 'bioplex3d']
    custom = resolve_contact_settings('bioplex3d', distance=10)
    rules = contact_rules(custom)
    assert [r['id'] for r in rules] == ['command_line', 'bioplex3d', 'bioplex2021']
    assert rules[0]['contact'] == custom


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'PASS {name}')
            except Skip as e:
                print(f'SKIP {name} ({e})')
