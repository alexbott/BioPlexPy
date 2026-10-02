# BioPlexPy
-------------
[![Documentation Status](https://readthedocs.org/projects/bioplexpy/badge/?version=latest)](http://bioplexpy.readthedocs.io/?badge=latest)

Python-side access to PPI data from Gygi lab

## Citation
-------------

If you use the BioPlexPy package in published research, please cite:

Ludwig Geistlinger, Roger Vargas Jr, Tyrone Lee, Joshua Pan, Edward Huttlin, Robert Gentleman (2023) BioPlexR and BioPlexPy: integrated data products for the analysis of human protein interactions. *Bioinformatics*. doi: [10.1093/bioinformatics/btad091](https://doi.org/10.1093/bioinformatics/btad091). 

## Installation
-------------
The package can be installed through pypi or through github

`pip install bioplexpy`
or 
`pip install git+https://github.com/ccb-hms/BioPlexPy.git#egg=BioPlexPy`


## Usage
-------------
See the [BioPlex examples notebook](https://github.com/ccb-hms/BioPlexPy/blob/main/docs/BioPlex_Examples.ipynb) for basic usage and how to obtain
BioPlex datasets.

https://bioplexpy.readthedocs.io/en/latest/

## Your own structures and predictions
-------------
`bioplexpy-structure` compares a structure with BioPlex interactions in 293T and HCT116 cells:
which chains are in direct contact, which of those pairs BioPlex detected, and a Figure 2-style
panel for each model. Give it the proteins in the complex and the structure, as the tool wrote it.

**AlphaFold Server: use the downloaded `.zip` as it is.** No need to unpack it:

```
bioplexpy-structure fold_my_complex.zip \
    --uniprots P61158 P61160 Q92747 O15144 O15145 P59998 O15511 \
    --out-dir results/
```

All five models are read (template hits are ignored), chains are matched to the proteins by
sequence, and the predictor's chain-pair ipTM is shown on the model network. The zip is only read,
never changed.

The same command takes:

* a Boltz output folder (`boltz_results_<name>/`) or a ColabFold output folder, as written;
* single `.pdb` / `.cif` files, experimental or predicted, or a folder of them;
* several of these at once, e.g. to compare predictors.

Useful options: `--compute-scores` also calculates interface scores from each model's PAE (pDockQ,
pDockQ2, LIS/cLIS/iLIS, ipSAE); `--reference 6NMI` adds an experimental structure's contacts to the
summary table; `--no-render` writes the tables only. See `bioplexpy-structure --help` and
`TESTING/TESTING BioPlex User Structure Funcs.ipynb`.

### What counts as a direct contact

By default two protein chains are a direct contact with the parameters of the BioPlex3D pipeline:
at least one residue pair with

* CA atoms closer than 8 Å,
* both residues with pLDDT ≥ 50, and
* PAE ≤ 10 in at least one direction.

The pLDDT and PAE conditions need the PAE file the predictor wrote next to the model. A structure
without one, such as an experimental structure, is judged on the CA distance alone. Contacts with a
nucleic acid chain, which BioPlex3D does not define, use any two atoms closer than 6 Å.

Each parameter can be changed: `--contact-atoms ca|any`, `--distance`, `--min-plddt` and
`--max-pae` (`none` switches a condition off).

`--contact-preset bioplex2021` sets them all to the rule of the BioPlex 3.0 paper (Huttlin et al.
2021), which BioPlexPy used before: any two atoms closer than 6 Å, with no pLDDT or PAE condition.
Use it to reproduce the paper's Figure 2.

In Python the render functions take the same settings (`contact_preset`, `contact_atoms`,
`interact_dist_threshold`, `min_plddt`, `max_pae`) with the same defaults.
`get_interacting_chains_from_PDB()` and `PDB_to_interacting_chains_uniprot_maps()`, which take the
distance as an argument, keep the paper's any-atom rule unless given `contact_atoms='ca'` or a
`contact_preset`.

### Filtering contacts by score (optional)

`--min-score VALUE` is a quality filter applied on top of the contact parameters: it hides contacts
whose interface score is below `VALUE`. The score is `ipsae_calc` (ipSAE computed from the PAE)
unless `--filter-score NAME` names another score column, e.g. `pair_iptm`. It is off unless given.

* A hidden contact is left out of the figure, whose title says how many were hidden. The tables keep
  it: `structure_contact` is unchanged and a `passes_filter` column says `False`.
  `summary_contacts.tsv` gains `n_structures_pass`.
* A contact that has no score is kept and `passes_filter` is left empty. This applies to a model
  without a PAE file and to contacts with a nucleic acid chain.
* There is no default cutoff yet. In the one negative control run so far (HSD17B14 folded together
  with the Arp2/3 complex, AlphaFold3 and Boltz, five models each) the default contact parameters
  already leave out every contact with the unrelated protein. Under the 2021 rule those contacts
  are present and all have `ipsae_calc` 0; the lowest value on any other model contact is 0.12, and
  the lowest on a contact that is also in the experimental structure is 0.23.
* ipSAE differs by direction (A→B and B→A). The filter tests the larger of the two
  (`--filter-reduce max`); `mean` and `min` are available. This choice is provisional and still to be
  confirmed with the BioPlex3D authors.

In Python: `filter_contacts_by_score()`, and the `min_score`, `filter_score` and `filter_reduce`
arguments of the render functions.

Predicted structures are subject to their predictor's terms; AlphaFold Server output is for
non-commercial use only (see the `terms_of_use.md` inside the zip).

## Running Tests
-------------

* `python -m doctest BioPlexPy/` to run the respective test for the file
*Warning*: By default, doctests will not print anything if tests are successful.
Add the `-v` option to print verbose output.

