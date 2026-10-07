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

`--min-score suggested` uses a suggested `ipsae_calc` cutoff that depends on the predictor, because
the predictors' scores are on different scales:

| Predictor | Suggested `ipsae_calc` cutoff |
|---|---|
| Boltz | 0.3 |
| AlphaFold3 | 0.2 |
| ColabFold | none (not calibrated; the filter stays off, with a warning) |

The values were chosen on a calibration set (small human complexes of known structure, each
folded together with an unrelated protein, and pairs of unrelated proteins folded alone; 72
unrelated pairs per predictor).

**Boltz (0.3) has since been tested on pairs it was not chosen on:** 200 pairs of unrelated
proteins and 100 known pairs (a BioPlex interaction in direct contact in an experimental
structure), each folded as a pair with ten models.

* With the default contact parameters, a contact was called in the top-ranked model for about 40
  in 100 unrelated pairs. Adding ipSAE >= 0.3 left 15 to 19 in 100, and kept 72 of the 75 known
  pairs that were called and correctly placed.
* Small pairs are called far more often than large ones: about three quarters of unrelated pairs
  of 300 residues or fewer, against about one in five above 600.
* A contact that passes the cutoff is not thereby correctly placed: of 23 known pairs that Boltz
  placed wrongly, 9 passed.
* The cutoff reduces false calls; it does not make a single call reliable. Agreement between the
  models of a run (next section) and independent evidence are stronger: none of the unrelated
  pairs is a BioPlex interaction (`bioplex_293T`, `bioplex_HCT116`).
* No higher cutoff does better: up to 0.6, unrelated pairs and known pairs are lost together.

**AlphaFold3 (0.2) has not been tested this way.** The value rests on 38 calibration jobs (72
unrelated pairs) only. There it removed most contacts with an unrelated protein, but one
unrelated pair was predicted confidently in every model of five seeds (ipSAE up to 0.45), and
the models of one AlphaFold3 run agree with each other, so counting models does not help. Treat
it as provisional.

The values are a single table, `SUGGESTED_MIN_SCORE` in `bioplexpy/analysis_funcs.py`.

* A hidden contact is left out of the figure, whose title says how many were hidden. The tables keep
  it: `structure_contact` is unchanged and a `passes_filter` column says `False`.
  `summary_contacts.tsv` gains `n_structures_pass`.
* A contact that has no score is kept and `passes_filter` is left empty. This applies to a model
  without a PAE file and to contacts with a nucleic acid chain.
* ipSAE differs by direction (A→B and B→A). The filter tests the larger of the two
  (`--filter-reduce max`); `mean` and `min` are available. This choice is provisional and still to be
  confirmed with the BioPlex3D authors.

In Python: `filter_contacts_by_score()`, and the `min_score` (a number or `'suggested'`),
`filter_score` and `filter_reduce` arguments of the render functions.

### Several models of one prediction

Give every model of a run, not only the top-ranked one. After a run on more than one structure,
`bioplexpy-structure` prints for each protein pair in how many of them it is a direct contact
(and, with `--min-score`, in how many it passes the filter); `summary_contacts.tsv` has the same
numbers as `n_structures_contact` and `n_structures_pass`:

```
Contacts across the 10 structures:
  CDK4-LAMTOR4: contact in 7 of 10 (6 pass the score filter)
```

This is information, not a filter. What it meant in the Boltz test above (ten models per pair):

* A known pair that Boltz placed correctly was a contact in all ten models in 70 of 77 cases.
* Of the 85 unrelated pairs called in at least one model, 58 were called in four models or
  fewer, and 8 in all ten.
* Counting a pair only if it is a contact in at least five of ten models left 10 to 17 in 100
  unrelated pairs, and 3 to 6 in 100 with ipSAE >= 0.3 as well, for 68 of 100 known pairs kept.
* The top-ranked model is the wrong one to judge from alone: where only some models had the false
  contact, the top-ranked model was nearly always one of them. Boltz ranks its models by a
  confidence score that rewards a docked chain.

For Boltz, ten models per prediction (`--diffusion_samples 10`) are recommended. These numbers are
from one seed; how much the count changes between seeds has not been measured yet.

### Looking at the models in a browser

`--viewer` also writes `viewer/index.html` in the output folder, a page for the models of one
prediction together:

```
bioplexpy-structure boltz_results_run/ --uniprots P61158 P61160 Q92747 O15144 O15145 P59998 O15511 \
    --compute-scores --viewer --out-dir results/
```

Open `results/viewer/index.html` in a browser. It shows:

* the structure in [Mol\*](https://molstar.org), as a cartoon in Mol\*'s illustrative style,
  colored by chain or by pLDDT, with a button per
  model (the models are superposed on the first one, so the view holds still when you switch);
* the sequence of every chain below the structure, colored as the structure is. Pointing at a
  letter highlights the residue in the structure, and the other way round; the interface
  residues of the selected pair are bold and underlined;
* the three network panels of the figure, with each protein at the centre of its chain(s): they
  turn as the structure is turned. In the model panel a line's width is the number of models that
  have the contact;
* the PAE of the model shown, as a heatmap. Clicking a cell marks the two residues in the structure;
* a table of the protein pairs: in how many models each is a contact, its score in the model
  shown, and whether BioPlex detected it. Clicking a pair (or its line in a network) marks its
  interface residues in the structure and its blocks in the heatmap.

Each contact-rule button shows how many protein pairs are a contact in at least one model under
that rule. If the rule in use finds none and another rule does, a line under the controls says so.

The two sliders hide contacts found in fewer than a number of models, or scoring below a cutoff;
they change what is drawn, not what was measured. What counts as a contact is stated at the top of
the page (hover over a rule's button to read it): it opens with the rule the tables were made
with (the command line's), and a button switches to the other preset (the BioPlex 3.0 paper's any
two atoms within 6 A, or BioPlex3D's), for which the contacts are worked out when the page is
written.

The page is a folder of plain files that needs no server. It loads Mol\* from the web
(cdn.jsdelivr.net), so the structure panel needs a connection; the rest works without one. Each
model's coordinates and PAE are in a file of their own, read when that model is first shown. All
the structures given must be models of the same prediction.

Predicted structures are subject to their predictor's terms; AlphaFold Server output is for
non-commercial use only (see the `terms_of_use.md` inside the zip).

## Running Tests
-------------

* `python -m doctest BioPlexPy/` to run the respective test for the file
*Warning*: By default, doctests will not print anything if tests are successful.
Add the `-v` option to print verbose output.

