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

Useful options: `--min-plddt 70` leaves low-confidence atoms out of the contact search;
`--compute-scores` also calculates interface scores from each model's PAE (pDockQ, pDockQ2,
LIS/cLIS/iLIS, ipSAE); `--reference 6NMI` adds an experimental structure's contacts to the summary
table; `--no-render` writes the tables only. See `bioplexpy-structure --help` and
`TESTING/TESTING BioPlex User Structure Funcs.ipynb`.

Predicted structures are subject to their predictor's terms; AlphaFold Server output is for
non-commercial use only (see the `terms_of_use.md` inside the zip).

## Running Tests
-------------

* `python -m doctest BioPlexPy/` to run the respective test for the file
*Warning*: By default, doctests will not print anything if tests are successful.
Add the `-v` option to print verbose output.

