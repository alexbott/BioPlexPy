# Example prediction: human GINS complex (Boltz)

A small predicted structure for trying `bioplexpy-structure` and the functions in
`TESTING BioPlex User Structure Funcs.ipynb` without running a predictor yourself.

- **Complex:** human GINS tetramer, full-length UniProt sequences: GINS1 (Q14691, chain A),
  GINS2 (Q9Y248, B), GINS3 (Q9BRX5, C), GINS4 (Q9BRT9, D); 820 residues.
- **Predictor:** [Boltz](https://github.com/jwohlwend/boltz) 2.2.1, run 2026-10-02 from
  `gins_2E9X_boltz.yaml` (3 recycling steps, 200 sampling steps). Boltz's code and weights are
  released under the MIT license.
- **MSAs:** precomputed alignments for the four proteins, downloaded from the AlphaFold Protein
  Structure Database, were supplied to the run; the Boltz MSA server was not used. The MSAs are not
  included here.
- **What is here:** model 0 of that run, in the folder layout Boltz writes, so the folder can be
  given to BioPlexPy as it is:
  - `predictions/gins_2E9X/gins_2E9X_model_0.cif`: the model;
  - `confidence_gins_2E9X_model_0.json`: Boltz's confidence scores, including chain-pair ipTM;
  - `pae_gins_2E9X_model_0.npz`, `plddt_gins_2E9X_model_0.npz`: PAE matrix and per-residue pLDDT,
    used by `--compute-scores`;
  - `processed/records/gins_2E9X.json`: Boltz's record of the chain names.
  The run's other models, MSAs and PDE files are left out to keep the example small.
- **Reference:** the crystal structure of the human GINS complex, PDB `2E9X`. All six protein
  pairs are in contact in the model and in 2E9X; BioPlex detects all six in 293T cells and five in
  HCT116.

```
bioplexpy-structure TESTING/example_boltz_gins \
    --uniprots Q14691 Q9Y248 Q9BRX5 Q9BRT9 \
    --compute-scores --reference 2E9X --out-dir results/
```

This is a prediction, provided as an example only.
