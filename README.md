# RBF reproducibility — code, version 1.0.0

Creators: Fábio Jorge de Nazaré Ferreira and Fernanda Guilhon-Simplicio, Universidade Federal do Amazonas.

This repository exposes the scientific source code and configuration for **Classificação integrada das contribuições termodinâmica e difusiva em um modelo de permeação por membranas lipídicas**. The full scientific distribution includes large archived datasets and figures and is deposited separately on Zenodo. Full reproducibility dataset (version 1.0.0): https://doi.org/10.5281/zenodo.23128726

## Verify the archived results

Download and extract the full Zenodo ZIP. Install `requirements-verificacao.txt` in a Python 3.12 environment. From this code repository, run:

```bash
python -m pip install -r requirements-verificacao.txt
python scripts/verificar_reproducibilidade.py --base /absolute/path/RBF_REPRODUCIBILIDADE_v1.0.0 --out ../RBF_VERIFICACAO_LOCAL.json
```

Replace the example path with the actual extracted directory. Expected status: `PASS_DOCUMENTAL_E_NUMERICO`. This checks files and reported statistics; no campaign is executed. Do not use Python `-O`.

The code repository alone has no large result tables or scientific evidence/input-hash files. For regeneration, use a working copy of the full Zenodo distribution and follow its README and methods. Do not execute the stage scripts directly in this lightweight checkout. Code/configuration files shared by both distributions are byte-identical.

`MANIFESTO_CODIGO.json` covers scientific source/configuration files; it intentionally excludes mutable citation and README metadata. `scripts/verificar_codigo.py` checks those source hashes without extra dependencies. Full archived results: 56,926 accepted profiles from 65,536 candidates; integrated/proxy disagreement 7.68%. Estimated numerical errors are not certified; physical uncertainty is unquantified; BD covers selected controls and the new boundary control is unresolved. FP/BD do not establish experimental validation.

The counterfactual antecedent https://doi.org/10.1021/acs.jpcb.6c02574 is credited and is not this dataset's DOI. Code/tests: MIT. Original data/configurations/documentation: CC BY 4.0. See license files.
