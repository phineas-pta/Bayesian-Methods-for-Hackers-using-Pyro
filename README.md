# Bayesian Methods for Hackers using `pyro`

original book: https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers

here are just code, not notebook, comments whatsoever

DISCLAIMER: just an attempt, so not official, neither guaranteed to work on your machine

```bash
pip install torch pyro-ppl matplotlib --extra-index-url https://download.pytorch.org/whl/cu128
```
optional: `pip install grapviz`

### remarks

`pyro.plate` express conditional independence among random variables, e.g. in a loop over dataset, ref: https://en.wikipedia.org/wiki/Plate_notation

`numpyro` is much faster in sampling (so more suitable for bayesian statistics) but `pip install "numpyro[cuda]"` not yet working on windows+cuda natively (need WSL)
