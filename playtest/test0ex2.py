# -*- coding: utf-8 -*-

"""
src:
- https://www1.swarthmore.edu/NatSci/peverso1/Sports%20Data/JamesSteinData/Efron-Morris%20Baseball/EfronMorrisBB.txt
- https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html
"""

import torch, pyro
import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

N = 18
at_bats = torch.tensor([45.]*N)
hits = torch.tensor([18., 17., 16., 15., 14., 14., 13., 12., 11., 11., 10., 10., 10., 10., 10., 9., 8., 7.])

def baseball_model(at_bats, hits):
	log_kappa = pyro.sample("log κ", dist.Exponential(1.5))
	kappa = pyro.deterministic("κ", torch.exp(log_kappa))
	phi = pyro.sample("φ", dist.Uniform(0., 1.))
	theta = pyro.sample("θ", dist.Beta(phi*kappa, phi*(1.-kappa)))
	with pyro.plate("data loop", N):
		pyro.sample("obs", dist.Binomial(at_bats, theta), obs=hits)

# pyro.render_model(baseball_model, model_args=(at_bats, hits), render_distributions=True, render_params=True)

mcmc = MCMC(
	kernel=NUTS(baseball_model, jit_compile=True, ignore_jit_warnings=True),
	num_samples=4000,
	warmup_steps=1000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc.run(at_bats, hits)

mcmc.summary()
mcmc.diagnostics()

posterior_samples = {k: v.to("cpu") for k, v in mcmc.get_samples().items()}

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["θ"])
_ = ax[1].hist(posterior_samples["θ"], bins=100, density=True)
