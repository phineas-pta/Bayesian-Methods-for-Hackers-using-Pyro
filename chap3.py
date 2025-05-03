# -*- coding: utf-8 -*-

"""
data creation algorithm:
For each data point, choose cluster 1 with probability p, else choose cluster 2.
Draw a random variate from a Normal(μᵢ, σᵢ) distribution where i was cluster chosen previously in step 1.
"""

import torch, pyro
import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, config_enumerate

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
pyro.enable_validation(True)
pyro.clear_param_store()

with open("data/mixture_data.txt", mode="r") as f:
	loaded_list = f.read().splitlines()
data = torch.tensor([float(x) for x in loaded_list])

# enumeration allows Pyro to marginalize out discrete latent variables in HMC and SVI models
@config_enumerate
def mixture_model(data, num_mix=2):
	weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(num_mix)))
	sigma = pyro.sample("σ", dist.LogNormal(0., 100.))
	with pyro.plate("number mixtures", num_mix):
		mu = pyro.sample("μ", dist.Normal(155., 10.))
	with pyro.plate("data loop", len(data)):
		assignment = pyro.sample("assignment", dist.Categorical(weights))
		pyro.sample("obs", dist.Normal(mu[assignment], sigma), obs=data)

# pyro.render_model(mixture_model, model_args=(data,), render_distributions=True, render_params=True)

mcmc = MCMC(
	kernel=NUTS(mixture_model, max_plate_nesting=1, jit_compile=True, ignore_jit_warnings=True),
	num_samples=4000,
	warmup_steps=1000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc.run(data)

mcmc.summary()
mcmc.diagnostics()

posterior_samples = {k: v.to("cpu") for k, v in mcmc.get_samples().items()}

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["μ"][:, 0])
_ = ax[1].hist(posterior_samples["μ"][:, 0], bins=100, density=True)

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["μ"][:, 1])
_ = ax[1].hist(posterior_samples["μ"][:, 1], bins=100, density=True)

###############################################################################
# faster version: marginalize discrete param
# see https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html

def mixture_model_bis(data):
	weight_cluster1 = pyro.sample("𝒫₁", dist.Uniform(0., 1.))
	mu1 = pyro.sample("μ₁", dist.Normal(120., 10.))
	mu2 = pyro.sample("μ₂", dist.Normal(190., 10.))
	sigma = pyro.sample("σ", dist.LogNormal(0., 100.))
	with pyro.plate("data loop", len(data)):
		log_proba_cluster1 = torch.log(     weight_cluster1) + dist.Normal(mu1, sigma).log_prob(data)
		log_proba_cluster2 = torch.log(1. - weight_cluster1) + dist.Normal(mu2, sigma).log_prob(data)
		pyro.factor("obs", torch.logaddexp(log_proba_cluster1, log_proba_cluster2))

mcmc_bis = MCMC(
	kernel=NUTS(mixture_model_bis, jit_compile=True, ignore_jit_warnings=True),
	num_samples=4000,
	warmup_steps=1000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc_bis.run(data)
