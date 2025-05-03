# -*- coding: utf-8 -*-

"""
You are given a series of daily text-message counts from a user of your system. You are curious to know if the user’s text-messaging habits have changed over time, either gradually or suddenly. How can you model this?

text-message count of day i: Cᵢ ~ Poisson(λ)
λ = λ₁ if day < τ, λ₂ if day ≥ τ, with τ = switchpoint (user’s text-messaging habits change), if λ₁ = λ₂ then no change
λ₁ ~ Exp(α) and λ₂ ~ Exp(α)
τ ~ Unif(1, N)
"""

import torch, pyro
import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, RandomWalkKernel

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
pyro.enable_validation(True)
pyro.clear_param_store()

count_data = torch.tensor([
	13., 24.,  8., 24.,  7., 35., 14., 11., 15., 11., 22., 22., 11., 57., 11., 19., 29.,  6., 19., 12., 22., 12., 18., 72., 32.,  9.,  7., 13.,
	19., 23., 27., 20.,  6., 17., 13., 10., 14.,  6., 16., 15.,  7.,  2., 15., 15., 19., 70., 49.,  7., 53., 22., 21., 31., 19., 11., 18., 20.,
	12., 35., 17., 23., 17.,  4.,  2., 31., 30., 13., 27.,  0., 39., 37.,  5., 14., 13., 22.,
])

def sms_model(count_data):
	N = len(count_data)
	alpha = 1. / torch.mean(count_data)
	lambda1 = pyro.sample("λ₁", dist.Exponential(alpha))
	lambda2 = pyro.sample("λ₂", dist.Exponential(alpha))
	tau = pyro.sample("τ", dist.Uniform(1., N))
	with pyro.plate("data loop", N):
		lambda_ = torch.where(tau < torch.arange(N), lambda1, lambda2)
		pyro.sample("obs", dist.Poisson(lambda_), obs=count_data)

# pyro.render_model(sms_model, model_args=(count_data,), render_distributions=True, render_params=True)

# `pyro` can be very slow with HMC/NUTS sampling sometimes if model too complicated
# alternative: `numpyro` but cannot yet use cuda on windows native because of `jax`
mcmc = MCMC(
	kernel=RandomWalkKernel(sms_model),
	num_samples=40000,
	warmup_steps=10000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc.run(count_data)

mcmc.summary()
mcmc.diagnostics()

thinning = 5
posterior_samples = {k: v[::thinning].to("cpu") for k, v in mcmc.get_samples().items()}

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["λ₁"])
_ = ax[1].hist(posterior_samples["λ₁"], bins=100, density=True)

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["λ₂"])
_ = ax[1].hist(posterior_samples["λ₂"], bins=100, density=True)

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["τ"])
_ = ax[1].hist(posterior_samples["τ"], bins=100, density=True)

###############################################################################
# faster version: marginalize discrete param
# see https://mc-stan.org/docs/stan-users-guide/latent-discrete.html

def sms_model_bis(count_data):
	lambda1 = pyro.sample("λ₁", dist.Gamma(8., .3))
	lambda2 = pyro.sample("λ₂", dist.Gamma(8., .3))
	N = len(count_data)
	with pyro.plate("data loop", N):
		lp1 = torch.cat([torch.tensor([0.]), torch.cumsum(dist.Poisson(lambda1).log_prob(count_data), dim=0)])
		lp2 = torch.cat([torch.tensor([0.]), torch.cumsum(dist.Poisson(lambda2).log_prob(count_data), dim=0)])
		lp = lp2[-1] - torch.log(torch.tensor(N)) + lp1[:N] - lp2[:N]
		pyro.factor("obs", torch.logsumexp(lp, dim=0))
		return pyro.sample("τ", dist.Categorical(logits=lp))

mcmc_bis = MCMC(
	kernel=NUTS(sms_model_bis, jit_compile=True, ignore_jit_warnings=True),
	num_samples=40000,
	warmup_steps=10000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc_bis.run(count_data)
