# -*- coding: utf-8 -*-

"""
modelling daily stock return
stocks: AAPL, GOOG, TSLA, AMZN
date: 2012/09/01 → 2015/04/27

Suppose Sᵢ is the price of the stock on day i then the daily return on that day is:
rᵢ = Sᵢ / Sᵢ₋₁ - 1
"""

import json, torch, pyro
import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
pyro.enable_validation(True)
pyro.clear_param_store()

with open("data/stocks_AAPL_GOOG_TSLA_AMZN.json", "r", encoding="utf-8") as f:
	data_full = json.load(f)

expert_prior_mu = torch.tensor(data_full["expert_mu"])
expert_prior_sigma =torch.diag( torch.square(torch.tensor(data_full["expert_sigma"])))  # UNKNOWN: square or not ???
stocks_data = torch.tensor(data_full["observations"])
Nstocks = 4
I = torch.eye(Nstocks)

# may give error because wishart can return matrix with negative values
def stocks_model(stocks_data, expert_prior_mu, expert_prior_sigma):
	mu = pyro.sample("μ", dist.MultivariateNormal(expert_prior_mu, 10.*I))
	sigma = pyro.sample("Σ", dist.Wishart(10., expert_prior_sigma))
	with pyro.plate("data loop", len(stocks_data)):
		pyro.sample("obs", dist.MultivariateNormal(mu, sigma), obs=stocks_data)

# more efficient Cholesky decomposition
def stocks_model_bis(stocks_data, expert_prior_mu, expert_prior_sigma):
	mu = pyro.sample("μ", dist.MultivariateNormal(expert_prior_mu, 10.*I))
	omega = pyro.sample("Ω", dist.LKJCholesky(Nstocks, 2.))
	L = pyro.deterministic("L", torch.mm(expert_prior_sigma, omega) + 1e-6*I)
	# sigma = pyro.deterministic("Σ", torch.mm(L, L.T))
	with pyro.plate("data loop", len(stocks_data)):
		pyro.sample("obs", dist.MultivariateNormal(mu, scale_tril=L), obs=stocks_data)

# pyro.render_model(stocks_model_bis, model_args=(stocks_data, expert_prior_mu, expert_prior_sigma), render_distributions=True, render_params=True)

mcmc = MCMC(
	kernel=NUTS(stocks_model_bis, jit_compile=True, ignore_jit_warnings=True),
	num_samples=4000,
	warmup_steps=1000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc.run(stocks_data, expert_prior_mu, expert_prior_sigma)

mcmc.summary()
mcmc.diagnostics()

posterior_samples = {k: v.to("cpu") for k, v in mcmc.get_samples().items()}

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["μ"][:, 0])
_ = ax[1].hist(posterior_samples["μ"][:, 0], bins=100, density=True)

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["μ"][:, 1])
_ = ax[1].hist(posterior_samples["μ"][:, 1], bins=100, density=True)
