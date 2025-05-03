# -*- coding: utf-8 -*-

"""
On 1986/01/28, the 25th flight of the USA space shuttle program ended in disaster when one of the rocket boosters of the Shuttle Challenger exploded shortly after lift-off, killing all 7 crew members.
The presidential commission on the accident concluded that it was caused by the failure of an O-ring in a field joint on the rocket booster, and that this failure was due to a faulty design that made the O-ring unacceptably sensitive to a number of factors including outside temperature.
Of the previous 24 flights, data were available on failures of O-rings on 23, (one was lost at sea), and these data were discussed on the evening preceding the Challenger launch, but unfortunately only the data corresponding to the 7 flights on which there was a damage incident were considered important and these were thought to show no obvious trend.

observation: probability of damage incidents occurring increases as the outside temperature decreases:
probability = 1 / (1 + exp(α + β × temperature))
"""

import torch, pyro
import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
pyro.enable_validation(True)
pyro.clear_param_store()

temperature = torch.tensor([66., 70., 69., 68., 67., 72., 73., 70., 57., 63., 70., 78., 67., 53., 67., 75., 70., 81., 76., 79., 75., 76., 58.])
damaged     = torch.tensor([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.])

def challenger_model(temperature, damaged):
	alpha = pyro.sample("α", dist.Normal(0., 1000.))
	beta  = pyro.sample("β", dist.Normal(0., 1000.))
	with pyro.plate("data loop", len(temperature)):
		prob = pyro.deterministic("proba", torch.sigmoid(- alpha - beta * temperature))  # attention sign because sigmoid use reverse
		pyro.sample("damaged", dist.Bernoulli(prob), obs=damaged)

# pyro.render_model(challenger_model, model_args=(temperature, damaged), render_distributions=True, render_params=True)

mcmc = MCMC(
	kernel=NUTS(challenger_model, jit_compile=True, ignore_jit_warnings=True),
	num_samples=4000,
	warmup_steps=1000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc.run(temperature, damaged)

mcmc.summary()
mcmc.diagnostics()

posterior_samples = {k: v.to("cpu") for k, v in mcmc.get_samples().items()}

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["α"])
_ = ax[1].hist(posterior_samples["α"], bins=100, density=True)

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["β"])
_ = ax[1].hist(posterior_samples["β"], bins=100, density=True)

predictions = Predictive(challenger_model, mcmc.get_samples())(torch.tensor([31.]), None)
_ = plt.hist(predictions["proba"].squeeze().to("cpu"), bins=100)
