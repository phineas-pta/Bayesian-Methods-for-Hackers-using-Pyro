# -*- coding: utf-8 -*-

"""
src:
- https://www.kaggle.com/c/overfitting
- http://timsalimans.com/winning-the-dont-overfit-competition/ (dead link)
- https://web.archive.org/web/20190718145349/http://timsalimans.com/winning-the-dont-overfit-competition/

In order to achieve this we have created a simulated data set with 200 variables and 20000 cases.
An ‘equation’ based on this data was created in order to generate a Target to be predicted.
Given the all 20000 cases, the problem is very easy to solve - but you only get given the Target value of 250 cases - the task is to build a model that gives the best predictions on the remaining 19750 cases.
"""

import json, torch, pyro
import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
pyro.enable_validation(True)
pyro.clear_param_store()

with open("data/overfitting.json", "r", encoding="utf-8") as f:
	data = json.load(f)

y = torch.tensor(data["y"])
X = torch.tensor(data["X"])  # shape: 250×200
new_X = torch.tensor(data["new_X"])  # shape: 19750×200

def overfit_model(X, y):
	alpha = pyro.sample("α", dist.Cauchy(0., 10.))
	beta = pyro.sample("β", dist.StudentT(1).expand([X.shape[1]]))
	with pyro.plate("data loop", X.shape[0]):
		logits = alpha + torch.mv(X, beta)
		pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

# pyro.render_model(overfit_model, model_args=(X, y), render_distributions=True, render_params=True)

mcmc = MCMC(
	kernel=NUTS(overfit_model, jit_compile=True, ignore_jit_warnings=True),
	num_samples=4000,
	warmup_steps=1000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc.run(X, y)

mcmc.summary()
mcmc.diagnostics()

posterior_samples = {k: v.to("cpu") for k, v in mcmc.get_samples().items()}

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["α"])
_ = ax[1].hist(posterior_samples["α"], bins=100, density=True)

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["β"][:, 1])
_ = ax[1].hist(posterior_samples["β"][:, 1], bins=100, density=True)

predictions = Predictive(overfit_model, mcmc.get_samples())(new_X, None)
for k, v in predictions.items():
	print(k, v.shape, sep=": ")
