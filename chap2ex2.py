# -*- coding: utf-8 -*-

"""
In the interview process for each student, the student flips a coin, hidden from the interviewer.
The student agrees to answer honestly if the coin comes up heads.
Otherwise, if the coin comes up tails, the student (secretly) flips the coin again, and answers “Yes, I did cheat” if the coin flip lands heads, and “No, I did not cheat”, if the coin flip lands tails.
This way, the interviewer does not know if a “Yes” was the result of a guilty plea, or a Heads on a second coin toss.
Thus privacy is preserved and the researchers receive honest answers.

┬ cheat = no  ┬ 1st flip = tails ┬ 2nd flip = tails » answer = no
|             |                  └ 2nd flip = heads » answer = YES
|             └ 1st flip = heads                    » answer = no
└ cheat = yes ┬ 1st flip = tails ┬ 2nd flip = tails » answer = no
              |                  └ 2nd flip = heads » answer = YES
              └ 1st flip = heads                    » answer = YES
►►► prob_yes = .5 × prob_cheat + .5² (0.5 = prob flip coin)
"""

import torch, pyro
import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
pyro.enable_validation(True)
pyro.clear_param_store()

proba_coin = torch.tensor( .5)
N_tot      = torch.tensor(100)
N_yes      = torch.tensor( 35)

def cheating_model(N_tot, N_yes):
	proba_cheat = pyro.sample("𝒫 cheat", dist.Uniform(0., 1.))
	proba_yes = pyro.deterministic("𝒫 yes", proba_coin * proba_cheat + proba_coin**2)
	pyro.sample("𝒩 yes", dist.Binomial(N_tot, proba_yes), obs=N_yes)

# pyro.render_model(cheating_model, model_args=(N_tot, N_yes), render_distributions=True, render_params=True)

mcmc = MCMC(
	kernel=NUTS(cheating_model, jit_compile=True, ignore_jit_warnings=True),
	num_samples=4000,
	warmup_steps=1000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc.run(N_tot, N_yes)

mcmc.summary()
mcmc.diagnostics()

posterior_samples = {k: v.to("cpu") for k, v in mcmc.get_samples().items()}

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["𝒫 cheat"])
_ = ax[1].hist(posterior_samples["𝒫 cheat"], bins=100, density=True)
