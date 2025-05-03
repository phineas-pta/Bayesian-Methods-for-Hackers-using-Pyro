# -*- coding: utf-8 -*-

"""
src:
- https://arxiv.org/pdf/1008.4686
- https://github.com/astroML/astroML/blob/main/astroML/datasets/hogg2010test.py
- https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-robust-with-outlier-detection.html
"""

import torch, pyro
import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

N = 20
x       = torch.tensor([201., 244.,  47., 287., 203.,  58., 210., 202., 198., 158., 165., 201., 157., 131., 166., 160., 186., 125., 218., 146.])
y       = torch.tensor([592., 401., 583., 402., 495., 173., 479., 504., 510., 416., 393., 442., 317., 311., 400., 337., 423., 334., 533., 344.])
sigma_x = torch.tensor([  9.,   4.,  11.,   7.,   5.,   9.,   4.,   4.,  11.,   7.,   5.,   5.,   5.,   6.,   6.,   5.,   9.,   8.,   6.,   5.])
sigma_y = torch.tensor([ 61.,  25.,  38.,  15.,  21.,  15.,  27.,  14.,  30.,  16.,  14.,  25.,  52.,  16.,  34.,  31.,  42.,  26.,  16.,  22.])
rho_xy  = torch.tensor([-.84,  .31,  .64, -.27, -.33,  .67, -.02, -.05, -.84, -.69,   .3, -.46, -.03,   .5,  .73, -.52,   .9,   .4, -.78, -.56])

# plt.errorbar(x.to("cpu"), y.to("cpu"), xerr=sigma_x.to("cpu"), yerr=sigma_y.to("cpu"), fmt="none")

# from matplotlib.patches import Ellipse
# plt.scatter(x.to("cpu"), y.to("cpu"))
# for i in range(N):
# 	cov_i = rho_xy[i] * sigma_x[i] * sigma_y[i]
# 	cov_mat_i = torch.tensor([[sigma_x[i]**2, cov_i], [cov_i, sigma_y[i]**2]])
# 	_, evecs_i = torch.linalg.eig(cov_mat_i) # eigenvalue, eigenvector
# 	angle_i = torch.rad2deg(torch.atan2(evecs_i[1][0].real, evecs_i[0][0].real))
# 	ellipse_i = Ellipse((x[i], y[i]), width=2*sigma_x[i], height=2*sigma_y[i], angle=angle_i, edgecolor="b", facecolor="none")
# 	plt.gca().add_patch(ellipse_i)

###############################################################################
#%% Linear Model with Custom Likelihood to Distinguish Outliers: Hogg Method
# idea: mixture model whereby datapoints can be: normal linear model vs outlier (for convenience also be linear)

def hogg_model(x, y, sigma_y):
	b0 = pyro.sample("intercept", dist.Normal(0., 5.))  # weakly informative Normal priors (L2 ridge reg) for inliers
	b1 = pyro.sample(    "slope", dist.Normal(0., 5.))
	y_outlier = pyro.sample("mean for all outliers", dist.Normal(0., 10.))
	sigma_y_outlier = pyro.sample("additional variance for outliers", dist.InverseGamma(.001, .001))
	weight_inlier = pyro.sample("𝒫 inlier", dist.Uniform(0., 1.))
	with pyro.plate("data loop", N):
		log_proba_inlier  = torch.log(     weight_inlier) + dist.Normal(b0 + b1 * x, sigma_y                  ).log_prob(y)
		log_proba_outlier = torch.log(1. - weight_inlier) + dist.Normal(y_outlier,   sigma_y + sigma_y_outlier).log_prob(y)
		pyro.factor("obs", torch.logaddexp(log_proba_inlier, log_proba_outlier))

# pyro.render_model(hogg_model, model_args=(x, y, sigma_y), render_distributions=True, render_params=True)

mcmc_hogg = MCMC(
	kernel=NUTS(hogg_model, jit_compile=True, ignore_jit_warnings=True),
	num_samples=4000,
	warmup_steps=1000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc_hogg.run(x, y, sigma_y)

mcmc_hogg.summary()
mcmc_hogg.diagnostics()

#%% declare outliers
# need to work out in case multiple chains

post_samples = mcmc_hogg.get_samples()

# Compute the un-normalized log probabilities for each cluster
inliers_log_prob = dist.Normal(
	post_samples["intercept"] + post_samples["slope"] * x.unsqueeze(1),
	sigma_y.unsqueeze(1)
).log_prob(y.unsqueeze(1)) + torch.log(post_samples["𝒫 inlier"])  # shape: N × N_samples
outliers_log_prob = dist.Normal(
	post_samples["mean for all outliers"],
	sigma_y.unsqueeze(1) + post_samples["additional variance for outliers"]
).log_prob(y.unsqueeze(1)) + torch.log(1. - post_samples["𝒫 inlier"])  # shape: N × N_samples

# Bayes rule to compute the assignment probability: P(cluster = 1 | data) ∝ P(data | cluster = 1) P(cluster = 1)
log_p_assign_outliers = outliers_log_prob - torch.logaddexp(inliers_log_prob, outliers_log_prob)  # shape: N × N_samples

# Average across the MCMC chain
log_p_assign_outliers_bis = torch.logsumexp(input=log_p_assign_outliers, dim=-1) - torch.log(torch.tensor(log_p_assign_outliers.shape[-1]))  # shape: N

proba_outlier = torch.exp(log_p_assign_outliers_bis)

# remove outliers
idx_inliers = proba_outlier <= .95
N_inliers = torch.sum(idx_inliers).item()
x_inliers       =       x[idx_inliers]
y_inliers       =       y[idx_inliers]
sigma_x_inliers = sigma_x[idx_inliers]
sigma_y_inliers = sigma_y[idx_inliers]
rho_xy_inliers  =  rho_xy[idx_inliers]

###############################################################################
#%% full model with all variances

# N_inliers = 17
# x_inliers       = torch.tensor([201., 203.,  58., 210., 202., 198., 158., 165., 201., 157., 131., 166., 160., 186., 125., 218., 146.])
# y_inliers       = torch.tensor([592., 495., 173., 479., 504., 510., 416., 393., 442., 317., 311., 400., 337., 423., 334., 533., 344.])
# sigma_x_inliers = torch.tensor([  9.,   5.,   9.,   4.,   4.,  11.,   7.,   5.,   5.,   5.,   6.,   6.,   5.,   9.,   8.,   6.,   5.])
# sigma_y_inliers = torch.tensor([ 61.,  21.,  15.,  27.,  14.,  30.,  16.,  14.,  25.,  52.,  16.,  34.,  31.,  42.,  26.,  16.,  22.])
# rho_xy_inliers  = torch.tensor([-.84, -.33,  .67, -.02, -.05, -.84, -.69,   .3, -.46, -.03,   .5,  .73, -.52,   .9,   .4, -.78, -.56])

def full_model(x, y, sigma_x, sigma_y, rho_xy):
	z = torch.stack([x, y], dim=1)
	cov_xy = sigma_x * sigma_y * rho_xy
	s = torch.stack([  # covariance matrix
		torch.stack([sigma_x**2,     cov_xy], dim=1),
		torch.stack([    cov_xy, sigma_y**2], dim=1)
	], dim=2)

	m = pyro.sample("slope", dist.Normal(0., 1.))
	b = pyro.sample("intercept", dist.Normal(0., 1.))
	with pyro.plate("data loop", N_inliers):
		y_hat = b + m * x
		z_hat = torch.stack([x, y_hat], dim=1)
		pyro.sample("obs", dist.MultivariateNormal(z_hat, s), obs=z)

# pyro.render_model(full_model, model_args=(x_inliers, y_inliers, sigma_x_inliers, sigma_y_inliers, rho_xy_inliers), render_distributions=True, render_params=True)

mcmc_full = MCMC(
	kernel=NUTS(full_model, jit_compile=True, ignore_jit_warnings=True),
	num_samples=4000,
	warmup_steps=1000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc_full.run(x_inliers, y_inliers, sigma_x_inliers, sigma_y_inliers, rho_xy_inliers)

mcmc_full.summary()
mcmc_full.diagnostics()

#%% full model with intrinsic scatter

angle90 = torch.pi / 2

def full_model_bis(x, y, sigma_x, sigma_y, rho_xy):
	z = torch.stack([x, y], dim=1)
	cov_xy = sigma_x * sigma_y * rho_xy
	s = torch.stack([  # covariance matrix
		torch.stack([sigma_x**2,     cov_xy], dim=1),
		torch.stack([    cov_xy, sigma_y**2], dim=1)
	], dim=2)

	theta = pyro.sample("θ", dist.Uniform(-angle90, angle90))  # angle of the fitted line, use this instead of slope
	v = torch.tensor([[-torch.sin(theta), torch.cos(theta)]]).T  # unit vector orthogonal to the line
	b = pyro.sample("b", dist.Normal(0., 1.))  # intercept
	V = pyro.sample("V", dist.InverseGamma(.001, .001))  # intrinsic Gaussian variance orthogonal to the line

	with pyro.plate("data loop", N_inliers):
		delta = z @ v - b*v[1]  # orthogonal displacement of each data point from the line
		sigma2 = v.T @ s @ v  # orthogonal variance of projection of each data point to the line
		tmp = sigma2 + V  # intermediary result
		lp = .5*(torch.log(tmp) + delta**2 / tmp)  # log prob
		pyro.factor("obs", -torch.sum(lp))  # ATTENTION sign

mcmc_full_bis = MCMC(
	kernel=NUTS(full_model_bis, jit_compile=True, ignore_jit_warnings=True),
	num_samples=4000,
	warmup_steps=1000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc_full_bis.run(x_inliers, y_inliers, sigma_x_inliers, sigma_y_inliers, rho_xy_inliers)

mcmc_full_bis.summary()
mcmc_full_bis.diagnostics()
