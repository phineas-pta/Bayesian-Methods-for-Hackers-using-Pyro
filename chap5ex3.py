# -*- coding: utf-8 -*-

"""
src:
- http://www.kaggle.com/c/DarkWorlds
- http://www.timsalimans.com/observing-dark-worlds (dead link)
- https://web.archive.org/web/20190706180949/http://timsalimans.com/observing-dark-worlds/

The dataset is actually 300 separate files, each representing a sky.
In each file, or sky, are between 300 and 720 galaxies.
Each galaxy has an x and y position associated with it, ranging from 0 to 4200, and measures of ellipticity: e1 and e2

Each sky has 1, 2 or 3 dark matter halos in it.
prior distribution of halo positions: xᵢ ~ Unif(0, 4200) and yᵢ ~ Unif(0, 4200) for i in 1,2,3

most skies had one large halo and other halos, if present, were much smaller
mass large halo ~ Unif(40, 180) | N.B. log uniform (like in original salimans solution) make MCMC struggle to find initial values
"""

import json, torch, pyro, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, RandomWalkKernel

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
pyro.enable_validation(True)
pyro.clear_param_store()

with open("data/DarkWorld_train.json", "r", encoding="utf-8") as f:
	train_data_full = json.load(f)

def draw_sky(skyID: int):
	fig = plt.figure(figsize = (9, 9))
	ax = fig.add_subplot(111, aspect = "equal")
	data = train_data_full[skyID]

	for i in range(data["n_galaxies"]):
		x, y = data["position_galaxies"][i]
		e1, e2 = data["ellipticity_galaxies"][i]
		d = np.sqrt(e1**2 + e2**2)
		a, b = 1 / (1 - d), 1 / (1 + d)
		theta = np.degrees(.5*np.arctan2(e2, e1))
		ax.add_patch(Ellipse(xy = (x, y), width = 40*a, height = 40*b, angle = theta))

	for i in range(data["n_halos"]):
		x, y = data["halo_positions"][i]
		ax.scatter(x, y, c = "k", s = 70)

	ax.autoscale_view(tight = True)
	return fig
# _ = draw_sky("sky3")

XYmin = 0.
XYmax = 4200.
XYdims = 2

skyID = "sky215"
n_halos = train_data_full[skyID]["n_halos"]
n_galaxies = train_data_full[skyID]["n_galaxies"]
position_galaxies = torch.tensor(train_data_full[skyID]["position_galaxies"]) # shape: n_galaxies × XYdims
ellipticity_galaxies = torch.tensor(train_data_full[skyID]["ellipticity_galaxies"]) # shape: n_galaxies × XYdims

fdist_cste = torch.tensor([240., 70., 70.]) # 1st large halo and 2 small ones, values independent of any sky

###############################################################################
#%% naive brute force model
# quite slow so use random walk, unstable result

def f_distance(position_galaxy: torch.Tensor, position_halo: torch.Tensor, cste: torch.Tensor) -> torch.Tensor:
	euclidean_distance = torch.sqrt(torch.sum(torch.square(position_galaxy - position_halo)))
	return torch.max(euclidean_distance, cste)

def tangential_distance(position_galaxy: torch.Tensor, position_halo: torch.Tensor) -> torch.Tensor:
	delta = position_galaxy - position_halo
	phi = 2. * torch.atan2(delta[1], delta[0])
	return torch.tensor([-torch.cos(phi), -torch.sin(phi)])

def halos_model(n_halos: int, n_galaxies: int, position_galaxies: torch.Tensor, ellipticity_galaxies: torch.Tensor):
	mass_large_halo = pyro.sample("mass large halo", dist.Uniform(40., 180.))
	mass_halos = torch.tensor([mass_large_halo, 20., 20.])
	position_halos = pyro.sample("position all halos", dist.Uniform(XYmin, XYmax).expand([n_halos, XYdims]))
	# the loop is too complicated to use vmap
	for i in pyro.plate("galaxies loop", n_galaxies):
		position_galaxy = position_galaxies[i, :]
		tmp0 = torch.zeros(n_halos, XYdims)
		for j in pyro.plate(f"halos loop for galaxy {i}", n_halos):
			position_halo = position_halos[j, :]
			tmp1 = f_distance(position_galaxy, position_halo, fdist_cste[j])  # scalar
			tmp2 = tangential_distance(position_galaxy, position_halo)  # shape: XYdims
			tmp0[j, :] = mass_halos[j] / tmp1 * tmp2
		means = torch.sum(tmp0, dim=0)  # shape: XYdims
		pyro.sample(f"ellipticity galaxy {i}", dist.Normal(means, .05), obs=ellipticity_galaxies[i])

mcmc = MCMC(
	kernel=RandomWalkKernel(halos_model),
	num_samples=400,
	warmup_steps=100,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc.run(n_halos, n_galaxies, position_galaxies, ellipticity_galaxies)

mcmc.summary()
mcmc.diagnostics()

posterior_samples = {k: v.to("cpu") for k, v in mcmc.get_samples().items()}

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["position all halos"][:, 0, 0])  # x coordinate of 1st halo
_ = ax[1].hist(posterior_samples["position all halos"][:, 0, 0], bins=100, density=True)

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples["position all halos"][:, 0, 1])  # y coordinate of 1st halo
_ = ax[1].hist(posterior_samples["position all halos"][:, 0, 1], bins=100, density=True)

###############################################################################
#%% attempt to manipulate tensor instead of a loop

## for example in R can use `mapply` to apply f_distance/tangential_distance to each row [x,y] from position_galaxies and [x,y] from position_halos
## maybe possible with `numpy.vectorize` but no equivalence in pytorch
## luckily all operations in f_distance/tangential_distance are pytorch built-in element-wise operations
## so we can instead adjust tensor shape before applying these operations

# glxy_pos = position_galaxies[3:7]  # shape: 4×2
# halo_pos = torch.tensor([[1000., 500.], [2100., 1500.], [3500., 4000.]])  # shape: 3×2

# N_g = len(glxy_pos)  # 4
# N_h = len(halo_pos)  # 3

# halo_pos_expa = halo_pos.unsqueeze(1).repeat(  1, N_g, 1)  # shape: 3×2 → 3×4×2
# glxy_pos_expa = glxy_pos.unsqueeze(0).repeat(N_h,   1, 1)  # shape: 4×2 → 3×4×2
# diff_pos = halo_pos_expa - glxy_pos_expa  # shape: 3×4×2

# ecl_dist = torch.sqrt(torch.sum(torch.square(diff_pos), dim=-1))  # shape: 3×4×2 → 3×4

# fdist_cste_expa = fdist_cste.unsqueeze(1).repeat(1, N_g)  # shape: 3 → 3×4
# f_dist = torch.max(ecl_dist, fdist_cste_expa)  # shape: 3×4

# angle = 2. * torch.atan2(diff_pos[:,:,1], diff_pos[:,:,0])  # shape: 3×4×2 → 3×4
# t_dist = torch.stack([-torch.cos(angle), -torch.sin(angle)])  # shape: 3×4 → 2×3×4

# halo_mas = torch.tensor([500., 20., 20.])
# halo_mas_expa = halo_mas.unsqueeze(1).repeat(1, N_g)  # shape: 3 → 3×4

# means_estim = torch.sum(halo_mas_expa / f_dist * t_dist, dim=1).T  # shape: 2×3×4 → 3×4 → 4×3

###############################################################################
#%% model with tensor manipulation instead of loop
# faster, quite stable result but completely different

position_galaxies_expa = position_galaxies.unsqueeze(0).repeat(n_halos, 1, 1)  # shape: n_galaxies × XYdims → n_halos × n_galaxies × XYdims
fdist_cste_expa = fdist_cste[:n_halos].unsqueeze(1).repeat(1, n_galaxies)  # shape: n_halos → n_halos × n_galaxies

def halos_model_bis(n_halos: int, n_galaxies: int, position_galaxies_expa: torch.Tensor, ellipticity_galaxies: torch.Tensor):
	mass_large_halo = pyro.sample("mass large halo", dist.Uniform(40., 180.))
	position_halos = pyro.sample("position all halos", dist.Uniform(XYmin, XYmax).expand([n_halos, XYdims]))

	mass_halos = torch.tensor([mass_large_halo, 20., 20.][:n_halos])
	mass_halos_expa = mass_halos.unsqueeze(1).repeat(1, n_galaxies)  # shape: n_halos → n_halos × n_galaxies
	position_halos_expa = position_halos.unsqueeze(1).repeat(1, n_galaxies, 1)  # shape: n_halos × XYdims → n_halos × n_galaxies × XYdims

	delta_position = position_galaxies_expa - position_halos_expa  # shape: n_halos × n_galaxies × XYdims

	matrix_euclidean_distance = torch.sqrt(torch.sum(torch.square(delta_position), dim=-1))  # shape: n_halos × n_galaxies × XYdims → n_halos × n_galaxies
	matrix_f_distance = torch.max(matrix_euclidean_distance, fdist_cste_expa)  # shape: n_halos × n_galaxies

	phi_position = 2. * torch.atan2(delta_position[:,:,1], delta_position[:,:,0])  # shape: n_halos × n_galaxies × XYdims → n_halos × n_galaxies
	matrix_tangential_distance = torch.stack([-torch.cos(phi_position), -torch.sin(phi_position)])  # shape: n_halos × n_galaxies → XYdims × n_halos × n_galaxies

	mean_ellipticity = torch.sum(mass_halos_expa / matrix_f_distance * matrix_tangential_distance, dim=1).T  # shape: XYdims × n_halos × n_galaxies → XYdims × n_galaxies → n_galaxies × XYdims
	pyro.sample("ellipticity galaxy", dist.Normal(mean_ellipticity, .05), obs=ellipticity_galaxies)  # shape: n_galaxies × XYdims

mcmc_bis = MCMC(
	kernel=NUTS(halos_model_bis, jit_compile=True, ignore_jit_warnings=True),
	num_samples=4000,
	warmup_steps=1000,
	# num_chains=4  # ERROR when >1 on windows+cuda
)
mcmc_bis.run(n_halos, n_galaxies, position_galaxies_expa, ellipticity_galaxies)

mcmc_bis.summary()
mcmc_bis.diagnostics()

posterior_samples_bis = {k: v.to("cpu") for k, v in mcmc_bis.get_samples().items()}

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples_bis["position all halos"][:, 0, 0])  # x coordinate of 1st halo
_ = ax[1].hist(posterior_samples_bis["position all halos"][:, 0, 0], bins=100, density=True)

fig, ax = plt.subplots(1, 2)
ax[0].plot(posterior_samples_bis["position all halos"][:, 0, 1])  # y coordinate of 1st halo
_ = ax[1].hist(posterior_samples_bis["position all halos"][:, 0, 1], bins=100, density=True)

fig = draw_sky("sky215")
plt.scatter(posterior_samples_bis["position all halos"][:, :, 0], posterior_samples_bis["position all halos"][:, :, 1], alpha = 0.015, c = "r")
