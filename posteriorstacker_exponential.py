#!/usr/bin/env python
"""
Posterior stacking tool.

Johannes Buchner, Matthew Pitkin (C) 2020-2021

Given posterior distributions of some parameter from many objects,
computes the sample distribution, using a simple hierarchical model.

The method is described in Baronchelli, Nandra & Buchner (2020)
https://ui.adsabs.harvard.edu/abs/2020MNRAS.498.5284B/abstract
One computation is performed with this tool:

- Exponential model (as in the paper)

The model is computed using UltraNest.
The output is plotted.
"""

import numpy as np
import matplotlib.pyplot as plt
import ultranest, ultranest.stepsampler
from scipy.stats import expon, truncnorm


print("fitting exponential model...")

eparam_names = ['mean']

def exponential_pdf(x, mean):
    """Same as scipy.stats.expon(scale=mean).pdf(x), but faster."""
    return (1.0 / mean) * np.exp(-x / mean)

def elikelihood(params):
    """Exponential sample distribution"""
    mean = params
    return np.log(exponential_pdf(data, mean).mean(axis=1) + 1e-300).sum()

def etransform(cube):
    """Exponential sample distribution priors"""
    params = cube.copy()
    params[0] = 3 * (maxval - minval) * cube[0] + minval
    return params


# run 100 times to generate data for a P-P plot

# draw values from exponential distribution with a given mean
mean = 100.0
err = 0.25  # maximum error is 25% of value

minval = 0.0
maxval = 50 * mean

credible_levels = []  # store credibel level that contains the true value

for j in range(100):
    # create set of values drawn from exponential
    values = expon.rvs(scale=mean, size=250)
    samples = []

    # create posterior samples - use truncated normal so values cannot be negative

    for i in range(len(values)):
        samples.append(
            truncnorm.rvs(
                0.0,
                np.inf,
                loc=values[i],
                scale=(0.25 * values[i] * np.random.rand()),
                size=400,
            )
        )

    data = np.array(samples)

    esampler = ultranest.ReactiveNestedSampler(
        eparam_names, elikelihood, etransform,
        log_dir='test_out_exp_{}'.format(j), resume=True)
    eresult = esampler.run(frac_remain=0.5, viz_callback=None)

    # get credible level of true mean
    mask = np.cumsum(np.array(eresult['weighted_samples']['weights'])) > 1e-4
    psamples = np.array(eresult['weighted_samples']['points'])[mask, :]
    weights = np.array(eresult['weighted_samples']['weights'])[mask]
    clevel = sum(np.array(psamples[:,0] < mean) * weights / sum(weights))
    
    credible_levels.append(clevel)

np.savetxt("credible_levels.txt", credible_levels)

x_values = np.linspace(0, 1, 1001)
pp = np.array([sum(np.array(credible_levels) < xx) / len(credible_levels) for xx in x_values])
fig, ax = plt.subplots()
ax.plot(x_values, pp)
ax.plot([0, 1], [0, 1], "k--")

ax.set_xlabel("C.I.")
ax.set_ylabel("Fraction of events in C.I.")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

fig.tight_layout()
fig.savefig("ppplot.png", dpi=200)

