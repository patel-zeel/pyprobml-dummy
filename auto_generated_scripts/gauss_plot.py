#!/usr/bin/env python
# coding: utf-8

# # CDF and PDF plots of Standard Normal Distribution

# In[1]:


import os

try:
    import jax
except:
    get_ipython().run_line_magic('pip', 'install jax jaxlib')
    import jax

import jax.numpy as jnp
from jax.scipy.stats import norm

try:
    import matplotlib.pyplot as plt
except:
    get_ipython().run_line_magic('pip', 'install matplotlib')
    import matplotlib.pyplot as plt

try:
    import seaborn as sns
except:
    get_ipython().run_line_magic('pip', 'install seaborn')
    import seaborn as sns


# In[2]:


dev_mode = "DEV_MODE" in os.environ

if dev_mode:
    import sys

    sys.path.append("scripts")
    import pyprobml_utils as pml
    from latexify import latexify

    latexify(scale_factor=2, fig_height=1.5)


# In[4]:


# Plot pdf and cdf of standard normal

x = jnp.linspace(-3, 3, 500)
random_var = norm
fig, ax = plt.subplots()
ax.plot(x, random_var.pdf(x))
plt.title("Gaussian pdf")
sns.despine()

if dev_mode:
    pml.save_fig("gaussian1d.pdf")

# plt.show()

fig, ax = plt.subplots()
ax.plot(x, random_var.cdf(x))
plt.title("Gaussian cdf")
sns.despine()
if dev_mode:
    pml.save_fig("gaussianCdf.pdf")

# plt.show()


# In[ ]:




