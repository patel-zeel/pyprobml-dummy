#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Anscombe's quartet
# Author: Drishtii
try:
    import seaborn as sns
except:
    get_ipython().run_line_magic('pip', 'install seaborn')
    import seaborn as sns

try:
    import matplotlib.pyplot as plt
except:
    get_ipython().run_line_magic('pip', 'install matplotlib')
    import matplotlib.pyplot as plt

try:
    from sklearn.linear_model import LinearRegression
except:
    get_ipython().run_line_magic('pip', 'install scikit-learn')
    from sklearn.linear_model import LinearRegression
try:
    import jax
except:
    get_ipython().run_line_magic('pip', 'install jax jaxlib')
    import jax
import jax.numpy as jnp


# In[2]:


import os

dev_mode = "DEV_MODE" in os.environ

if dev_mode:
    import sys

    sys.path.append("scripts")
    import pyprobml_utils as pml
    from latexify import latexify


# In[3]:


SCATTER_SIZE = 20 if dev_mode else 20
FIG_SIZE = (12, 3)
df = sns.load_dataset("anscombe")


# In[4]:


def make_graph(ax, data, color=None):
    x = data["x"]
    y = data["y"]
    dataset_no = data["dataset_no"]

    model = LinearRegression().fit(x, y)
    x_range = jnp.linspace(1, 20, num=20).reshape(-1, 1)
    y_pred = model.predict(x_range)

    ax.plot(x_range, y_pred, color=color)
    ax.scatter(x, y, s=SCATTER_SIZE, color=color)

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.set_title(f"Dataset = {dataset_no}")

    ax.set_xlabel("$x$")


# In[5]:


if dev_mode:
    latexify(width_scale_factor=1, fig_height=1.5)
names = df["dataset"].unique()
fig, axes = plt.subplots(1, 4, sharey=True)
colors = ["blue", "orange", "green", "red"]
for i in range(4):
    data_df = df[df["dataset"] == names[i]]
    data_df = data_df.sort_values(by="x")
    x = data_df["x"].values.reshape(-1, 1)
    y = data_df["y"].values.reshape(-1, 1)

    data = {"x": x, "y": y, "dataset_no": names[i]}
    ax = axes[i]
    make_graph(ax, data, color=colors[i])
axes[0].set_ylabel("$y$")
sns.despine()
if dev_mode:
    pml.savefig(f"anscombes_quartet_latexified.pdf")


# In[6]:


for i, name in enumerate(names):
    plt.figure()
    print(name)
    name_index = df["dataset"] == name
    data_df = df[name_index]
    data_df = data_df.sort_values(by="x")
    x = data_df["x"].values.reshape(-1, 1)
    y = data_df["y"].values.reshape(-1, 1)

    data = {"x": x, "y": y, "dataset_no": names[i]}
    ax = plt.gca()
    make_graph(ax, data, colors[i])

    mean_x = data_df["x"].to_numpy().mean()
    mean_y = data_df["y"].to_numpy().mean()
    ax.set_title(f"{name}, mean_x={mean_x:0.3f}, mean_y={mean_y:0.3f}")
    print(data_df[["x", "y"]].agg(["count", "mean", "var"]))
    if dev_mode:
        pml.savefig(f"anscombes_quartet_{name}_latexified.pdf")

    sns.despine();


# In[7]:


# Compare the two different estimators for the variance
# https://github.com/probml/pml-book/issues/264
for d in ["I", "II", "III", "IV"]:
    print("dataset ", d)

    x = df[df["dataset"] == d]["x"].to_numpy()
    print("var x, MLE = {:.2f}".format(((x - x.mean()) ** 2).mean()))
    print("var x, numpy: {:.2f}".format(x.var()))
    print("var x, unbiased estimator: {:.2f}\n".format(x.var(ddof=1)))

    y = df[df["dataset"] == d]["y"].to_numpy()
    print("var y, MLE = {:.2f}".format(((y - y.mean()) ** 2).mean()))
    print("var y, numpy: {:.2f}".format(y.var()))
    print("var y, unbiased estimator: {:.2f}\n".format(y.var(ddof=1)))

