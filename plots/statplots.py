# Purpose: Recreate statistical plots using Altair
# from https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers


import altair as alt
import scipy.stats as stats
import numpy as np
import pandas as pd



# Ch 1
# poisson distribution
poi = stats.poisson
x = np.arange(16)
lambda_ = [1.5, 4.25]
colours = ["#348ABD", "#A60628"]

df = pd.DataFrame({
    "x": np.tile(x, 2),
    "pmf": np.concatenate([poi.pmf(x, mu=l) for l in lambda_]),
    #"color": np.repeat(colours, len(x)),
    "lambda": np.repeat(lambda_, len(x))
})

# Plot probability mass function for discrete variables
plt = alt.Chart(df).mark_bar().encode(
    x=alt.X("x:O", title="k (Number of Events)").axis(labelAngle=0),  # Use ordinal for x to enable xOffset
    xOffset="lambda:N",
    y=alt.Y("pmf:Q", title="Probability Mass Function"),
    color=alt.Color("lambda:N", title="Lambda", scale=alt.Scale(domain=lambda_, range=colours))
)
plt.show()


# Plot probability density function for continuous variables
x = np.linspace(0, 4, 100)
expo = stats.expon
lambda_ = [0.5, 1]

df = pd.DataFrame({
    "x": np.tile(x, 2),
    "pdf": np.concatenate([expo.pdf(x, scale=1./l) for l in lambda_]),
    "color": np.repeat(colours, len(x)),
    "lambda": np.repeat(lambda_, len(x))
})

# Plot probability density function for continuous variables
# plt = alt.Chart(df.loc[df["lambda"] == 0.5,]).mark_area().encode(
plt = alt.Chart(df).mark_area().encode(
    x=alt.X("x:Q", title="x"),
    y=alt.Y("pdf:Q", title="Probability Density Function"),
    opacity=alt.value(0.5),
    color=alt.Color("lambda:N", title="$\lambda", scale=alt.Scale(domain=lambda_, range=colours))
)
plt.show()

expo.pdf(x, scale=1./0.5)

df.query("lambda == 0.5").head()
df.loc[df["lambda"] == 0.5, ]

df.loc[df["pdf"] > 1, ].head()
df.loc[df["lamnd"] > 1, ].head()

plt = alt.Chart(df).mark_line().encode(
    x=alt.X("x:Q", title="x"),
    y=alt.Y("pdf:Q", title="Probability Density Function"),
    opacity=alt.value(0.5),
    color=alt.Color("lambda:N", title="$\lambda", scale=alt.Scale(domain=lambda_, range=colours))
)
plt.show()



import pymc

with pymc.Model() as model:
    # Define a prior distribution for the parameter lambda
    scaling_factor = 1/my_observed_count_data.mean()  # Example scaling factor based on observed data
    # alternatively, you could use a prior distribution
    scaling_factor = pm.DiscreteUniform("scaling_factor", 0, 50) # value between 0 and 50


    lambda_ = pymc.Exponential("lambda", scaling_factor)
    
    # Define the likelihood function
    obs = pymc.Poisson("obs", mu=lambda_, observed=my_observed_count_data)
    
    # Sample from the posterior distribution
    trace = pymc.sample(1000, return_inferencedata=False)

# Look at your posterior distribution, which helps answer the question "What is the most likely value of lambda?"
lambda = trace['lambda']


tmp = np.random.poisson(lam=4.5, size=(10,4))
tmp.mean(axis=0)
