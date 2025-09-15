# The purpose of this file is to practice probabilistic programming with PyMC.

import numpy as np
#import pandas as pd
import pytensor as pt
import pymc
import arviz
import altair as alt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
#iris = load_iris(as_frame=True)

SEED = 42  # For reproducibility

# Data info: https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
diabetes = load_diabetes(as_frame=True)
df = diabetes.data

"""" Diabetes dataset: mean-centered at 0, scaled to sum-of-squares
age = age in years
sex
bmi = body mass index
bp = average blood pressure
s1 = tc, total serum cholesterol
s2 = ldl, low-density lipoproteins
s3 = hdl, high-density lipoproteins
s4 = tch, total cholesterol / HDL
s5 = ltg, possibly log of serum triglycerides level
s6 = glu, blood sugar level
target = a quantitative measure of disease progression one year after baseline
"""


X_train, X_test, y_train, y_test = train_test_split(df, diabetes.target, test_size=0.2, random_state=SEED)

df_train = X_train.copy()
df_train["target"] = y_train

# Plot a bunch of possible relationships between features and target.
plt = alt.Chart(df_train).mark_point().encode(
    x=alt.X("bmi:Q", title="Body Mass Index (BMI)")
    , y=alt.Y("target:Q", title="Disease Progression")
    , color=alt.Color("age:Q", title="Age", scale=alt.Scale(scheme="blues"))
    , shape=alt.Shape("sex:N", title="Sex", scale=alt.Scale(range=["circle", "square"]))
    , tooltip=["target", "bmi", "age", "sex", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
).properties(width=800, height=400).interactive()
plt

# What about age vs target?
alt.Chart(df_train).mark_point().encode(
    x=alt.X("age:Q", title="Age")
    , y=alt.Y("target:Q", title="Disease Progression")
    , color=alt.Color("bmi:Q", title="Body Mass Index (BMI)", scale=alt.Scale(scheme="greens"))
)

# target distribution
alt.Chart(df_train).mark_bar().encode(
    x=alt.X("target:Q", title="Disease Progression")
    , y=alt.Y("count()", title='count')
)


# Older ages and higher BMI are associated with higher disease progression, so let's regress on those.
# Our model:
# target = slope1*age + slope2*bmi + intercept + error
coords = {'coeffs':X_train.columns
          , "ydim":range(y_train.size)}
xvars = ["age", "bmi"]  # independent variables

with pymc.Model(coords=coords) as model:
    # data
    X = pymc.MutableData("X", X_train[xvars])  # Note: I should remove unused columns from X_train before this step.
    y = pymc.MutableData("y", y_train)

    # priors
    intercept = pymc.Normal("intercept", mu=0, sigma=50)
    age = pymc.Normal("age", mu=0, sigma=50)
    bmi = pymc.Uniform("bmi", lower=-10, upper=300)
    error = pymc.Normal("error", mu=0, sigma=80)  # sigma

    # expected value, predicted y
    mu = pymc.Deterministic("mu", age*X_train["age"] + bmi*X_train["bmi"] + intercept)

    # likelihood
    #y_pred = pymc.Normal("y_pred", mu=mu, sigma=error, shape=X.shape, observed=y_train)  # need data to match variables (no extra columns)
    y_pred = pymc.Normal("y_pred", mu=mu, sigma=error, observed=y_train)

    # infer the posterior distribution, conditional on the observed data
    trace = pymc.sample(draws=1000, tune=1000, return_inferencedata=True, random_seed=SEED)

# Plot the posterior distributions of the parameters
pymc.model_to_graphviz(model)
arviz.plot_posterior(trace, var_names=["intercept", "age", "bmi", "error"])
arviz.plot_trace(trace, var_names=["intercept", "age", "bmi", "error"])
arviz.plot_autocorr(trace, var_names=["intercept", "age", "bmi", "error"])
arviz.summary(trace, round_to=2)
arviz.plot_forest(trace, var_names=["intercept", "age", "bmi"], combined=True, hdi_prob=0.95, r_hat=True)

arviz.plot_trace(trace, combined=True)


# predict on training data (in-sample predictions)
with model:
    pymc.set_data({"X": X_train}) # requires pymc.Data() initially
    ytrain_pred = pymc.sample_posterior_predictive(trace=trace, random_seed=SEED)

# predict on new data
with model:
    pymc.set_data({"X": X_test, "y": np.zeros(len(X_test))}) # requires pymc.Data() initially
    model.set_dim("ydim", y_test.size, coord_values=range(y_test.size))
    ytest_pred = pymc.sample_posterior_predictive(trace=trace, predictions=True, random_seed=SEED)

# model training data predictions
ytrain_model = ytrain_pred.predictions["y_pred"].mean(dim=["chain", "draw"])

# model test data predictions
ytest_model = trace.posterior_predictive["y_pred"].mean(dim=["chain", "draw"])

len(trace.posterior['y_pred'].mean(dim=["chain", "draw"]).values)
y_train.shape

ytrain_pred_val = ytrain_pred.posterior_predictive['y_pred'].mean(dim=["chain", "draw"]).values
ytest_pred_val =  ytest_pred.predictions['y_pred'].mean(dim=["chain", "draw"]).values
ytest_pred_val.shape

len(X_test)
len(ytest_pred_val)

# data = array (4, 1000, 353)
tmp = np.concatenate(np.concatenate(trace.posterior['mu'].data[:,:,:]))

pd.DataFrame({'mu':tmp}).hist()
alt.Chart(pd.DataFrame({'mu':tmp})).mark_bar().encode(
    x=alt.X("mu:Q", title="Expected Value of Mu"),
    y=alt.Y("count()", title="Count"),
).properties(width=800, height=400)







# Effectiveness of ad campaign
from scipy import stats

# 1.5% click-through rate + some noise so we don't know the true value.
# This is unknown, but we use it to simulate data.
np.random.seed(SEED)
p_true = 0.015  + np.random.uniform(-0.01, 0.01, size=1)[0]
# I get bad results with the above probability, so try a higher one
#p_true = 0.1  

N = 10000  # Number of impressions

# simulate data
np.random.seed(SEED)
clicks = stats.bernoulli.rvs(p=p_true, size=int(N), random_state=SEED)

print(f"Clicks / Impressions: {clicks.sum()}/{int(N)} = {clicks.sum()/N:.2%}")

# Model the click-through rate
with pymc.Model() as ctr_model:

    # we think people are unlikely to click, so the prior is weighted closer to 0
    # Beta distribution is between [0,1]
    p_click = pymc.Beta("p_click", alpha=1, beta=100)     # bad results
    #p_click = pymc.Uniform("p_click", lower=0, upper=.1) # bad results
    #p_click = pymc.Uniform("p_click")


    # Likelihood: observed clicks.
    clicks_obs = pymc.Bernoulli("clicks_obs", p=p_click, observed=clicks)

    # Sample from the posterior distribution
    #step = pymc.Metropolis()
    step = pymc.NUTS()  
    trace = pymc.sample(step=step, draws=16000, tune=4000, random_seed=SEED, chains=2)


arviz.plot_posterior(trace, var_names=["p_click"])
np.concatenate(trace.posterior["p_click"][:,:]).shape # shape [2000]
#.mean(dim=["chain", "draw"]).values


# Compare another ad campaign
# This follows the Ch2 example
p_true1 = 0.015  + np.random.uniform(-0.01, 0.01, size=1)[0]
N1 = 4000 # Number of impressions for the second campaign
clicks1 = stats.bernoulli.rvs(p=p_true1, size=N1, random_state=SEED)

with pymc.Model() as ctr_model2:

    # we think people are unlikely to click, so the prior is weighted closer to 0
    # Beta distribution is between [0,1]
    p_click = pymc.Beta("p_click", alpha=1, beta=3)
    p_click1 = pymc.Beta("p_click1", alpha=1, beta=3)

    delta = pm.Deterministic("delta", p_click - p_click1)

    # Likelihood: observed clicks. clicks=A, clicks1=B
    clicks_obsA = pymc.Bernoulli("clicks_obsA", p=p_click, observed=clicks)
    clicks_obsB = pymc.Bernoulli("clicks_obsB", p=p_click1, observed=clicks1)

    # Sample from the posterior distribution
    #step = pymc.Metropolis()
    step = pymc.NUTS()  
    trace = pymc.sample(step=step, draws=16000, tune=4000, random_seed=SEED, chains=2)

p_A_samples =  np.concatenate(trace.posterior.p_click.data[:,4000:])
p_B_samples =  np.concatenate(trace.posterior.p_click1.data[:,4000:])
delta_samples = np.concatenate(trace.posterior.delta.data[:,4000:])
#histogram of posteriors

ax = plt.subplot(311)

plt.xlim(0, .1)
plt.hist(p_A_samples, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_A$", color="#A60628", density=True)
#plt.vlines(p_click, 0, 80, linestyle="--", label="true $p_A$ (unknown)")
plt.legend(loc="upper right")
plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")

ax = plt.subplot(312)

plt.xlim(0, .1)
plt.hist(p_B_samples, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_B$", color="#467821", density=True)
#plt.vlines(p_click1, 0, 80, linestyle="--", label="true $p_B$ (unknown)")
plt.legend(loc="upper right")

ax = plt.subplot(313)
plt.hist(delta_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of delta", color="#7A68A6", density=True)
#plt.vlines(p_click - p_click1, 0, 60, linestyle="--",
#           label="true delta (unknown)")
plt.vlines(0, 0, 60, color="black", alpha=0.2)
plt.legend(loc="upper right");

print(f"pA: {round(p_true,3)}, pB: {round(p_true1,3)}, delta: {round(p_true - p_true1,3)}")
print(f"posteriorA: {p_A_samples.mean():.3f}, posteriorB: {p_B_samples.mean():.3f}, delta: {delta_samples.mean():.3f}")



# Recreate sample from book:
#set constants
p_true = 0.05  # remember, this is unknown.
N = 1500

# sample N Bernoulli random variables from Ber(0.05).
# each random variable has a 0.05 chance of being a 1.
# this is the data-generation step
occurrences = stats.bernoulli.rvs(p_true, size=N)

print(occurrences) # Remember: Python treats True == 1, and False == 0
print(np.sum(occurrences))
print("What is the observed frequency in Group A? %.4f" % np.mean(occurrences))

with pm.Model() as model:
    p = pm.Uniform('p')#, lower=0, upper=1)
with model:
    observed = pm.Bernoulli("obs", p, observed=occurrences)
#include the observations, which are Bernoulli
with model:
    # To be explained in chapter 3
    step = pm.Metropolis()
    trace = pm.sample(18000,step=step,chains=3) #default value of chains is 2, runs independent chains
    # We have a new data structure to burn in pymc current
    # if you use return_inferencedata=False, the code below will still work, but for little ArviZ, let's use the default True value.
    # burned_trace = trace[1000:] 
trace.posterior.p.data[:,1000:].mean()

plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)",color='black')
combine_3_chains = np.concatenate(trace.posterior.p.data[:,1000:])
plt.hist( combine_3_chains, bins=25, histtype="stepfilled", density=True)
plt.legend()
arviz.plot_posterior(trace, var_names=["p"])