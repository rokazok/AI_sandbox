# Plots
This folder includes sample plots, focusing on [altair](https://altair-viz.github.io/#) 
because it's the closest Python equivalent I've found to R's wonderful 
[ggplot2](https://ggplot2.tidyverse.org/) library.

The economic data plot in [gapminder_economic_data.py](gapminder_economic_data.py) 
recreates the Gapminder [bubble plot](https://www.gapminder.org/tools/#$chart-type=bubbles&url=v2) 

Features:
- Life expectancy vs per-capita GDP with population size mapped to size and region to color.
- Year maps to a slider bar that lets users move forward and backward in time.
- Interactive multi-click selection lets users annotate and follow countries.

Gapminder has an autoplay feature, but moving forwards/backwards in time is stuttery. 
I was unable to allow users to toggle the x-axis scale between linear (to show actual numbers)
 and log scales (to better separate the densely-clustered countries at lower GDPs).