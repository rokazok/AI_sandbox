# An Altair chart with a time slider, using real Gapminder datasets.
# See Gapminder visualizations: https://www.gapminder.org/tools/#$chart-type=bubbles&url=v2
# Selection widget examples from altair: https://altair-viz.github.io/gallery/multiple_interactions.html

import altair as alt
import pandas as pd
import numpy as np

# 1. Load the datasets from the provided URLs.
# We'll use pandas to read the CSV files directly from the web.
try:
    gdp_data = pd.read_csv('https://raw.githubusercontent.com/open-numbers/ddf--gapminder--systema_globalis/master/countries-etc-datapoints/ddf--datapoints--gdppercapita_us_inflation_adjusted--by--geo--time.csv')
    life_expectancy_data = pd.read_csv('https://raw.githubusercontent.com/open-numbers/ddf--gapminder--systema_globalis/master/countries-etc-datapoints/ddf--datapoints--life_expectancy_at_birth_data_from_ihme--by--geo--time.csv')
    population_data = pd.read_csv('https://raw.githubusercontent.com/open-numbers/ddf--gapminder--systema_globalis/master/countries-etc-datapoints/ddf--datapoints--total_population_with_projections--by--geo--time.csv')
    region_data = pd.read_csv('https://raw.githubusercontent.com/open-numbers/ddf--gapminder--systema_globalis/master/ddf--entities--geo--country.csv')
except Exception as e:
    print(f"Error loading data from URLs: {e}")
    # Create a simplified dummy dataframe if the URLs fail, to allow the code to still run
    print("Using dummy data instead.")
    years = np.arange(1950, 2011)
    geos = ['usa', 'chn', 'ind', 'deu']
    countries = {'usa': 'United States', 'chn': 'China', 'ind': 'India', 'deu': 'Germany'}
    regions = {'usa': 'America', 'chn': 'Asia', 'ind': 'Asia', 'deu': 'Europe'}
    gdp_data = pd.DataFrame([{'geo': g, 'time': y, 'gdppercapita_us_inflation_adjusted': 1000 + (y-1950)*500 + np.random.uniform(-500,500)} for g in geos for y in years])
    life_expectancy_data = pd.DataFrame([{'geo': g, 'time': y, 'life_expectancy_at_birth_data_from_ihme': 40 + (y-1950)*0.5 + np.random.uniform(-1,1)} for g in geos for y in years])
    population_data = pd.DataFrame([{'geo': g, 'time': y, 'total_population_with_projections': 1e7 + (y-1950)*1e5 + np.random.uniform(-1e5,1e5)} for g in geos for y in years])
    region_data = pd.DataFrame([{'geo': g, 'name': countries[g], 'world_4region': regions[g]} for g in geos])


# 2. Merge the datasets into a single DataFrame.
# We'll merge on both 'geo' (country code) and 'time' (year).
# The region data is merged on 'geo' only.
df = pd.merge(gdp_data, life_expectancy_data, on=['geo', 'time'], how='inner')
df = pd.merge(df, population_data, on=['geo', 'time'], how='inner')
df = pd.merge(df, region_data[['country', 'world_4region', 'name']].rename(columns={'country':'geo'}), on='geo', how='left')
del(gdp_data, life_expectancy_data, population_data, region_data)  # Clean up to save memory

# Drop any rows with missing values
df = df.dropna()

# 3. Define the time slider selection.
# The slider will be based on the 'time' column.
min_year = int(df['time'].min())
max_year = int(df['time'].max())

slider = alt.binding_range(
    min=min_year,
    max=max_year,
    step=1,
    name='    Year'
)

select_year = alt.selection_point(
    name='slider',
    fields=['time'], # what column does the slider link to.
    bind=slider,
    value=min_year
)

# 4. Create the scatter plot with the specified encodings.
# Calculate axes' boundaries
min_gdp = df['gdppercapita_us_inflation_adjusted'].min()
max_gdp = df['gdppercapita_us_inflation_adjusted'].max()
min_life_exp = df['life_expectancy_at_birth_data_from_ihme'].min()
max_life_exp = df['life_expectancy_at_birth_data_from_ihme'].max()

# This selection will toggle on and off when a point is clicked.
select_country = alt.selection_point(
    fields=['name'],
    toggle=True,
    on='click',
    empty='none',  # No countries selected by default
)

# This functionality does not exist
# # Define a new selection for the log scale toggle.
# # This is a single-value selection bound to a checkbox.
# log_scale_toggle = alt.selection_point(
#     name='log_scale_toggle',
#     fields=['log_scale'], # The field name doesn't matter much here
#     bind=alt.binding_checkbox(name='Log Scale for GDP:'),
#     value=False # The initial value is unchecked
# )
# Errored with both: 1) scale=alt.Scale(type=alt.condition(log_scale_toggle, 'log', 'linear')
# and 2) x=alt.condition(log_scale_toggle, alt.X(..., scale(type='log')), alt.X(..., scale(type='linear')))

# The `transform_filter` part ensures only the data for the selected year is shown.
base_chart = alt.Chart(df).mark_circle(size=60).encode(
    x=alt.X('gdppercapita_us_inflation_adjusted:Q',
            title='GDP per Capita (US inflation adjusted)',
            #scale=alt.Scale(type="log"), # Use a log scale for better distribution
            scale=alt.Scale(type="log", domain=[min_gdp, max_gdp]), # Use constant domain for x-axis
    ),
    y=alt.Y('life_expectancy_at_birth_data_from_ihme:Q',
            title='Life Expectancy at Birth', #),
            scale=alt.Scale(domain=[min_life_exp, max_life_exp])), # Use constant domain for y-axis
    color=alt.Color('world_4region:N', title='World Region'),
    size=alt.Size('total_population_with_projections:Q',
                  title='Population',
                  scale=alt.Scale(range=[10, 5000]),  # Adjust circle size range
                  legend=None),   # remove non-intuitive scale legend
    tooltip=[
        alt.Tooltip('name:N', title='Country'),
        alt.Tooltip('gdppercapita_us_inflation_adjusted:Q', title='GDP per Capita', format='$,.2f'),
        alt.Tooltip('life_expectancy_at_birth_data_from_ihme:Q', title='Life Expectancy', format='.1f'),
        alt.Tooltip('world_4region:N', title='Region')
    ]
).add_params(
    select_year,
    select_country,
).transform_filter(
    select_year
).properties(
    width=600,
    height=600
)


# 5. Create a text chart to display country labels on click.
text_chart = alt.Chart(df).mark_text(
    align='left',
    dx=7,  # Offset the label to the right of the point
    dy=-5, # Offset the label up a bit
    color='black', # Set the text color to black
    fontSize=12,   # Set a consistent font size
).encode(
    # Only show the text if the country is in the `select_country` selection
    text=alt.condition(select_country, 'name:N', alt.value('')),
    # We need to explicitly define the other encodings to avoid inheriting from base_chart
    x=alt.X('gdppercapita_us_inflation_adjusted:Q'),
    y=alt.Y('life_expectancy_at_birth_data_from_ihme:Q'),
).transform_filter(
    select_country, # Filter to show labels only for selected countries
    select_year,
)

# 6. Combine the scatter plot and the text chart.
layered_chart = base_chart + text_chart


# 7. Create a title for the chart that updates with the slider.
# This uses a transform to filter the data for the title based on the selection.
title_text = alt.Chart(df).mark_text(
    align='center',
    baseline='middle',
    fontSize=20,
).encode(
    text=alt.condition(
        select_year,
        alt.Text('time:O', format='d', title='Year'),
        alt.value('')
    )
).add_params(select_year).transform_filter(select_year)


# 8. Combine the scatter plot and the title.
final_chart = alt.vconcat(title_text, layered_chart,
    title='Gapminder-style Data over Time'
).interactive()

# 9. Save the chart as an HTML file.
final_chart.save('interactive_gapminder_chart.html')

print("Chart saved to interactive_gapminder_chart.html")


