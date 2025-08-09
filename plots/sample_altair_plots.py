#import polars as pd
import altair as alt
from sklearn.datasets import load_iris

# Load iris dataset and convert to DataFrame
iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

# 1. Interactive Scatter Plot: Sepal Length vs Sepal Width
scatter = alt.Chart(df).mark_point().encode(
    x='sepal length (cm)',
    y='sepal width (cm)',
    color='species',
    tooltip=['species', 'sepal length (cm)', 'sepal width (cm)']
).interactive().properties(
    title="Sepal Length vs Sepal Width (Iris)"
)

# 2. Interactive Bar Chart: Count of Each Species
bar = alt.Chart(df).mark_bar().encode(
    x=alt.X('species:N', title='Species'),
    y=alt.Y('mean(petal length (cm)):Q', title='mean Petal Length (cm)'),
    color='species:N',
    tooltip=['species', 'count():Q']
).properties(
    title="Mean Petal Length of Each Iris Species"
)

# 3. Interactive Histogram: Petal Length Distribution
hist = alt.Chart(df).mark_bar(opacity=.5).encode(
    alt.X('petal length (cm):Q', bin=alt.Bin(maxbins=30), title='Petal Length (cm)'),
    y='count()',
    color='species',

    tooltip=['species', 'count()']
).properties(
    title="Histogram of Petal Length"
).interactive()

# Optional: Facet the histogram by species
hist.facet(row=alt.Facet("species", title="Species", ).header( titleFontSize=24, labelFontSize=22))

# 4. Heatmap: Correlation Matrix
corr = df.drop(columns='species').corr().stack().reset_index()
corr.columns = ['feature_x', 'feature_y', 'correlation']

heatmap = alt.Chart(corr).mark_rect().encode(
    x=alt.X('feature_x:N', title=None),
    y=alt.Y('feature_y:N', title=None),
    color=alt.Color('correlation:Q', scale=alt.Scale(scheme='redblue')),
    tooltip=['feature_x', 'feature_y', 'correlation']
).properties(
    title="Feature Correlation Heatmap"
)
heatmap

# To display in a Jupyter notebook, use:
# scatter | bar & hist | heatmap

if __name__ == "__main__":
    scatter.show()
    bar.show()
    hist.show()
    heatmap.show()