#import polars as pd
import altair as alt
from sklearn.datasets import load_iris
import numpy as np

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
def quantile_histogram(df, column, target_column=None, quantiles=10, labels=None, width=600, height=300):
    """
    Create two plots: 
    1. Mean of target variable by quantile bin (2/3 height)
    2. Histogram of quantiles for a given column (1/3 height)

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to compute quantiles on.
        target_column (str): The column to compute means for each bin.
        quantiles (int): Number of quantile bins.
        labels (list, optional): List of labels for the bins.

    Returns:
        alt.Chart: Combined Altair chart.
    """
    df = df.copy()
    bins = pd.qcut(df[column], q=quantiles, labels=labels)
    df['quantile_bin'] = bins.astype(str)
    df['bin_order'] = pd.Categorical(bins).codes

    # Create mean value plot (2/3 height)
    if target_column:
        mean_plot = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('quantile_bin:N',
                    title=None,  # Remove x-axis title
                    axis=alt.Axis(labels=False),  # Remove x-axis labels
                    sort=alt.EncodingSortField('bin_order')),
            y=alt.Y(f'mean({target_column}):Q',
                   title=f'Mean {target_column}'),
            tooltip=['quantile_bin', f'mean({target_column}):Q']
        ).properties(
            title=f'Mean {target_column} by {column} Quantiles',
            height=height * 0.67,
            width=width
        )
    
    # Create histogram (1/3 height)
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X('quantile_bin:N',
                title=f'{column} Quantile Bin',
                sort=alt.EncodingSortField('bin_order'),
                axis=alt.Axis(labelAngle=-45)),  # Add tilted labels
        y=alt.Y('count():Q', title='Count'),
        tooltip=['quantile_bin', 'count():Q']
    ).properties(
        title=f'Histogram of {column} Quantiles',
        height=height * 0.33,
        width=width
    )

    # Combine plots vertically
    return (mean_plot & hist) if target_column else hist


# To display in a Jupyter notebook, use:
# scatter | bar & hist | heatmap

if __name__ == "__main__":
    scatter.show()
    bar.show()
    hist.show()
    heatmap.show()
    quantile_hist = quantile_histogram(df=df1, column='petal length (cm)', target_column='petal width (cm)', quantiles=10)
    quantile_hist.show()