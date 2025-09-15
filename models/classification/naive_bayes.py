from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import polars as pd
import altair as alt

# Load the iris dataset
iris = load_iris(as_frame=True)
# Convert to Polars DataFrame
df = pd.DataFrame(iris.frame)
# Rename columns for easier access
df = df.rename({
    'sepal length (cm)': 'sepal_length',
    'sepal width (cm)': 'sepal_width',
    'petal length (cm)': 'petal_length',
    'petal width (cm)': 'petal_width',
    'target': 'species'
})
# Map species names to integers
df = df.with_columns(pd.col("species").cast(str))
df = df.with_columns(pd.col("species").replace(dict(enumerate(iris.target_names))))


# Split the dataset into features and target variable
X = df.drop('species')
y = df['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)
# Create a Gaussian Naive Bayes classifier

gnb = GaussianNB()
# Fit the model on the training data
gnb.fit(X_train, y_train)
# Make predictions on the test set
y_pred = gnb.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
# Convert confusion matrix to Polars DataFrame for better visualization
conf_matrix_df = pd.DataFrame(conf_matrix, schema=['Predicted Setosa', 'Predicted Versicolor', 'Predicted Virginica'])


# Extra: plots

# 1. Interactive Scatter Plot: Sepal Length vs Sepal Width
scatter = alt.Chart(df).mark_point().encode(
    x='sepal_length',
    y='sepal_width',
    color='species',
    tooltip=['species', 'sepal_length', 'sepal_width']
).interactive().properties(
    title="Sepal Length vs Sepal Width (Iris)"
)

# 1a. Combine train and test data for visualization
import pandas as pd
df1 = pd.DataFrame(X_train, columns=X_train.columns)
df1['species'] = y_train
df1['test'] = 0
df1['prediction'] = None
df1['correct'] = None
df2 = pd.DataFrame(X_test, columns=X_test.columns)
df2['species'] = y_test
df2['test'] = 1
df2['prediction'] = y_pred
df2['correct'] = df2['species'] == df2['prediction']
df1 = pd.concat([df1, df2], axis=0, ignore_index=True)
df1

# Define shape and fill logic
df1["shape"] = np.where(
    (df1["test"] == 0) | ((df1["test"] == 1) & (df1["correct"] == True)),
    "circle",
    "triangle"
)
df1["fill"] = np.where(df1["test"] == 0, "filled", "open")


# TODO: fix shape
# Scatter plot with shape and fill
scatter = (alt.Chart(df1)
    .add_params(x_select, y_select)
    .transform_calculate(
        xval=f"datum[{x_select.name}]",
        yval=f"datum[{y_select.name}]"
    )
    .mark_point().encode(
    x='sepal_length',
    y='sepal_width',
    color='species',
    shape=alt.Shape('shape:N', legend=alt.Legend(title="Shape")),
    fillOpacity=alt.condition(
        alt.datum.fill == "filled", alt.value(1), alt.value(0)
    ),
        tooltip=[
        'species',
        'sepal_length',
        'sepal_width',
        'test',
        alt.Tooltip('species', title='Actual'),
        alt.Tooltip('prediction', title='Predicted'),
        'correct'
    ]
).properties(
    title="Sepal Length vs Sepal Width (Train/Test, Correct/Incorrect)",
    width=800,
    height=800
).interactive()
)
scatter


# 2. Interactive Bar Chart: Count of Each Species
bar = alt.Chart(df).mark_bar().encode(
    x=alt.X('species:N', title='Species'),
    y=alt.Y('count():Q', title='Count'),
    color='species:N',
    tooltip=['species', 'count():Q']
).properties(
    title="Count of Each Iris Species"
)
# 3. Interactive Histogram: Petal Length Distribution         
