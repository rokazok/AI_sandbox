
# Multiclass classification with neural network
# Adapted from https://codesignal.com/learn/courses/modeling-the-wine-dataset-with-pytorch/lessons/building-a-multi-class-classification-model-with-pytorch


import torch
import torch.nn as nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import altair as alt

wine = load_wine()
# https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset
# each row consists of 13 features: alcohol, malic_acid, ash, alcalinity_of_ash,
# magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
# color_intensity, hue, od280/od315_of_diluted_wines, proline
# the target is the type of wine (0, 1, or 2)
X, y = wine.data, wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

# Plot some of the features
df = pd.concat([pd.DataFrame(X_train), pd.Series(y_train)], axis=1)
df.columns = wine.feature_names + ['target']

x_dropdown = alt.binding_select(options=df.columns, name='X Axis:')
y_dropdown = alt.binding_select(options=df.columns, name='Y Axis:')
x_select = alt.selection_point(fields=['x'], bind=x_dropdown, value=[{'x': df.columns[0]}])
y_select = alt.selection_point(fields=['y'], bind=y_dropdown, value=[{'y': df.columns[1]}])

base = alt.Chart(df).mark_point().encode(
    x=alt.X('x:Q', title=''),
    y=alt.Y('y:Q', title=''),
    color=alt.Color('target:N', title='Target')
).add_params(
    x_select, y_select
).transform_calculate(
    x=f'datum[{x_select.name}_x]',
    y=f'datum[{y_select.name}_y]'
).interactive()

# The above transform_fold is not strictly necessary, but helps with dynamic referencing.
# For Altair 5+, you can use alt.datum and alt.expr.ref for dynamic fields.

# Show the chart
base

# The plot shows the features can cluster the wines.

# Processing
# feature scaling
scaler = StandardScaler().fit(X_train)
# transform the data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long) # 64-bit signed integer
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Define the neural network model
model = nn.Sequential(
    nn.Linear(in_features=13, out_features=10),
    nn.ReLU(), # rectified linear unit
    nn.Linear(in_features=10, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=3) # final layer: 3 classes for wine types
)

# loss function
criterion = nn.CrossEntropyLoss()  # suitable for multi-class classification
# CrossEntroyLoss applies softmax to get from logits to probabilities

# optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001) # Adaptive Moment Estimation

# TRAIN the model
num_epochs = 150
history = {'train_loss': [], 'val_loss': []}

for epoch in range(num_epochs):
    # Forward pass
    model.train()  # set the model to training mode
    optimizer.zero_grad()  # zero the gradients
    outputs = model(X_train_tensor) # forward pass
    loss = criterion(outputs, y_train_tensor) # compute the loss

    # Backward pass and optimization
    loss.backward()  # compute gradients
    optimizer.step()  # update weights

    history['train_loss'].append(loss.item())

    model.eval()  # set the model to evaluation mode

    # Validation loss
    with torch.no_grad():   # disable gradient calculation
        outputs_val = model(X_test_tensor)
        val_loss = criterion(outputs_val, y_test_tensor)
        history['val_loss'].append(val_loss.item())

    # Logging: print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

outputs_val
val_loss