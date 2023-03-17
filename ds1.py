import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import datasets
from sklearnex import patch_sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Importing Intel Optimized Scikit-learn
# import daal4py.sklearn

#patch Scikit Learn
patch_sklearn()

# Loading the diabetes dataset
X, y = load_diabetes(return_X_y=True)
df = pd.DataFrame(data=X, columns=load_diabetes().feature_names)
df["Target"] = y

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting the Ridge model on the training data
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Displaying the mean squared error using Streamlit
st.write(f"Mean squared error: {mse}")

# Interactive visualization of the diabetes dataset using Plotly
fig = px.scatter_matrix(df, dimensions=load_diabetes().feature_names, color="Target")
st.plotly_chart(fig)
