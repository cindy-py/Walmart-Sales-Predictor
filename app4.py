# Import necessary libraries
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("Walmart.csv")

# Display the image
st.image("Walmart.jpg")

# Add title to the app
st.title("Random Forest Regression App")

# Add header
st.header("About.", divider="rainbow")

# Add paragraph
st.write("""The Walmart dataset contains information about weekly sales and various factors that may influence them. 
            In this app, we'll use Random Forest Regression to predict weekly sales based on these factors.
            This model analyzes historical sales data to forecast future weekly sales for various Walmart stores. 
            Using this predictive model, retailers and stakeholders can make informed decisions about inventory management, staffing, 
            and promotional strategies to optimize their business operations.""")

# Display exploratory data analysis (EDA)
st.header("Exploratory Data Analysis (EDA).", divider="rainbow")

if st.checkbox("Show Dataset Info"):
    st.write("Dataset Info:", df.info())

if st.checkbox("Number of Rows"):
    st.write("Number of Rows:", df.shape[0])

if st.checkbox("Column Names"):
    st.write("Column Names:", df.columns.tolist())

if st.checkbox("Data Types"):
    st.write("Data Types:", df.dtypes)

if st.checkbox("Missing Values"):
    st.write("Missing Values:", df.isnull().sum())

if st.checkbox("Statistical Summary"):
    st.write("Statistical Summary:", df.describe())

# Split the dataset into features (X) and target (y)
X = df.drop(columns=['Weekly_Sales'])
y = df['Weekly_Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the regression model
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)

# Make predictions
y_pred = rfr.predict(X_test)

# Evaluate the model
#mse = mean_squared_error(y_test, y_pred)
#st.write("Mean Squared Error:", mse)

r2_score = r2_score(y_test, y_pred)
st.write("r2_score:", r2_score)

# Prediction section
st.sidebar.title("Enter Values to Predict Weekly Sales")

# Create input fields for each feature
user_input = {}
for feature in df.columns[:-1]:
    user_input[feature] = st.sidebar.text_input(f"Enter {feature}", 0.0)

# Button to trigger the prediction
if st.sidebar.button("Predict"):
    # Create DataFrame for user input
    user_input_df = pd.DataFrame([user_input])
    # Predict using the trained model
    prediction = rfr.predict(user_input_df)
    # Display the predicted result
    st.write('Predicted Weekly Sales:', prediction[0])