from sklearn.ensemble import RandomForestClassifier


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import warnings

# Ignore warnings for now (specifically, convergence warnings)
warnings.filterwarnings("ignore")

# Step 1: Data Loading
train_data = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Step 2: Exploratory Data Analysis (EDA)
print("Train Data:")
print(train_data.head())
print(train_data.info())

print("\nTest Data:")
print(test_df.head())
print(test_df.info())

print(train_data.isnull().sum())

# Handle missing values
imputer = SimpleImputer(strategy='mean')
train_data['Cholesterol'] = imputer.fit_transform(train_data[['Cholesterol']])

# Convert categorical variables using one-hot encoding
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_cols = encoder.fit_transform(train_data[categorical_cols])
encoded_cols_names = encoder.get_feature_names_out(categorical_cols)
encoded_cols_df = pd.DataFrame(encoded_cols, columns=encoded_cols_names)
train_data.drop(categorical_cols, axis=1, inplace=True)
train_data = pd.concat([train_data, encoded_cols_df], axis=1)

# Scale numerical variables
scaler = StandardScaler()
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])

# Splitting the data into features (X) and target variable (y)
X = train_data.drop(['HeartDisease', 'PatientId'], axis=1)
y = train_data['HeartDisease']

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Hyperparameter Tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Step 4: Model Training and Evaluation (Random Forest Classifier)
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 5: Preprocessing Test Data and Making Predictions
X_train = train_data.drop(['HeartDisease', 'PatientId'], axis=1)
y_train = train_data['HeartDisease']

# Preprocessing Test Data
X_test = test_df.drop('PatientId', axis=1)

# Handle missing values
test_df['Cholesterol'] = imputer.transform(test_df[['Cholesterol']])

# Encode categorical variables using one-hot encoding
encoded_test_cols = encoder.transform(test_df[categorical_cols])
encoded_test_cols_df = pd.DataFrame(encoded_test_cols, columns=encoded_cols_names)
test_df.drop(categorical_cols, axis=1, inplace=True)
test_df = pd.concat([test_df, encoded_test_cols_df], axis=1)

# Scale numerical variables
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

# Make Predictions
test_predictions = best_model.predict(test_df.drop('PatientId', axis=1))

# Creating Submission DataFrame and Saving to CSV
submission_df = pd.DataFrame({'PatientId': test_df['PatientId'], 'HeartDisease': test_predictions})
submission_df.to_csv('submission2.csv', index=False)
