import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import warnings

# Ignore warnings for now
warnings.filterwarnings("ignore")

# Step 1: Data Loading
train_data = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Step 2: Preprocessing
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

# Step 3: Hyperparameter Tuning using Grid Search and Cross-Validation
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'sigmoid']
}

best_accuracy = 0
best_model = None

svm_classifier = SVC(random_state=42)
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Get the best model
best_model = grid_search.best_estimator_

# Step 4: Model Training and Evaluation with Cross-Validation
cv_scores = cross_val_score(best_model, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Step 5: Preprocessing Test Data and Making Predictions
test_df['Cholesterol'] = imputer.transform(test_df[['Cholesterol']])
encoded_test_cols = encoder.transform(test_df[categorical_cols])
encoded_test_cols_df = pd.DataFrame(encoded_test_cols, columns=encoded_cols_names)
test_df.drop(categorical_cols, axis=1, inplace=True)
test_df = pd.concat([test_df, encoded_test_cols_df], axis=1)
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

# Make Predictions
test_predictions = best_model.predict(test_df.drop('PatientId', axis=1))

# Creating Submission DataFrame and Saving to CSV
submission_df = pd.DataFrame({'PatientId': test_df['PatientId'], 'HeartDisease': test_predictions})
submission_df.to_csv('submission_svm_hype.csv', index=False)
