# UWICOMP6130 Heart Disease Prediction

This project focuses on predicting heart disease using machine learning techniques. The dataset used in this project is obtained from Kaggle under the competition titled "UWICOMP6130 Heart Disease Prediction" by Matthew Stone.

## Citation
- Stone, Matthew. "UWICOMP6130 Heart Disease Prediction." Kaggle. 2024. [https://kaggle.com/competitions/uwicomp6130-heart-disease-prediction](https://kaggle.com/competitions/uwicomp6130-heart-disease-prediction)

## Project Overview

The heart disease prediction task involves several steps:

1. **Data Loading:** The dataset is loaded using pandas from CSV files for training and testing.

2. **Preprocessing:** Data preprocessing involves handling missing values, encoding categorical variables using one-hot encoding, and scaling numerical features.

3. **Model Training:** Support Vector Machine (SVM) classification model is trained using hyperparameter tuning with grid search and cross-validation.

4. **Evaluation:** The model is evaluated using cross-validation to estimate its performance.

5. **Making Predictions:** The trained model is used to make predictions on the test dataset.

Additionally, the project includes `train.py`, which utilizes Random Forest for model training, and `gradientb.py`, which utilizes Gradient Boosting.

For further details, refer to the provided code files.
