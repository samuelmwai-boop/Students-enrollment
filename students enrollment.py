
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load your historical student enrollment data, academic records, and demographic data
# Replace this with the actual path or method to load your dataset
# For example: df = pd.read_csv('your_data.csv')
# Ensure your dataset includes features like 'enrollment_status' and relevant student information
# For graduation success, you might consider having a target variable like 'graduation_status'
# and additional features related to academic performance
# Modify the data loading process accordingly

# Create a DataFrame
df = pd.read_csv('data.csv')

# 1. Identify the target variables and features
# For enrollment prediction
X_enrollment = df[['age', 'gender', 'score', 'other_features']]
y_enrollment = df['enrollment_status']  # Binary: 1 for enrolled, 0 for not enrolled

# For graduation prediction
X_graduation = df[['age', 'gender', 'score', 'other_features', 'enrollment_status']]
y_graduation = df['graduation_status']  # Binary: 1 for graduated, 0 for not graduated

# 2. Preprocess the data for enrollment prediction
# Use one-hot encoding for categorical variables like 'gender' and 'other_features'
preprocessor_enrollment = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'score']),
        ('cat', OneHotEncoder(), ['gender', 'other_features'])
    ])
# Split the data set
X_train_enrollment, X_test_enrollment, y_train_enrollment, y_test_enrollment = train_test_split(
    X_enrollment, y_enrollment, test_size=0.2, random_state=0)
# Convert the transformed data back to a DataFrame
X_train_transformed_enrollment = preprocessor_enrollment.fit_transform(X_train_enrollment)
X_train_columns_enrollment = ['age', 'score'] + list(
    preprocessor_enrollment.named_transformers_['cat'].get_feature_names_out(['gender', 'other_features']))
X_train_enrollment = pd.DataFrame(X_train_transformed_enrollment, columns=X_train_columns_enrollment)

X_test_transformed_enrollment = preprocessor_enrollment.transform(X_test_enrollment)
X_test_columns_enrollment = ['age', 'score'] + list(
    preprocessor_enrollment.named_transformers_['cat'].get_feature_names_out(['gender', 'other_features']))
X_test_enrollment = pd.DataFrame(X_test_transformed_enrollment, columns=X_test_columns_enrollment)

# 3. Train the model for enrollment prediction
classifier_enrollment = RandomForestClassifier(n_estimators=200, random_state=0)
classifier_enrollment.fit(X_train_enrollment, y_train_enrollment)

# 4. Make predictions and evaluate the model for enrollment prediction
y_pred_enrollment = classifier_enrollment.predict(X_test_enrollment)
print("Enrollment Prediction - Confusion Matrix:")
print(confusion_matrix(y_test_enrollment, y_pred_enrollment))
print("\nEnrollment Prediction - Classification Report:")
print(classification_report(y_test_enrollment, y_pred_enrollment))




# Save the enrollment prediction model for future use
joblib.dump(classifier_enrollment, 'enrollment_model.pkl')

# Optional: Load the enrollment prediction model later if needed
loaded_model_enrollment = joblib.load('enrollment_model.pkl')

# 5. Repeat the process for graduation prediction

# 1. Identify the target variables and features for graduation prediction
X_graduation = df[['age', 'gender', 'score', 'other_features', 'enrollment_status']]
y_graduation = df['graduation_status']  # Binary: 1 for graduated, 0 for not graduated

# 2. Preprocess the data for graduation prediction
# Use one-hot encoding for categorical variables like 'gender' and 'other_features'
preprocessor_graduation = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'score']),
        ('cat', OneHotEncoder(), ['gender', 'other_features'])
    ])
# Split the data set
X_train_graduation, X_test_graduation, y_train_graduation, y_test_graduation = train_test_split(
    X_graduation, y_graduation, test_size=0.2, random_state=0)
# Convert the transformed data back to a DataFrame
X_train_transformed_graduation = preprocessor_graduation.fit_transform(X_train_graduation)
X_train_columns_graduation = ['age', 'score'] + list(
    preprocessor_graduation.named_transformers_['cat'].get_feature_names_out(['gender', 'other_features']))
X_train_graduation = pd.DataFrame(X_train_transformed_graduation, columns=X_train_columns_graduation)

X_test_transformed_graduation = preprocessor_graduation.transform(X_test_graduation)
X_test_columns_graduation = ['age', 'score'] + list(
    preprocessor_graduation.named_transformers_['cat'].get_feature_names_out(['gender', 'other_features']))
X_test_graduation = pd.DataFrame(X_test_transformed_graduation, columns=X_test_columns_graduation)

# 3. Train the model for graduation prediction
classifier_graduation = RandomForestClassifier(n_estimators=200, random_state=0)
classifier_graduation.fit(X_train_graduation, y_train_graduation)

# 4. Make predictions and evaluate the model for graduation prediction
y_pred_graduation = classifier_graduation.predict(X_test_graduation)
print("\nGraduation Prediction - Confusion Matrix:")
print(confusion_matrix(y_test_graduation, y_pred_graduation))
print("\nGraduation Prediction - Classification Report:")
print(classification_report(y_test_graduation, y_pred_graduation))