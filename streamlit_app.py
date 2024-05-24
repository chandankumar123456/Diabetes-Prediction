import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load data
df = pd.read_csv('diabetes.csv')
df.drop(columns=['Pregnancies', 'SkinThickness'], inplace=True)

# Train-test split
X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cols = ['BloodPressure', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='mean')
ct = ColumnTransformer(transformers=[('imputer', imputer, cols)], remainder='passthrough')
X_train_imputed = ct.fit_transform(X_train)
X_test_imputed = ct.transform(X_test)

# Model training
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'K Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Classifier': SVC()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

# Model evaluation
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(X_test_scaled)

metrics = {}
for name, prediction in predictions.items():
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, prediction),
        'Precision': precision_score(y_test, prediction),
        'Recall': recall_score(y_test, prediction),
        'F1 Score': f1_score(y_test, prediction),
        'ROC AUC': roc_auc_score(y_test, prediction)
    }

# Display metrics
st.title('Diabetes Prediction')
st.subheader('Model Evaluation Metrics')

metrics_df = pd.DataFrame(metrics).T
st.write(metrics_df)

# Prediction based on user input
st.sidebar.title('Make a Prediction')
st.sidebar.write('Enter the following details:')


glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=199, value=100)
blood_pressure = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=72)
insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=30)
bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=67.1, value=30.0)
diabetes_pedigree_function = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.3725)
age = st.sidebar.number_input('Age', min_value=21, max_value=81, value=25)

user_data = np.array([glucose, blood_pressure,  insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)
user_data_scaled = scaler.transform(user_data)

prediction_result = {}
for name, model in models.items():
    prediction_result[name] = model.predict(user_data_scaled)[0]

st.sidebar.subheader('Prediction Results')
for name, result in prediction_result.items():
    st.sidebar.write(f'{name}: {"Diabetic" if result == 1 else "Not Diabetic"}')
