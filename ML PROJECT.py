import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r'C:\Users\raish\OneDrive\Pictures\healthcare-dataset-stroke-data.csv')

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
data[['bmi']] = imputer.fit_transform(data[['bmi']])

# Encode categorical variables
label_encoders = {}
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# One-hot encode 'stroke' column
one_hot_encoder = OneHotEncoder(sparse_output=False)
stroke_encoded = one_hot_encoder.fit_transform(data[['stroke']])
data = data.drop('stroke', axis=1)
stroke_df = pd.DataFrame(stroke_encoded, columns=['stroke_0', 'stroke_1'])
data = pd.concat([data, stroke_df], axis=1)

# Split the dataset into features and target variable
X = data.drop(['id', 'stroke_0', 'stroke_1'], axis=1)
y = data[['stroke_1']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a RandomForestClassifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train.values.ravel())

# Make predictions
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Function to get user input and predict
def predict_stroke():
    user_data = {}
    user_data['gender'] = input("Enter gender (Male/Female): ").capitalize()
    user_data['age'] = float(input("Enter age: "))
    user_data['hypertension'] = int(input("Enter hypertension (0/1): "))
    user_data['heart_disease'] = int(input("Enter heart disease (0/1): "))
    user_data['ever_married'] = input("Enter marital status (Yes/No): ").capitalize()
    user_data['work_type'] = input("Enter work type (Private/Self-employed/Govt_job/Children/Never_worked): ").capitalize()
    user_data['Residence_type'] = input("Enter residence type (Urban/Rural): ").capitalize()
    user_data['avg_glucose_level'] = float(input("Enter average glucose level: "))
    user_data['bmi'] = float(input("Enter BMI: "))
    user_data['smoking_status'] = input("Enter smoking status (Formerly smoked/Never smoked/Smokes): ").capitalize()

    user_df = pd.DataFrame(user_data, index=[0])

    # Preprocess the user input
    for col in categorical_columns:
        if col in user_df:
            try:
                user_df[col] = label_encoders[col].transform(user_df[col])
            except ValueError:
                # If the label is unseen, add it to the encoder
                label_encoders[col].classes_ = np.append(label_encoders[col].classes_, user_df[col].values[0])
                user_df[col] = label_encoders[col].transform(user_df[col])

    # Scale the user input
    user_df = scaler.transform(user_df)

    # Predict stroke for the user input
    prediction = classifier.predict(user_df)
    prediction_proba = classifier.predict_proba(user_df)

    # Output the prediction and probability
    print(f'Prediction (0: No Stroke, 1: Stroke): {prediction[0]}')
    print(f'Prediction Probability: {prediction_proba[0]}')

# Call the function to predict based on user input
predict_stroke()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Histogram of predicted probabilities
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba, bins=20, color='blue', edgecolor='k', alpha=0.7)
plt.title('Histogram of Predicted Probabilities')
plt.xlabel('Predicted Probability of Stroke')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of predicted probabilities
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_pred_proba)), y_pred_proba, color='blue', alpha=0.7)
plt.title('Scatter Plot of Predicted Probabilities')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Probability of Stroke')
plt.show() 