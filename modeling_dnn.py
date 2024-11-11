import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

data = pd.read_csv('./data/compas-scores.csv')
features = ['age','juv_fel_count','juv_misd_count','juv_other_count','priors_count','days_b_screening_arrest']
#Todo: decide if age of age_cat is better
categorical_features = ['sex','race',  'age_cat', 'c_charge_degree']
target = 'decile_score'

X = data[features + categorical_features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Define the DNN model
def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Train the model
def train_model():
    input_dim = X_train_preprocessed.shape[1]
    model = create_model(input_dim)
    model.fit(X_train_preprocessed, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    return model

# Test the model
def test_model(model):
    y_pred = model.predict(X_test_preprocessed).flatten()
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f'Model R-squared score: {r2:.4f}')
    print(f'Mean Absolute Error: {mae:.4f}')

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Decile Scores')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()

    # Plot the differences
    differences = y_pred - y_test
    plt.figure(figsize=(10, 6))
    plt.hist(differences, bins=30, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig('prediction_errors.png')
    plt.close()

# Prediction function
def predict_decile_score(model, data):
    input_data = pd.DataFrame(data, columns=features + categorical_features)
    preprocessed_data = preprocessor.transform(input_data)
    prediction = model.predict(preprocessed_data)
    return prediction[0][0]

if __name__ == '__main__':
    trained_model = train_model()
    test_model(trained_model)