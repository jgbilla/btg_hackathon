import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('./data/compas-scores.csv')

features = ['age','juv_fel_count','juv_misd_count','juv_other_count','priors_count','days_b_screening_arrest']
#Todo: decide if age of age_cat is better
categorical_features = ['sex','race',  'age_cat', 'c_charge_degree']
target = 'decile_score'


X = data[features + categorical_features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
    ])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)


pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

def train_model():
    pipeline.fit(X_train, y_train)

def test_model():
    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate R-squared score
    r2 = r2_score(y_test, y_pred)
    
    # Calculate Mean Absolute Error
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


def predict_decile_score(data):
    input_data = pd.DataFrame(data, columns=features + categorical_features)
    prediction = pipeline.predict(input_data)
    return prediction[0]


if __name__ == '__main__':
    train_model()
    test_model()

