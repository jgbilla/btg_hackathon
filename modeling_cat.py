import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression


# Load the data
data = pd.read_csv('./data/compas-scores.csv')

# Define features and target
features = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'days_b_screening_arrest']
categorical_features = ['sex', 'race', 'c_charge_degree',]
target = 'score_text'


# Select features and target
X = data[features + categorical_features]

y = data[target].astype(str)  # Convert target to string

# Print the number of rows before imputation
print(f"Rows in original dataset: {len(data)}")

# Remove 'nan' values from the target
mask = y != 'nan'
X = X[mask]
y = y[mask]

# Print the number of rows after removing 'nan' from target
print(f"Rows after removing 'nan' from target: {len(X)}")


# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
   
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



param_grid = {
    'model__n_estimators': [100, 150, 200, 300],
    'model__max_depth': [3, 4, 5],
    'model__learning_rate': [0.01, 0.1, 0.3]
}


# Create the full pipeline with XGBoost
# model = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
# model = RandomForestClassifier(n_estimators=300,max_depth=5, random_state=42)
# model = LogisticRegression(random_state=42)
model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.01, max_depth=4, random_state=42)


model1 = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
model2 = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
model3 = LogisticRegression(random_state=42)

ensemble = VotingClassifier(
    estimators=[('xgb', model1), ('rf', model2), ('lr', model3)],
    voting='soft'
)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)


def train_model():
    pipeline.fit(X_train, y_train)

def test_model():
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.4f}')
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def predict_score(data):
    # Convert the input data to a DataFrame with a single row
    input_data = pd.DataFrame([data], columns=features + categorical_features)
    prediction = pipeline.predict(input_data)
    return le.inverse_transform(prediction)[0]

train_model()
# if __name__ == '__main__':
    
#     feature_importance = pipeline.named_steps['model'].feature_importances_
#     feature_names = (pipeline.named_steps['preprocessor']
#                 .named_transformers_['cat']
#                 .named_steps['onehot']
#                 .get_feature_names_out(categorical_features).tolist())
#     feature_names = features + feature_names
#     for name, importance in zip(feature_names, feature_importance):
#         print(f"{name}: {importance}")

#     test_model()