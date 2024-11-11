import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load the data
data = pd.read_csv('./data/compas-scores.csv')

# Define features and target
features = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'days_b_screening_arrest']
categorical_features = ['sex', 'race', 'c_charge_degree']
target = 'score_text'

# Select features and target
X = data[features + categorical_features]
y = data[target].astype(str)

# Remove 'nan' values from the target
mask = y != 'nan'
X = X[mask]
y = y[mask]

print(f"Rows in original dataset: {len(data)}")
print(f"Rows after removing 'nan' from target: {len(X)}")

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    # ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Feature selection
selector = SelectKBest(f_classif, k=10)  # Select top 10 features

# Create base models
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.01, max_depth=4, random_state=42)
lr = LogisticRegression(random_state=42)
svm = SVC(probability=True, random_state=42)

# Create the ensemble model
ensemble = VotingClassifier(
    estimators=[('xgb', xgb), ('rf', rf), ('gb', gb), ('lr', lr), ('svm', svm)],
    voting='soft',
    weights=[2, 1, 1, 1, 1]
)

# Create the full pipeline
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    # ('selector', selector),
    # ('smote', SMOTE(random_state=42)),
    ('classifier', ensemble)
])

# Hyperparameter tuning
param_grid = {
    'classifier__xgb__n_estimators': [50, 100],
    'classifier__xgb__learning_rate': [0.01, 0.1],
    'classifier__rf__n_estimators': [50, 100],
    'classifier__rf__max_depth': [None, 10],
    'classifier__gb__n_estimators': [50, 100],
    'classifier__gb__learning_rate': [0.01, 0.1],
    'classifier__lr__C': [0.1, 1.0],
    'classifier__svm__C': [0.1, 1.0],
}

def train_model():
    # grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    # grid_search.fit(X_train, y_train)
    # print("Best parameters:", grid_search.best_params_)
    # return grid_search.best_estimator_
    pipeline.fit(X_train, y_train)
    return pipeline

def test_model(model):
    y_pred = model.predict(X_test)
    
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

def cross_validate(model):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV accuracy: {scores.mean():.4f}")

def predict_score(model, data):
    input_data = pd.DataFrame(data, columns=features + categorical_features)
    prediction = model.predict(input_data)
    return le.inverse_transform(prediction)[0]

if __name__ == '__main__':
    best_model = train_model()
    test_model(best_model)
    # cross_validate(best_model)