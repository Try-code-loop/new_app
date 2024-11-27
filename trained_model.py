import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix
import sklearn
import imblearn

data = pd.read_csv('C:/Users/Srnzzz/Documents/5- Ironhack/ML/heart_disease_health_indicators_BRFSS2015.csv', delimiter=';')

# 'HeartDiseaseorAttack' as target variable (y)
features = data.drop('HeartDiseaseorAttack', axis=1)
target = data['HeartDiseaseorAttack']

# Data split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Noramlize only those that are non-binary
columns_to_normalize = X_train.columns[X_train.nunique() > 2]
normalizer = MinMaxScaler()
X_train_norm = X_train.copy()  
X_train_norm[columns_to_normalize] = normalizer.fit_transform(X_train[columns_to_normalize])

X_test_norm = X_test.copy()
X_test_norm[columns_to_normalize] = normalizer.transform(X_test[columns_to_normalize])

# Undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_norm, y_train)

# Class distribution check before and after undersampling
print("Original class distribution:")
print(y_train.value_counts(normalize=True))
print("\nResampled class distribution:")
print(y_train_resampled.value_counts(normalize=True))

# Random Forest model initialzing
base_rf = RandomForestClassifier(random_state=42)

# Hyperparameters
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Randomized search for hyperparameter tuning
random_search = RandomizedSearchCV(
    base_rf, 
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1,
    scoring='recall' 
)

# Model fit
random_search.fit(X_train_resampled, y_train_resampled)

# Get the best model
final_model = random_search.best_estimator_

# Saving trained model and normalizer
with open('model.pkl', 'wb') as file:
    pickle.dump(final_model, file)
with open('normalizer.pkl', 'wb') as file:
    pickle.dump(normalizer, file)
with open('columns_to_normalize.pkl', 'wb') as file:
    pickle.dump(columns_to_normalize, file)




print("Scikit-learn version:", sklearn.__version__)
print("Imbalanced-learn version:", imblearn.__version__)

# For specific submodules, you can check if they are available
print("Model selection module available:", hasattr(sklearn, 'model_selection'))
print("Preprocessing module available:", hasattr(sklearn, 'preprocessing'))
print("Ensemble module available:", hasattr(sklearn, 'ensemble'))
print("Metrics module available:", hasattr(sklearn, 'metrics'))
print("Under sampling module available:", hasattr(imblearn, 'under_sampling'))