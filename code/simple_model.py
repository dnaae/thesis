# %%
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# %%
infolder = Path('..')

# Load merged data
data = pd.read_csv(infolder / 'data/merged_data.csv')
data.rename(columns={'Unnamed: 0': 'subject'}, inplace=True)
data.set_index('subject', inplace=True)

# Split into connectivity (X) vs behaviour/demographics (y)
X_data = data[[c for c in data.columns if c.startswith('conn_')]]
y_data = data[[c for c in data.columns if not c.startswith('conn_')]]

# Train/test split
train_ind, test_ind = train_test_split(
    np.arange(X_data.shape[0]),
    test_size=0.2, random_state=42
)
X_train_df, y_train_df = X_data.iloc[train_ind], y_data.iloc[train_ind]
# %%
# Write an sklearn pipeline with a feature scaler, PCA and a ridge regression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
import numpy as np

# Define pipeline
pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('var_thresh', VarianceThreshold(threshold=0.01)),
    ('pca', PCA()),
    ('regressor', Ridge())
])

# Define hyperparameter grid
param_dist = {
    'pca__n_components': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'regressor__alpha': np.logspace(-4, 4, 10)
}

# Nested cross-validation using ShuffleSplit
outer_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Store the R² scores from outer splits
nested_scores = []

for train_idx, test_idx in outer_cv.split(X_data):
    # Split data for this fold
    X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
    y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]

    # Inner cross-validation for hyperparameter tuning
    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=50,  # Fewer iterations for faster search
        cv=3,  # Inner CV splits
        scoring='r2', # Use R² for scoring
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train['FS_IntraCranial_Vol'])

    # Evaluate on outer test split
    test_score = random_search.score(X_test, y_test['FS_IntraCranial_Vol'])
    nested_scores.append(test_score)

    print(f"Outer Fold Test R² Score: {test_score}")
    print(f"Best Parameters in Inner CV: {random_search.best_params_}")

# Print final results
print("Nested CV R² Scores:", nested_scores)
print("Mean Nested CV R²:", np.mean(nested_scores))
print("Standard Deviation of Nested CV R²:", np.std(nested_scores))

# Fit and evaluate
random_search.fit(X_train_df, y_train_df['FS_IntraCranial_Vol'])
print("Best R² score:", random_search.best_score_)
print("Best parameters:", random_search.best_params_)