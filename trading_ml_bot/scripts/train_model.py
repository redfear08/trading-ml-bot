import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from utils.features import add_features
from tqdm import tqdm
import time

# Load historical data
df = pd.read_csv('../data/historical_data.csv')
df = add_features(df)

# Prepare data for machine learning
X = df[['SMA_50', 'SMA_200', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'MACD', 'MACD_Signal', 'Stochastic_%K', 'Stochastic_%D', 'Momentum', 'VWAP']]
y = df['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using Grid Search with progress bar
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           scoring='accuracy',
                           verbose=0)

# Custom fit method to include progress bar
def fit_with_progress_bar(grid_search, X_train, y_train):
    n_iterations = len(grid_search.cv_results_.get('mean_test_score', [])) * len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])
    pbar = tqdm(total=n_iterations, desc="GridSearch Progress", unit="iteration")

    class CustomCallback:
        def __init__(self):
            self.start_time = time.time()

        def __call__(self, x):
            elapsed_time = time.time() - self.start_time
            remaining_time = (n_iterations - pbar.n) * (elapsed_time / pbar.n) if pbar.n > 0 else 0
            pbar.set_postfix(elapsed=f"{elapsed_time:.2f}s", remaining=f"{remaining_time:.2f}s")
            pbar.update(1)

    grid_search.fit(X_train, y_train, callback=CustomCallback())
    pbar.close()

fit_with_progress_bar(grid_search, X_train, y_train)

best_model = grid_search.best_estimator_

# Save the trained model
joblib.dump(best_model, '../models/best_model.pkl')

# Evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
