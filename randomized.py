import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('./data/felicidad.csv')
    print(df.head(5))
    
    # Split the dataset into features (X) and target (y)
    X = df.drop(['country', 'rank', 'score'], axis=1)
    y = df['score']

    # Create a RandomForestRegressor model
    reg = RandomForestRegressor()

    # Define the hyperparameters to search using RandomizedSearchCV
    params = {
        'n_estimators': range(4, 16),
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': range(2, 11)
    }

    # Use RandomizedSearchCV to find the best hyperparameters for the model
    rand_est = RandomizedSearchCV(reg, params, n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X, y)

    # Print the best estimator and hyperparameters found by RandomizedSearchCV
    print(rand_est.best_estimator_)
    print(rand_est.best_params_)

    # Make a prediction using the best estimator found by RandomizedSearchCV
    print(rand_est.predict(X.loc[[0]]))