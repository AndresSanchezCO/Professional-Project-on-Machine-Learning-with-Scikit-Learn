import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score, KFold
)

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('/Users/andressanchez/Desktop/Data Scientist/Professional Course on Machine Learning with Scikit-Learn/data/felicidad.csv')

    # Split the dataset into features (X) and target (y)
    X = df.drop(['country', 'rank', 'score'], axis=1)
    y = df['score']

    # Fit a Decision Tree Regressor model to the data using cross-validation
    model = DecisionTreeRegressor()
    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error') # Fast validation 
    print(np.abs(np.mean(score))) # Mean of the absolute value of the score

    # Use KFold cross-validation to split the data into training and testing sets
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(df):
        print(train)
        print(test)
        print('*'*64)