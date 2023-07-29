import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('./data/heart.csv')
    print(df['target'].describe())

    # Split the dataset into features (X) and target (y)
    X = df.drop(['target'], axis=1)
    y = df['target']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    # Fit a Gradient Boosting classifier to the training data and evaluate its performance on the testing data
    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    boost_pred = boost.predict(X_test)
    print('=' * 64)
    print('Gradient Boosting Accuracy: ', accuracy_score(y_test, boost_pred))