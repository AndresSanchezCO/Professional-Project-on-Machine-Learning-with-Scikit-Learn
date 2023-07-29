import pandas as pd

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

    # Fit a KNN classifier to the training data and evaluate its performance on the testing data
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print('=' * 64)
    print('KNN Accuracy: ', accuracy_score(y_test, knn_pred))

    # Fit a Bagging classifier with KNN as the base estimator to the training data and evaluate its performance on the testing data
    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)
    print('=' * 64)
    print('Bagging Accuracy: ', accuracy_score(y_test, bag_pred))

    # Random forest 
    from sklearn.ensemble import RandomForestClassifier
    rf_class = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
    rf_pred = rf_class.predict(X_test)
    print('=' * 64)
    print('Random Forest Accuracy: ', accuracy_score(y_test, rf_pred))

    # Other classifiers
    from sklearn.ensemble import ExtraTreesClassifier
    et_class = ExtraTreesClassifier(n_estimators=50).fit(X_train, y_train)
    et_pred = et_class.predict(X_test)
    print('=' * 64)
    print('Extra Trees Accuracy: ', accuracy_score(y_test, et_pred))

    from sklearn.ensemble import AdaBoostClassifier
    ada_class = AdaBoostClassifier(n_estimators=50).fit(X_train, y_train)
    ada_pred = ada_class.predict(X_test)
    print('=' * 64)
    print('AdaBoost Accuracy: ', accuracy_score(y_test, ada_pred))

    from sklearn.ensemble import GradientBoostingClassifier
    gb_class = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    gb_pred = gb_class.predict(X_test)
    print('=' * 64)
    print('Gradient Boosting Accuracy: ', accuracy_score(y_test, gb_pred))

    # Bagging with Extra Trees
    bag_class = BaggingClassifier(base_estimator=ExtraTreesClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)
    print('=' * 64)
    print('Bagging with Extra Trees Accuracy: ', accuracy_score(y_test, bag_pred))