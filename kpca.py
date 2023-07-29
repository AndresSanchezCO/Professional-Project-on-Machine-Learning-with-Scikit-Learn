import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA



from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load the heart dataset
    dt_heart = pd.read_csv('./data/heart.csv')

    # Print the first 5 rows of the dataset
    print(dt_heart.head(5))

    # Split the dataset into features and target
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    # Standardize the features
    dt_features = StandardScaler().fit_transform(dt_features)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    kpca = KernelPCA(n_components=4, kernel='poly') # Create a KernelPCA object with 4 components and a polynomial kernel
    kpca.fit(X_train) # Fit the KernelPCA object to the training data

    dt_train = kpca.transform(X_train) # Transform the training data
    dt_test = kpca.transform(X_test) # Transform the test data

    # Logistic regression
    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(dt_train, y_train) # Fit the logistic regression model to the transformed training data
    print("SCORE KPCA: ", logistic.score(dt_test, y_test)) # Print the accuracy score of the model on the transformed test data