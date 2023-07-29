import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA


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

    print(X_train.shape)
    print(y_train.shape)

    # Create a PCA object
    pca = PCA(n_components=3) # Create a PCA object with 3 components
    pca.fit(X_train) # Fit the PCA object to the training data

    ipca = IncrementalPCA(n_components=3, batch_size=10) # Create an IncrementalPCA object with 3 components and a batch size of 10
    ipca.fit(X_train) # Fit the IncrementalPCA object to the training data

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_) # Plot the explained variance ratio for the PCA object
    #plt.show()

    # Logistic regression
    logistic = LogisticRegression(solver='lbfgs')

    # Train the model using PCA
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train) # Fit the logistic regression model to the transformed training data
    print("SCORE PCA: ", logistic.score(dt_test, y_test)) # Print the accuracy score of the model on the transformed test data

    # Train the model using IPCA
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train) # Fit the logistic regression model to the transformed training data
    print("SCORE IPCA: ", logistic.score(dt_test, y_test)) # Print the accuracy score of the model on the transformed test data