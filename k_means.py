import pandas as pd

from sklearn.cluster import MiniBatchKMeans # Low memory usage

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('/Users/andressanchez/Desktop/Data Scientist/Professional Course on Machine Learning with Scikit-Learn/data/candy.csv')
    print(df.head(5))

    # Drop the competitorname column from the dataset
    X = df.drop('competitorname', axis=1)

    # Fit a MiniBatchKMeans model to the data and predict the clusters for each data point
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print('*'*64)
    print('Total of centers: {}'.format(len(kmeans.cluster_centers_)))
    print('='*64)
    print(kmeans.predict(X))

    # Add a new column to the dataframe with the predicted cluster for each data point
    df['group'] = kmeans.predict(X)

    # Print the updated dataframe with the predicted clusters
    print(df)