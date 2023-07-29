import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('/Users/andressanchez/Desktop/Data Scientist/Professional Course on Machine Learning with Scikit-Learn/data/candy.csv')
    print(df.head(5))

    # Drop the competitorname column from the dataset
    X = df.drop('competitorname', axis=1)

    # Fit a MeanShift model to the data and predict the clusters for each data point
    meanshift = MeanShift().fit(X)
    print('*'*64)
    print(max(meanshift.labels_))
    print('='*64)
    print(meanshift.cluster_centers_)

    # Add a new column to the dataframe with the predicted cluster for each data point
    df['group'] = meanshift.labels_

    # Print the updated dataframe with the predicted clusters
    print('='*64)
    print(df)