import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('./data/felicidad.csv')
    print(df.describe())

    # Split the dataset into features (X) and target (y)
    X = df[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = df[['score']]

    # Print the shapes of X and y
    print(X.shape)
    print(y.shape)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Fit a linear regression model to the training data
    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    # Fit a LASSO regression model to the training data
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    # Fit a Ridge regression model to the training data
    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    # Calculate the mean squared error for each model
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print('Linear Loss: ', linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print('Lasso Loss: ', lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print('Ridge Loss: ', ridge_loss)

    elasticnet = ElasticNet(alpha=0.02).fit(X_train, y_train)
    y_predict_elasticnet = elasticnet.predict(X_test)
    elasticnet_loss = mean_squared_error(y_test, y_predict_elasticnet)
    print('ElasticNet Loss: ', elasticnet_loss)

    # Print the coefficients for the LASSO and Ridge models
    print('=' * 32)
    print('Coef LASSO')
    print(modelLasso.coef_)

    print('=' * 32)
    print('Coef RIDGE')
    print(modelRidge.coef_)

    print( "%.10f" % float(linear_loss))
    print( "%.10f" % float(lasso_loss))
    print( "%.10f" % float(ridge_loss))
    print( "%.10f" % float(elasticnet_loss))