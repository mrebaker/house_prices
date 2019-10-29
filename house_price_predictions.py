import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date, datetime
from house_prices import run_query
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


def pr():
    """
    Polynomial regresison between time and property price.
    :return:
    """

    cols = ['transaction_date', 'transaction_amount']
    where = 'property_type = "F"'
    data = run_query(cols, where, 0)
    df = pd.DataFrame(data, columns=cols)

    df['transaction_date'] = df['transaction_date'].astype('datetime64[D]')
    d1 = datetime(1995, 1, 1)
    df['tx_day'] = (df['transaction_date'] - d1).dt.days
    # df['tx_day'] = df['tx_day'].astype('int')

    print(df.head())
    # high value filter
    df = df[df['transaction_amount'] < 10**6]

    lm = LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(df['tx_day'][:,None],
                                                        df['transaction_amount'][:,None],
                                                        random_state=0)

    lm.fit(X_train, y_train)
    print(f'yhat = {lm.intercept_[0]} + x*{lm.coef_[0][0]}')
    print(f'R^2 = {lm.score(X_train, y_train)}')

    # fig, ax = plt.subplots(figsize=(9, 7))
    # ax.scatter(df['transaction_date'].index.values,
    #            df['transaction_amount'])
    # plt.show()


def slr():
    """
    Basic single linear regression between time and property price.
    :return:
    """

    cols = ['transaction_date', 'transaction_amount']
    where = 'property_type = "F"'
    data = run_query(cols, where, 0)
    df = pd.DataFrame(data, columns=cols)

    df['transaction_date'] = df['transaction_date'].astype('datetime64[D]')
    d1 = datetime(1995, 1, 1)
    df['tx_day'] = (df['transaction_date'] - d1).dt.days
    # df['tx_day'] = df['tx_day'].astype('int')

    print(df.head())
    # high value filter
    df = df[df['transaction_amount'] < 10**6]

    lm = LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(df['tx_day'][:,None],
                                                        df['transaction_amount'][:,None],
                                                        random_state=0)

    lm.fit(X_train, y_train)
    print(f'yhat = {lm.intercept_[0]} + x*{lm.coef_[0][0]}')
    print(f'R^2 = {lm.score(X_train, y_train)}')

    # fig, ax = plt.subplots(figsize=(9, 7))
    # ax.scatter(df['transaction_date'].index.values,
    #            df['transaction_amount'])
    # plt.show()


if __name__ == '__main__':
    slr()
