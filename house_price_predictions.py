from datetime import date, datetime
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


import house_prices as hp


def create_model(model_type):
    """
    Generates new regression model of the specified type.
    :param model_type:
    :return:
    """

    # TODO add more models
    supported_models = ['slr']
    if model_type not in supported_models:
        print(f'{model_type} is not a supported type of model.')
        return -2

    # TODO model data is currently hardcoded - this should be made flexible
    cols = ['transaction_date', 'transaction_amount']
    where = 'property_type = "F"'
    data = hp.run_query(cols, where, 0)
    df = pd.DataFrame(data, columns=cols)

    df['transaction_date'] = df['transaction_date'].astype('datetime64[D]')
    d1 = datetime(1995, 1, 1)
    df['tx_day'] = (df['transaction_date'] - d1).dt.days

    # high value filter
    df = df[df['transaction_amount'] < 10 ** 6]
    X = df['tx_day'][:, None],
    y = df['transaction_amount'][:, None]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.30,
                                                        random_state=0)

    lm = LinearRegression()
    lm.fit(X_train, y_train)

    print(f"""New model created:
              yhat = {lm.intercept_[0]} + x*{lm.coef_[0][0]}')
              R^2 = {lm.score(X_test, y_test)}""")
    pickled_model = pickle.dumps(lm)
    insert_query = """INSERT INTO models (type, date_created, model_obj) 
                      VALUES (?, ?, ?)"""

    conn = hp.create_connection(DB_PATH)
    cur = conn.cursor()
    cur.executemany(insert_template, to_db)
    conn.commit()
    conn.close()
    return lm


def fetch_model(model_type):
    """
    Retrieve a pickled model from the database; returns None if it does not exit.
    :param model_type:
    :return:
    """


def run_model(model_type):
    """
    Wrapper for creating and running models.
    :param model_type:
    :return:
    """
    # TODO


def run_pr():
    """
    Polynomial regression between time and property price.
    :return:
    """

    cols = ['transaction_date', 'transaction_amount']
    where = 'property_type = "F"'
    data = hp.run_query(cols, where, 0)
    df = pd.DataFrame(data, columns=cols)

    df['transaction_date'] = df['transaction_date'].astype('datetime64[D]')
    d1 = datetime(1995, 1, 1)
    df['tx_day'] = (df['transaction_date'] - d1).dt.days

    # high value filter
    df = df[df['transaction_amount'] < 10**6]

    X_train, X_test, y_train, y_test = train_test_split(df['tx_day'][:, None],
                                                        df['transaction_amount'][:, None],
                                                        test_size=0.30,
                                                        random_state=0)

    for order in range(2, 6):
        lm = LinearRegression()
        pr = PolynomialFeatures(degree=order)
        X_train_pr = pr.fit_transform(X_train)
        X_test_pr = pr.fit_transform(X_test)
        lm.fit(X_train_pr, y_train)
        print(f'For order {order}, R^2 = {lm.score(X_test_pr, y_test):.5f}')

    # fig, ax = plt.subplots(figsize=(9, 7))
    # ax.scatter(df['transaction_date'].index.values,
    #            df['transaction_amount'])
    # plt.show()


def run_slr(force_new):
    """
    Basic single linear regression between time and property price.
    :return:
    """

    if force_new:
        lm = create_model('slr', )
    else:
        lm = fetch_model('slr')
        if lm is None:
            slr_model = create_model('slr')


if __name__ == '__main__':
    # pca_list = hp.get_postcode_areas()
    run_slr(False)
    run_pr()
