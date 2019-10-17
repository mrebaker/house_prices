"""
Tools for tracking and predicting UK residential property values, based on data from Land Registry.
"""

import csv
from datetime import datetime as dt
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Prop:
    """
    A residential property in the UK, based on info both from Land Registry and privately held.
    """
    def __init__(self, prop_type, address):
        self.prop_type = prop_type
        self.address = address
        self.value_history = self.get_value_history()
        self.predicted_value = 0

    def get_value_history(self):
        """
        Get the history of known values for this property
        :return:
        """
        # TODO: either look up values on Land Registry, or search the database
        return []

    def predict_value(self):
        """
        Predicts the current value of a property based on the full Land Registry dataset.
        :param prop:
        :return:
        """

        # TODO: implement predictive model and return predicted value instead
        self.predicted_value = 0


def avg_value_by_month_and_type(month, prop_type):
    """
    Calculates average sale price for a given property type and month.
    :param month: 7 character string in format yyyy-mm
    :param prop_type: single letter corresponding to Land Registry property type (D, S, T, F, O)
    :return: average value as float
    """
    if prop_type not in 'DSTFO':
        print("Property type must be one of D, S, T, F and O.")
        return -1

    db_path = os.path.normpath("F:/Databases/hmlr_pp/hmlr_pp.db")
    conn = create_connection(db_path)
    cur = conn.cursor()
    criteria = (f'{month}%',)
    sales = cur.execute("""SELECT property_type, transaction_amount
                         FROM ppd 
                         WHERE transaction_date LIKE ?
                         ;""", criteria).fetchall()
    data = pd.DataFrame(sales, columns=['property_type', 'transaction_amount'])
    print(data.groupby('property_type').agg(['count', 'mean']))
    return data


def chart_sales():
    db_path = os.path.normpath("F:/Databases/hmlr_pp/hmlr_pp.db")
    conn = create_connection(db_path)
    cur = conn.cursor()
    sales = cur.execute("""SELECT transaction_amount, transaction_date 
                           FROM ppd 
                           WHERE property_type = 'D' AND town_city = 'BRISTOL' """).fetchall()
    x_vals, y_vals = [], []
    for sale in sales:
        y_vals.append(sale[0])
        x_vals.append(dt.strptime(sale[1], "%Y-%m-%d %H:%M"))

    plt.scatter(x_vals, y_vals)
    plt.show()


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)


def create_table(conn, create_table_sql):
    """
    Creates a table from the create_table_sql statement.
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)


def load_initial_data():
    """
    Takes Land Registry price data from a text file and creates a database from it.
    :return:
    """
    data_path = os.path.normpath("F:/Databases/hmlr_pp_complete.txt")
    db_path = os.path.normpath("F:/Databases/hmlr_pp/hmlr_pp.db")
    sql_create_ppd_table = """ CREATE TABLE IF NOT EXISTS ppd (
                                        id integer PRIMARY KEY,
                                        guid text NOT NULL,
                                        transaction_amount integer,
                                        transaction_date text,
                                        postcode text,
                                        property_type text,
                                        new_build text,
                                        tenure text,
                                        paon text,
                                        saon text,
                                        street text,
                                        locality text,
                                        town_city text,
                                        district text,
                                        county text,
                                        transaction_category text,
                                        record_status text
                                    ); """

    conn = create_connection(db_path)
    if conn is not None:
        create_table(conn, sql_create_ppd_table)
    else:
        print("Error: cannot create the database connection.")

    with open(data_path, 'r') as f:
        field_names = ['guid', 'transaction_amount', 'transaction_date', 'postcode',
                       'property_type', 'new_build', 'tenure', 'paon',
                       'saon', 'street', 'locality', 'town_city',
                       'district', 'county', 'transaction_category', 'record_status']
        dr = csv.DictReader(f, fieldnames=field_names)
        to_db = [tuple([row[field] for field in field_names]) for row in dr]
        print(to_db[0])

        cur = conn.cursor()
        insert_template = f"INSERT INTO ppd ({', '.join(field_names)}) VALUES ({', '.join(['?']*len(field_names))});"
        print(insert_template)
        cur.executemany(insert_template, to_db)
        conn.commit()
        conn.close()


def load_new_data():
    """
    Takes Land Registry price data for a single month and updates the database with it.
    :return:
    """


def select_rows(limit):
    db_path = os.path.normpath("F:/Databases/hmlr_pp/hmlr_pp.db")
    conn = create_connection(db_path)
    cur = conn.cursor()
    for row in cur.execute("""SELECT postcode, transaction_amount, transaction_date 
                              FROM ppd 
                              WHERE property_type = 'F' AND transaction_amount >= 1000000"""):
        print(row)


def validate_new_data():
    """
    Checks that the data to be uploaded does not already existing in the database.
    :return:
    """


if __name__ == '__main__':
    # load_initial_data()
    # select_rows(10)
    # chart_sales()
    avg_value_by_month_and_type("2019-08", 'D')