"""
Tools for tracking and predicting UK residential property values, based on data from Land Registry.
"""

import csv
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DB_PATH = os.path.normpath("F:/Databases/hmlr_pp/hmlr_pp.db") 


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
    Calculates geometric mean of sale price for a given property type and month.
    :param month: 7 character string in format yyyy-mm
    :param prop_type: single letter corresponding to Land Registry property type (D, S, T, F, O)
    :return: average value as float
    """
    if prop_type not in 'DSTFO':
        print("Property type must be one of D, S, T, F and O.")
        return -1

    conn = create_connection()
    cur = conn.cursor()
    criteria = (f'{month}%', prop_type)
    sales = cur.execute("""SELECT transaction_amount
                         FROM ppd 
                         WHERE transaction_date LIKE ?
                         AND property_type = ?
                         ;""", criteria).fetchall()

    g_mean = stats.gmean(sales)
    return g_mean


def avg_price_history():
    conn = create_connection()
    cur = conn.cursor()

    dates = cur.execute("""SELECT transaction_date 
                           FROM ppd  
                           ORDER BY transaction_date asc""").fetchall()

    min_date = dt.strptime(dates[0][0], "%Y-%m-%d %H:%M").replace(day=1)
    max_date = dt.strptime(dates[-1][0], "%Y-%m-%d %H:%M").replace(day=1)

    date = min_date
    month_list = []
    while date <= max_date:
        month_list.append(date.strftime("%Y-%m"))
        date = date + relativedelta(months=1)

    sales = cur.execute("""SELECT transaction_date, property_type, transaction_amount
                             FROM ppd;""").fetchall()
    df_sales = pd.DataFrame(sales, columns=['transaction_date', 'property_type', 'transaction_amount'])

    history = {}
    for month in month_list:
        avg_prices = {}
        df_sales_month = df_sales[df_sales['transaction_date'].str.contains(month)]
        for prop_type in 'DSTF':
            df_values = df_sales_month[df_sales_month['property_type'] == prop_type]
            g_mean = stats.gmean(df_values['transaction_amount'])
            avg_prices[prop_type] = g_mean
        history[month] = avg_prices

    with open('avg_price_history.csv', 'w+') as f:
        # field_names = ['month', 'property_type', 'average_price']
        writer = csv.writer(f)
        for k, v in history.items():
            row = [k, v['D'], v['S'], v['T'], v['F']]
            print(row)
            writer.writerow(row)


def chart_sales():
    conn = create_connection()
    cur = conn.cursor()
    sales = cur.execute("""SELECT transaction_amount, transaction_date 
                           FROM ppd 
                           WHERE property_type = 'D' 
                           AND town_city = 'BRISTOL' """).fetchall()
    x_vals, y_vals = [], []
    for sale in sales:
        y_vals.append(sale[0])
        x_vals.append(dt.strptime(sale[1], "%Y-%m-%d %H:%M"))

    plt.scatter(x_vals, y_vals)
    plt.show()


def chart_avg_prices():
    with open('avg_price_history.csv', 'r') as f:
        df = pd.read_csv(f, names=['month', 'D', 'S', 'T', 'F'])

    df['month'] += '-01'
    ax1 = df.plot()
    tick_count = len(ax1.get_xticklabels())
    month_count = len(df['month'])
    label_spacing = month_count // tick_count
    new_ticks = []
    for x in range(tick_count):
        new_ticks.append(df['month'][x * label_spacing])
    print(new_ticks)
    ax1.set_xticklabels(new_ticks)
    plt.show()


def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
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


def create_property_id_table():
    create_table_query = """CREATE TABLE IF NOT EXISTS address (
                      id integer PRIMARY KEY,
                      postcode text,
                      paon text, 
                      saon text, 
                      street text, 
                      locality text, 
                      town_city text, 
                      district text, 
                      county text
                      );"""

    insert_query = """INSERT INTO address
                      SELECT NULL, postcode, paon, saon, street, locality, town_city, district, county
                      FROM ppd
                      GROUP BY postcode, paon, saon, street, locality, town_city, district, county; """

    conn = create_connection()

    if conn is not None:
        create_table(conn, create_table_query)
    else:
        print("Error: cannot create the database connection.")

    cur = conn.cursor()
    cur.execute(insert_query).fetchall()
    conn.commit()
    conn.close()


def get_postcode_areas():
    conn = create_connection()
    cur = conn.cursor()
    query = f"""SELECT DISTINCT postcode
                FROM ppd"""

    pc_list = cur.execute(query).fetchall()
    pca_list = []


def load_initial_data():
    """
    Takes Land Registry price data from a text file and creates a database from it.
    :return:
    """
    data_path = os.path.normpath("F:/Databases/hmlr_pp_complete.txt")
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
                                        record_status text,
                                        fk_address_id integer
                                    ); """

    conn = create_connection()
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
        insert_template = f"""INSERT INTO ppd ({', '.join(field_names)})
                              VALUES ({', '.join(['?']*len(field_names))});"""
        print(insert_template)
        cur.executemany(insert_template, to_db)
        conn.commit()
        conn.close()


def load_new_data():
    """
    Takes Land Registry price data for a single month and updates the database with it.
    :return:
    """
    # TODO


def load_oapcs_data():
    """
    Loads the output area to postcode sector mappings into the database.
    :return:
    """
    data_path = os.path.normpath("F:/Databases/hmlr_pp/OC_PCS_2011_EW.csv")
    sql_create_ppd_table = """ CREATE TABLE IF NOT EXISTS oapcs (
                                                id integer PRIMARY KEY,
                                                OA11CD text,
                                                PCDS11CD text,
                                                ObjectId integer
                                            ); """

    conn = create_connection()
    if conn is not None:
        create_table(conn, sql_create_ppd_table)
    else:
        print("Error: cannot create the database connection.")

    with open(data_path, 'r', encoding='utf-8-sig') as f:
        field_names = ['OA11CD', 'PCDS11CD', 'ObjectId']
        dr = csv.DictReader(f)
        to_db = [tuple([row[field] for field in field_names]) for row in dr]
        print(to_db[0])

        cur = conn.cursor()
        insert_template = f"""INSERT INTO oapcs ({', '.join(field_names)})
                                      VALUES ({', '.join(['?']*len(field_names))});"""
        print(insert_template)
        cur.executemany(insert_template, to_db)
        conn.commit()
        conn.close()


def load_ruc_data():
    """
    Loads the rural-urban classification data into the database.
    :return:
    """
    data_path = os.path.normpath("F:/Databases/hmlr_pp/RUC11_OA11_EW.csv")
    sql_create_ppd_table = """ CREATE TABLE IF NOT EXISTS ruc (
                                            id integer PRIMARY KEY,
                                            OA11CD text,
                                            RUC11CD text,
                                            RUC11 text,
                                            BOUND_CHGIND text,
                                            ASSIGN_CHGIND text,
                                            ASSIGN_CHREASON text
                                        ); """

    conn = create_connection()
    if conn is not None:
        create_table(conn, sql_create_ppd_table)
    else:
        print("Error: cannot create the database connection.")

    with open(data_path, 'r') as f:
        field_names = ['OA11CD', 'RUC11CD', 'RUC11', 'BOUND_CHGIND', 'ASSIGN_CHGIND', 'ASSIGN_CHREASON']
        dr = csv.DictReader(f)
        to_db = [tuple([row[field] for field in field_names]) for row in dr]
        print(to_db[0])

        insert_template = f"""INSERT INTO ruc ({', '.join(field_names)})
                                  VALUES ({', '.join(['?']*len(field_names))});"""
        print(insert_template)
        cur = conn.cursor()
        cur.executemany(insert_template, to_db)
        conn.commit()
        conn.close()


def run_query(columns, where, limit):
    """
    Helper to execute queries on the database - intended for use by the predictions model.
    :return:
    """
    conn = create_connection()
    cur = conn.cursor()
    query = f"""SELECT {','.join(columns)}
                FROM ppd
                WHERE {where}"""

    if limit == 0:
        data = cur.execute(query).fetchall()
    else:
        data = cur.execute(query).fetchmany(limit)

    return data


def select_rows(tbl):
    """
    Executes hardcoded SQL queries for testing purposes.
    :param tbl: a single letter indicating the table to query
    :return: -1 on fail, 0 otherwise
    """
    conn = create_connection()
    cur = conn.cursor()

    if tbl == 'a':
        query = """SELECT * 
                   FROM address;"""
    elif tbl == 'p':
        query = """SELECT postcode, transaction_amount, transaction_date 
                   FROM ppd 
                   WHERE property_type = 'F' AND transaction_amount >= 1000000"""
    else:
        return -1

    rows = cur.execute(query).fetchall()
    for row in rows:
        print(row)


def set_address_id_in_ppd():
    select_query = """SELECT address.id, ppd.id
                      FROM ppd 
                      LEFT JOIN address on (ppd.postcode = address.postcode and
                      ppd.paon = address.paon and
                      ppd.saon = address.saon and
                      ppd.street = address.street and
                      ppd.locality = address.locality and
                      ppd.town_city = address.town_city and
                      ppd.district = address.district and
                      ppd.county = address.county);"""
    conn = create_connection()
    cur = conn.cursor()
    rows = cur.execute(select_query).fetchall()

    update_template = """UPDATE ppd SET fk_address_id = ?
                         WHERE id = ?;"""
    for row in rows:
        cur.execute(update_template, row)

    conn.commit()
    conn.close()


def validate_new_data():
    """
    Checks that the data to be uploaded does not already existing in the database.
    :return:
    """


if __name__ == '__main__':
    # load_initial_data()
    load_oapcs_data()
    # create_property_id_table()
    # select_rows('p')
    # chart_sales()
    # avg_value_by_month_and_type("2019-08", 'D')
    # avg_price_history()
    # chart_avg_prices()
    # set_address_id_in_ppd()
