import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
def load_data(DATA_PATH):
    df_store = pd.read_csv(os.path.join(DATA_PATH, 'stores.csv'))
    df_item = pd.read_csv(os.path.join(DATA_PATH, 'items.csv'))
    df_train = pd.read_csv(os.path.join(DATA_PATH, 'df_train_revised.csv')) # Only selected stores in Guyas, Top 3 Families
    return df_store, df_item, df_train

def prepare_training_data(df_train):
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_train.set_index('date', inplace=True)
    # Lag features
    df_train['lag_1'] = df_train['unit_sales'].shift(1)
    df_train['lag_7'] = df_train['unit_sales'].shift(7)
    df_train['lag_30'] = df_train['unit_sales'].shift(30)
    # Rolling features
    df_train['rolling_mean_7'] = df_train['unit_sales'].rolling(window=7).mean()
    df_train['rolling_std_7'] = df_train['unit_sales'].rolling(window=7).std()
    df_train.dropna(inplace=True)
    # Time-based features
    df_train['year'] = df_train.index.year
    df_train['month'] = df_train.index.month
    df_train['day_of_week'] = df_train.index.dayofweek
    df_train['is_weekend'] = df_train['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    # Outlier detection and replacement
    z_scores = zscore(df_train['unit_sales'])
    outliers = df_train[z_scores > 5]
    df_train.loc[outliers.index, 'unit_sales'] = df_train.loc[outliers.index, 'rolling_mean_7']
    return df_train.reset_index()  # Restore 'date' as a column

def create_input_row(store_id, item_id, date, df_train):
    # Based on the input item/store
    date = pd.to_datetime(date)
    relevant_data = df_train[
        (df_train['store_nbr'] == store_id) & (df_train['item_nbr'] == item_id) & (df_train['date'] < date)
    ]

    if relevant_data.empty:
        raise ValueError("No historical data for this item/store combination.")




    #lag features
    lag_1 = relevant_data.iloc[-1]['unit_sales']
    lag_7 = relevant_data.iloc[-7]['unit_sales']if len(relevant_data) >= 7 else lag_1
    lag_30 = relevant_data.iloc[-30]['unit_sales'] if len(relevant_data) >= 30 else lag_7
    rolling_data=relevant_data.iloc[-7:]['unit_sales']
    rolling_mean_7 = relevant_data.iloc[-7:]['unit_sales'].mean() if len(rolling_data) == 7 else lag_1
    rolling_std_7 = relevant_data.iloc[-7:]['unit_sales'].std() if len(rolling_data) == 7 else 0.0
    latest_row = relevant_data.iloc[-1].copy()
    input_data = latest_row.copy()

    input_data['lag_1'] = lag_1
    input_data['lag_7'] = lag_7
    input_data['lag_30'] = lag_30
    input_data['rolling_mean_7'] = rolling_mean_7
    input_data['rolling_std_7'] = rolling_std_7

    input_data['date'] = date
    input_data['year'] = date.year
    input_data['month'] = date.month
    input_data['day_of_week'] = date.dayofweek
    input_data['is_weekend'] = 1 if date.dayofweek >= 5 else 0
    input_features = input_data[['store_nbr', 'item_nbr', 'id', 'onpromotion',
                                 'lag_1', 'lag_7', 'lag_30',
                                 'rolling_mean_7', 'rolling_std_7',
                                 'year', 'month', 'day_of_week', 'is_weekend']]

    st.write("\n=== Input features  ===")
    st.write(input_features)
    return input_features.astype(float)

