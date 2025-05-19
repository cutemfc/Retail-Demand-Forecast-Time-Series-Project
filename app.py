import sys
import os
import streamlit as st
from data.config import DATA_PATH, MODEL_PATH
from data.data_utils import load_data, prepare_training_data, create_input_row
from models.model_utils import load_model, predict
import datetime  # Used for handling date inputs
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from utils import run_visualization


@st.cache_resource  # Only read the data once, and repeat to use it
def cache_load_data():
    df_store, df_item, df_train = load_data(DATA_PATH)
    df_train_processed = prepare_training_data(df_train)
    return df_store, df_item, df_train, df_train_processed
@st.cache_resource
def cache_load_model():
    return load_model(MODEL_PATH)

def run_app():
    st.title("Corporación Favorita Sales Forecasting")
    # load data and model from cache
    df_store, df_item, df_train, df_train_processed = cache_load_data()
    model=cache_load_model()

    # Store selection
    store_id = st.selectbox("Store", [24, 26, 27, 28])  # For testing limit to one store
    item_id = st.selectbox("Item", [564533, 838216, 582865, 364606])  # For testing limit to a few items


    # Set default and allowed date range for forecasting
    default_date = datetime.date(2014, 1, 1)  # Default date is Jan 1, 2014
    min_date = datetime.date(2013, 1, 2)  # Minimum date allowed is January 2, 2013
    max_date = datetime.date(2014, 4, 1)  # Maximum date allowed is April 1, 2014

    # Date input for selecting forecast date, within the range [min_date, max_date]
    date = st.date_input("Forecast Date", value=default_date, min_value=min_date, max_value=max_date)

    # When the user clicks the "Get Forecast" button
    if st.button("Get Forecast"):
        hist_data = df_train_processed[
        (df_train_processed['store_nbr'] == store_id) &
        (df_train_processed['item_nbr'] == item_id) &
        (df_train_processed['date'] < pd.to_datetime(date))
        ]
        st.write(f"Number of historical records: {len(hist_data)}")

        if hist_data.empty:
            st.warning("No historical data found for selection")
        else:
            input_data = create_input_row(store_id, item_id, date, df_train_processed)
            prediction = predict(model, input_data.to_frame().T)
            st.write(f"Predicted Sales for {date}: {prediction[0]}")

    # Visualization
    split_date=st.sidebar.date_input("Forecast Start Date",value=default_date,min_value=min_date, max_value=max_date)
    run_visualization(df_train, model, split_date,store_filter=store_id, item_filter=item_id)

# Ensure the script runs the main function if executed directly
if __name__ == "__main__":
    run_app()
