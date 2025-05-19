import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score

def run_visualization(df_train, model, split_date='2014-01-01', max_plots=1, store_filter=None, item_filter=None):
    split_date = pd.to_datetime(split_date)  # Ensure datetime
    rmad_values = []
    bias_values = []
    rmse_values = []
    mape_values = []
    r2_values = []
    plot_count = 0

    # Check for required inputs
    if store_filter is None or item_filter is None:
        st.warning("Please provide both store and item filter.")
        return

    # Filter the target group
    target_group = df_train[(df_train['store_nbr'] == store_filter) & (df_train['item_nbr'] == item_filter)]
    if target_group.empty:
        st.warning("No data available for the selected item/store combination.")
        return

    target_group = target_group.reset_index()
    target_group['date'] = pd.to_datetime(target_group['date'])

    test_series = target_group[target_group['date'] >= split_date]
    train_series = target_group[target_group['date'] < split_date]

    if len(test_series) <= 5:
        st.warning("Not enough test data for visualization.")
        return

    X_test = test_series.drop(['unit_sales', 'date'], axis=1)
    y_test = test_series['unit_sales']
    y_pred = model.predict(X_test)

    if plot_count < max_plots:
        plt.figure(figsize=(12, 6))
        plt.plot(train_series['date'], train_series['unit_sales'], label='Train Sales', color='black')
        plt.plot(test_series['date'], y_test, label='Actual Sales', color='blue')
        plt.plot(test_series['date'], y_pred, label='Predicted Sales', color='red')
        plt.title(f'Store {store_filter}, Item {item_filter}', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Unit Sales')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()
        plot_count += 1

    # Calculate the metrics
    bias = np.mean(y_pred - y_test)
    rmad = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_pred - y_test) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    # Show the data
    st.markdown(f"### Store {store_filter}, Item {item_filter}")
    st.write(
        f"**Bias:** {bias:.2f} &nbsp;&nbsp; "
        f"**RMAD:** {rmad:.2f} &nbsp;&nbsp; "
        f"**RMSE:** {rmse:.2f} &nbsp;&nbsp; "
        f"**MAPE:** {mape:.2f}% &nbsp;&nbsp; "
        f"**R²:** {r2:.2f}"
    )
