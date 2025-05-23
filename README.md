# 🎇Retail Demand Forecast (Time Series Project)
### Data of this project is download from -[kaggle](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)


### 🌍 Goal:
This project aims to forecast retail unit sale using time-series analysis, modelling past sales patterns to identify recurring trends and generate informed predictions about future demand. 

### 🌍Descriptions:
The project involves forecasting unit sales in the Guayas region for the period from January 1 to March 31, 2014, using various time series models including Exponential Smoothing, ARIMA, SARIMA, XGBoost (with hyperparameter tuning), LSTM, and the Prophet model.

Additionally, a forecast application was developed using Streamlit, allowing users to input the store number, item number, and prediction date, and view the forecast results, visualizations, and evaluation metrics.

Part I: Data Exploration with Python

Loaded and preprocessed the data, visualized the sales volume of the top three product families, and handled outliers.
Analyzed the impact of holidays and perishable items on sales performance.


Part II: Modelling & Visualization with Python

Built and evaluated models using classical time series approaches including Exponential Smoothing, ARIMA, SARIMA, as well as machine learning models such as XGBoost (with hyperparameter tuning), LSTM, and Prophet.
Among these, XGBoost performed the best, achieving a Mean Absolute Percentage Error (MAPE) of approximately 10%. The most important feature identified was the lag-1 sales value.

Part III: Created a Streamlit app for interactive forecasting

Developed a retail sales forecast application using Streamlit, powered by an XGBoost model.
Users can input the store number, item number, and desired forecast date. The app predicts unit sales and provides visualizations of historical and forecasted data.
The model's performance is evaluated using multiple metrics, including RMSE (Root Mean Square Error), RMAD (Root Mean Absolute Deviation), MAPE (Mean Absolute Percentage Error), and R² (Coefficient of Determination).

![App Screenshot](https://github.com/user-attachments/assets/ca685c74-00a5-43a0-a0f2-49f2b1c8e5ba)

### 🌍Skills:

1.Data cleaning and transformation (Pandas, NumPy)

2.Exploratory data analysis and descriptive statistics

3.Outlier detection and handling (z-score)

4.Time-series modeling (Exponential Smoothing, ARIMA, SARIMA, Prophet)

5.Machine learning for time-series (XGBoost with hyperparameter tuning, LSTM)

6.Data visualization (Matplotlib, Seaborn, Streamlit)

7.Forecast model evaluation (RMSE, RMAD, MAPE, R²)

8.Interactive web app development (Streamlit)


### Insight:
1.Exploratory data analysis revealed that holidays and the perishability of items significantly impact unit sales.

2.The naive model served as a baseline, while XGBoost achieved the best performance with a mean absolute percentage error (MAPE) of 9.49% and an R² score of 0.59.

3.Hyperparameter tuning further improved the predictive accuracy of the XGBoost model.

4.A Streamlit app was developed to provide an interactive interface for forecasting future sales and anticipating customer demand.


### Presentation
[Presentation](https://youtu.be/zcPRyP_dtSE)


