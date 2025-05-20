# Retail_demad_Forecast-Time-Series-Project-
### Data of this project is download from -[kaggle](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)

### The Forecast app is in the website-[stream](https://asn-tables-asthma-older.trycloudflare.com/)

### Goal:
This project aims to forecast the unit sale in the retail by time-series analysis to model past sales parrtens and identify recurring trends and make informed predictions about future demand. 

### Descriptions:
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

### Skills:

Data cleaning and transformation (Pandas, NumPy)

Exploratory data analysis and descriptive statistics

Outlier detection and handling (z-score)

Time-series modeling (Exponential Smoothing, ARIMA, SARIMA, Prophet)

Machine learning for time-series (XGBoost with hyperparameter tuning, LSTM)

Data visualization (Matplotlib, Seaborn, Streamlit)

Forecast model evaluation (RMSE, RMAD, MAPE, R²)

Interactive web app development (Streamlit)

