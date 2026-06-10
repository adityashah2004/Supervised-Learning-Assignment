import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from torch.fx.experimental.unification.multipledispatch.conflict import ordering


st.set_page_config(layout= "wide")

st.title("Time Series Analysis (Passengers)")

sel_df = st.sidebar.radio("Select Data",["Uploaded Data","Default Data"])

disp_data = st.sidebar.checkbox("View Original Data")

if sel_df == "Default Data":
    df = pd.read_csv("airline-passengers.csv", parse_dates=['Month'], index_col='Month')
    if disp_data:
        st.write(df)

elif sel_df == "Uploaded Data":

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file,parse_dates=["Month"],index_col='Month')

        st.success("File Uploaded Successfully")

    if disp_data:
        st.write(df)

ar = st.sidebar.number_input("AR",1,5,2,1)
i = st.sidebar.number_input("I",0,2,0,1)
ma = st.sidebar.number_input("MA",1,5,2,1)

model = ARIMA(df, order=(ar,i,ma))
b10 = st.sidebar.checkbox("SARIMA")
if b10:
    model = SARIMAX(df, order=(ar,i,ma), seasonal_order=(1,1,1,12))

new_model = model.fit()
btn_sum = st.sidebar.button("Summary")

if btn_sum:
    st.write(new_model.summary())

forecast_chk = st.sidebar.checkbox("Forecast")

if forecast_chk:
    total_num = st.number_input("Enter number",1,50,5,1)

    btn_predict = st.button("Predict")

    if btn_predict:

        forecast = new_model.forecast(steps=total_num)

        st.subheader("Forecasted Values")
        st.write(forecast)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(df.index, df['Passengers'], label='Actual')
        ax.plot(forecast.index, forecast, label='Forecast')

        # ax.set_title("Actual vs Forecast - Airline Passengers")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)



