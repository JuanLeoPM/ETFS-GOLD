import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="Gold ETF Price Predictor", layout="wide")
st.title("📈 Gold ETF Price Prediction App")

# Entrada del usuario
ticker = st.text_input("Enter Gold ETF Ticker (e.g., GLD, IAU):", value="GLD")
start_date = st.date_input("Start Date", value=datetime(2010, 1, 1))
end_date = st.date_input("End Date", value=datetime.today())

# Descargar datos
@st.cache_data
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        df = df[['Close']].dropna()
        df.columns = ['Price']
        return df
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()


if st.button("Train Model and Predict"):
    df = load_data(ticker, start_date, end_date)

    # Crear características retardadas
    for i in range(1, 6):
        df[f'lag_{i}'] = df['Price'].shift(i)
    df.dropna(inplace=True)

    # Dividir en train y test
    X = df[[f'lag_{i}' for i in range(1, 6)]]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Modelo
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.write(f"Root Mean Squared Error: {rmse:.2f}")

    # Graficar
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pred_df['Actual'], label='Actual')
    ax.plot(pred_df['Predicted'], label='Predicted')
    ax.set_title("Gold ETF Price Prediction")
    ax.legend()
    st.pyplot(fig)

    # Mostrar último precio y predicción siguiente día
    last_row = df.iloc[-5:][['Price']]
    last_features = last_row[::-1]['Price'].values.reshape(1, -1)
    next_day_pred = model.predict(last_features)[0]
    st.success(f"📍 Next Day Price Prediction: {next_day_pred:.2f} USD")

