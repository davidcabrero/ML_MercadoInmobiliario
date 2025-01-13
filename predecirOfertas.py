import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv('houses.csv', usecols=['house_type', 'purchase_price', 'sqm', 'region', '%_change_between_offer_and_purchase'])
    # Limpiar y preprocesar datos
    df = df.dropna()
    df['accepted'] = (df['%_change_between_offer_and_purchase'] <= 0).astype(int)
    return df

df = load_data()

# Preprocesar datos
X = df[['house_type', 'purchase_price', 'sqm', 'region', '%_change_between_offer_and_purchase']]
y = df['accepted']
X = pd.get_dummies(X, columns=['house_type', 'region'], drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Crear la aplicación Streamlit
st.title('Predicción de Aceptación de Oferta de Casas')

# Entrada de datos del usuario
house_type = st.selectbox('Tipo de Casa', df['house_type'].unique())
purchase_price = st.number_input('Precio por el que quieres comprar', min_value=0.0)
offer_price = st.number_input('Precio original', min_value=0.0)
sqm = st.number_input('Metros Cuadrados', min_value=0.0)
region = st.selectbox('Región', df['region'].unique())

# Calcular el % de cambio entre oferta y compra solo si purchase_price no es cero
if purchase_price != 0:
    percent_change = ((offer_price - purchase_price) / purchase_price) * 100
else:
    percent_change = None
    st.error("El precio de compra no puede ser cero.")

# Preprocesar entrada del usuario
input_data = pd.DataFrame({
    'house_type': [house_type],
    'purchase_price': [purchase_price],
    'sqm': [sqm],
    'region': [region],
    '%_change_between_offer_and_purchase': [percent_change]
})

# Asegurar que las características de entrada coincidan con las del modelo
input_data = pd.get_dummies(input_data, columns=['house_type', 'region'], drop_first=True)
input_data = input_data.reindex(columns=X.columns, fill_value=0)
input_data_scaled = scaler.transform(input_data)

# Predicción
if st.button('Predecir') and percent_change is not None:
    prediction = model.predict(input_data_scaled)
    acceptance_probability = model.predict_proba(input_data_scaled)[0][1]
    st.write(f'Probabilidad de Aceptación: {acceptance_probability * 100:.2f}%')
    st.write('La oferta será aceptada' if prediction == 1 else 'La oferta no será aceptada')

    # Predicción de precios futuros (simulación simple)
    future_years = np.arange(1, 30 + 1)
    future_prices = purchase_price * (1 + 0.03) ** future_years  # Supongamos un aumento anual del 3%

    # Gráfico de predicciones futuras
    plt.figure(figsize=(10, 5))
    plt.plot(future_years, future_prices, label='Precio Futuro')
    plt.xlabel('Años en el Futuro')
    plt.ylabel('Precio de la Casa')
    plt.title('Predicción de Precio de la Casa en los Próximos 30 Años')
    plt.legend()
    st.pyplot(plt.gcf())