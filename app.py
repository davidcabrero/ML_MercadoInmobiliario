import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv('houses.csv')

# Título de la aplicación
st.title('Predicción de características de casas')

# Selección de la característica a predecir
features = ['sqm', 'sqm_price', 'no_rooms', 'purchase_price']
feature = st.selectbox('Seleccione la característica a predecir', features)

# Selección del año futuro
year = st.number_input('Ingrese el año futuro para la predicción', min_value=2025, step=1)

# Opcional: Selección de región, tipo de casa y tipo de venta
region_options = ['Ninguno'] + list(data['region'].unique())
house_type_options = ['Ninguno'] + list(data['house_type'].unique())
sales_type_options = ['Ninguno'] + list(data['sales_type'].unique())

region = st.selectbox('Seleccione la región', region_options)
house_type = st.selectbox('Seleccione el tipo de casa', house_type_options)
sales_type = st.selectbox('Seleccione el tipo de venta', sales_type_options)

# Filtrar datos según las selecciones opcionales
filtered_data = data.copy()
if region != 'Ninguno':
    filtered_data = filtered_data[filtered_data['region'] == region]
if house_type != 'Ninguno':
    filtered_data = filtered_data[filtered_data['house_type'] == house_type]
if sales_type != 'Ninguno':
    filtered_data = filtered_data[filtered_data['sales_type'] == sales_type]

# Mostrar datos
st.write('Datos de casas filtrados:')
st.write(filtered_data)

# Preparar datos para el modelo
X = np.array(filtered_data['year_build']).reshape(-1, 1)
y = filtered_data[feature]

# Entrenar el modelo
model = LinearRegression()
model.fit(X, y)

# Realizar predicción
future_year = np.array([[year]])
prediction = model.predict(future_year)

# Mostrar predicción
st.write(f'Predicción para el año {year}: {prediction[0]}')

# Gráfico de la predicción
plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['year_build'], filtered_data[feature], color='blue', label='Datos reales')
plt.plot(filtered_data['year_build'], model.predict(X), color='red', label='Modelo lineal')
plt.scatter(future_year, prediction, color='green', label='Predicción futura')

# Título del gráfico
title = f'Predicción de {feature} en el año {year}'
if region != 'Ninguno':
    title += f' en la región {region}'
if house_type != 'Ninguno':
    title += f', tipo de casa {house_type}'
if sales_type != 'Ninguno':
    title += f', tipo de venta {sales_type}'

plt.xlabel('Año')
plt.ylabel(feature)
plt.title(title)
plt.legend()
st.pyplot(plt)