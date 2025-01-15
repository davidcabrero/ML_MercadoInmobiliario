import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Cargar los datos
# Asegúrate de ajustar la ruta del archivo a tu archivo de datos
df_houses = pd.read_csv('houses.csv')

# Configurar el título de la aplicación
st.title('Recomendador de Casas')

# Crear los filtros de entrada
precio_max = st.slider('Precio máximo', int(df_houses['purchase_price'].min()), int(df_houses['purchase_price'].max()), int(df_houses['purchase_price'].max()))
ciudad = st.selectbox('Ciudad', df_houses['region'].unique())
habitaciones = st.slider('Número de habitaciones', int(df_houses['no_rooms'].min()), int(df_houses['no_rooms'].max()), int(df_houses['no_rooms'].min()))
tipo_casa = st.selectbox('Tipo de casa', df_houses['house_type'].unique())
anio_construccion = st.slider('Año de construcción', int(df_houses['year_build'].min()), int(df_houses['year_build'].max()), int(df_houses['year_build'].min()))
metros_cuadrados = st.slider('Metros cuadrados', int(df_houses['sqm'].min()), int(df_houses['sqm'].max()), int(df_houses['sqm'].min()))

# Filtrar los datos según los filtros de entrada
filtered_houses = df_houses[
    (df_houses['purchase_price'] <= precio_max) &
    (df_houses['region'] == ciudad) &
    (df_houses['no_rooms'] >= habitaciones) &
    (df_houses['house_type'] == tipo_casa) &
    (df_houses['year_build'] >= anio_construccion) &
    (df_houses['sqm'] >= metros_cuadrados)
]

# Si hay casas que cumplen con los filtros, usar sklearn para predecir las casas más recomendables
if not filtered_houses.empty:
    # Seleccionar las características relevantes
    features = ['purchase_price', 'no_rooms', 'year_build', 'sqm']
    X = df_houses[features]
    
    # Crear el modelo de vecinos más cercanos
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X)
    
    # Encontrar las casas más recomendables con knn
    user_input = [[precio_max, habitaciones, anio_construccion, metros_cuadrados]]
    distances, indices = knn.kneighbors(user_input)
    
    # Mostrar las casas recomendadas
    st.write('Casas recomendadas:')
    st.write(df_houses.iloc[indices[0]])
else:
    st.write('No se encontraron casas que cumplan con los filtros seleccionados.')