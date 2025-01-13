# Predicciones en el Mercado Inmobiliario de Dinamarca 🏠

Este proyecto utiliza técnicas de Machine Learning para predecir los precios de las casas en dinamarca durante los próximos años como también ventas o las características de las mismas según diferentes regiones.

## Descripción

El objetivo de este proyecto es construir un modelo de Machine Learning que pueda predecir el precio de una casa basándose en diversas características como el tamaño, la ubicación, el número de habitaciones, etc. y así poder aumentar las ventas en empresas constructoras, llevando a cada región de dinamarca y a cada mercado la construcción más necesaria, como a su vez predecir la evolución de los precios en diferentes mercados de la construcción.

## Estructura del Proyecto

- `EDAyPredicciones.ipynb`: Análisis exploratorio de los datos de ventas de casas y características en varios años y predicciones con Machine Learning de años futuros.
- `predecirOfertas.py`: Utiliza Machine Learning para predecir qué oferta será buena según características de la casa.
- `RecomendadorCasas.py`: Elige la vivienda ideal para cada usuario según sus intereses.
- `app.py`: Aplicación para ver las predicciones de forma visual.
- `houses.csv`: Dataset con todos los datos de ventas de casas desde 1600 a 2024.

## Instalación

1. Clona el repositorio:
    ```sh
    git clone https://github.com/davidCabrero/ML_MercadoInmobiliario.git
    ```
2. Instala los requisitos:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

1. Ejecuta la interfaz de predicciones:
    ```sh
    streamlit run app.py
    ```
2. Ejecuta el recomendador de casas:
    ```sh
    streamlit run recomendadorCasas.py
    ```

## Contacto

Para cualquier consulta, por favor contacta a [davidcabrerojimenez@gmail.com](mailto:davidcabrerojimenez@gmail.com).
