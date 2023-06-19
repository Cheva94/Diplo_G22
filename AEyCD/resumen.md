
## Resumen del entregable
# Análisis Exploratorio y Curación de Datos

**Integrantes:**
- Chevallier-Boutell, Ignacio José
- Ribetto, Federico Daniel
- Rosa, Santiago
- Spano, Marcelo
==================================================

# Parte 1: Filtrado de datos

## Datasets

Partimos de:
1. El dataset de la [compentencia Kaggle](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot) sobre estimación de precios de ventas de propiedades en Melbourne, Australia, versión reducida reducida por [DanB](https://www.kaggle.com/dansbecker). El dataset está disponible [aquí](https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/melb_data.csv).
2. [Dataset](https://www.kaggle.com/tylerx/melbourne-airbnb-open-data?select=cleansed_listings_dec18.csv) de publicaciones de la plataforma AirBnB en Melbourne en el año 2018, realizado por [Tyler Xie](https://www.kaggle.com/tylerx).

--------------------------------------------------

## Conexiones SQL:
1. Dialecto SQLite, consultas a las bases de datos a través de instancias de conexión con bloque `with()` para evitar overhead, con entradas unsando `text()`.

--------------------------------------------------

## Combimnación de los datasets con SQL:
Se utilizaron los promedios asociados al código postal (CP) del precio diario, semanal y mensual del dataset de AirBnB. La combinación se hace considerando los CP.

--------------------------------------------------
## Selección de la información relevante

En el dataset de Kaggle hay 21 columnas, 13 numéricas y 8 categóricas.

Las numéricas son: _Distance_, _Lattitude_, _Longtitude_, _Propertycount_, _Postcode_, _Rooms_, _Price_, _Bedroom2_, _Bathroom_, _Car_, _Landsize_, _BuildingArea_ y _YearBuilt_.

Las categóricas son: _Suburb_, _Address_, _CouncilArea_ y _Regionname_, _Method_, _SellerG_, _Date_ y _Type_.

Se realizaron los siguientes filtros (0.0108 de los datos afuera), conservando los valores nulos para luego ser imputados:

* _Bedroom2_ < 20
* _Landsize_ < 3000
* _BuildingArea_ < 1500
* _YearBuilt_ < 1200
* _Price_ < 6000000

Las variables numéricas _Propertycount_ y _Car_ se descartaron por no tener correlación con el precio de la vivienda.

Las variables categóricas _Suburb_,  _Adress_ y  _Date_ se descartaron, ya que sus 10 registros mayoritarios no corresponden a un porcentaje considerable de entradas de la tabla.


 De las variables categóricas de mayor cardinalidad (_CouncilArea_ y _SellerG_), agrupamos todos los registros que contribuyen en menos del 2% en un nuevo registro "Other". Pasamos de tener 220 valores únicos para SellerG y 28 para CouncilArea, a tener 13 y 18 valores únicos, respectivamente.

Mediante un análisis de boxplot, nos quedamos con las variables _CouncilArea_, _Regionname_, _SellerG_, _Type_ y _Postcode_.

El último filtro aplicado es a la variable _zipcode_: nos quedamos con los valores que tengan una cantidad mayor o igual a 100 registros.

----------------------------------------------------------------------------------------------------

# Parte 2: Transformaciones e imputación de datos

Se aplicaron las siguientes operaciones:
1. One Hot encoding (implementación de [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)) a las variables _CouncilArea_, _Regionname_, _SellerG_ y _Type_, dado que no tienen un orden en sus categorías.

2. unimos las variables numéricas originales con las categóricas.

3. Aplicamos el método de imputación iterativa (implementación de [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html)), usando el método KNN de [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) como regresor, con `max_iter=50`, `tol=2e-1` y escaleo robusto, a las variables _YearBuilt_ y _BuildingArea_, que son las que más datos faltantes presentan.
