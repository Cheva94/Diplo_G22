# Diplomatura en ciencia de datos, aprendizaje automático y sus aplicaciones - Edición 2023 - FAMAF (UNC)

## Resolución del entregable de **aprendizaje supervisado**

### **Grupo 22:**

**Integrantes:**
- [Chevallier-Boutell, Ignacio José](https://github.com/Cheva94)
- [Ribetto, Federico Daniel](https://github.com/federibet)
- [Rosa, Santiago](https://github.com/santrosa)
- [Spano, Marcelo](https://github.com/marcespano)

**Seguimiento:** Meinardi, Vanesa

---

## Introducción

En este entregable se pretende participar de una competencia de Kaggle, donde el dataset es una recopilación de datos méedicos y demográficos de pacientes junto a su estado de diabetes.

El dataset consta de un total de 100000 registros: 95000 se utilizan en la etapa de entrenamiento, mientras que los otros 5000 son para la evaluación del modelo. Cada registro tiene 10 variables: además del ID del paciente y su estado de diabetes, hay 2 variables categóricas (sexo e historial de fumador) y 6 variables numéricas (edad, hipertensión, enfermedades cardíacas, índice de masa corporal, hemoglobina glicosilada y glucemia).

El objetivo es construir modelos que predigan si una persona tiene o no diabetes. Teniendo como referencia un árbol de decisión cuya exactitud es del 0.97266, al menos uno de nuestros modelos debe superar dicho valor.

## Exploración 

### Datos faltantes

Salvo las etiquetas que fueron eliminadas a propósito en el conjunto de datos de evaluación, no se tienen valores faltantes.

### Variables categóricas

Se puede observar que cada variable categórica posee pocas categorías. Además se observa que son variables nominales.

La variable del `sexo` tiene 3 opciones posibles: varón, mujer y otro. La variable `historial de fumador` tiene 6 opciones posibles: sin información, nunca, fumaba, fuma actualmente, no fuma actualmente y siempre.

Respecto a `sexo` hay casi un 60% de mujeres y un 40% de varones, por lo que no hay un gran desbalance entre estas dos categorías. Sí está extremadamente desbalanceado con otros: hay mucho menos que el 1%.

![](sexo.png)

Respecto a `historial de fumador` las dos categorías más preponderantes con un 35% cada una son nunca y sin información, llevándose prácticamente el 70% de los datos. Las otras cuatro categorías ocupan el 30% restante de una manera más o menos equitativa.

![](fumador.png)

### Variables numéricas

En ningún caso se observan outliers significativos, por lo que se decide no realizar ninguna eliminación ni imputación.

![](boxplot.png)

Las variables `hipertensión` y `enfermedad cardíaca` son en realidad respuestas binarias: 0 es que no tiene y 1 es que sí. En ambas variables hay desbalance: la enorme mayoría tiene valor 0.

La `edad` ocupa desde 0 a 80 años, habiendo un pico distintivo en torno a los 80, pero el resto de las edades están bastante distribuidas.

Tanto la `hemoglobina glicosilada` como la `glucemia` presentan unos pocos valores elevados, pero los demás tiene una buena distribución. La mayoría están alrededor de 6 para la `hemoglobina glicosilada`, lo que significa que muchos se encuentran en la zona umbral. Luego, la mayoría están alrededor de 150 para la `glucemia`, encontrándose ya unos 25 puntos por encima del umbral.

Finalmente, el `índice de masa corporal`va desde 10 hasta 96, aunque el 75% de los valores se encuentra por debajo de los 30, ubicándose entre valores normales y sobrepeso.

![](histogramas.png)

### Variable objetivo

Las clases están super desbalanceadas: más del 80% corresponden a personas no diabéticas.

![](diabetes.png)

## Preprocesamiento

Armamos un pipeline de preprocesamiento, el cual se encarga de:
* Mapeamos las variables categíricas a numéricas utilizando one-hot encoding.
* Normalizamos las variables categóricas.

![](pipeline.png)

Lo guardamos en un archivo pkl para utilizarlo al inicio de los modelos.

## XGBoost

Marce. Comentar un poco sobre el clasificador, que se hace, resultados, etc

## Redes neuronales

Santi. Comentar un poco sobre el clasificador, que se hace, resultados, etc

## Máquinas de vectores de soporte

Las **máquinas de vectores de soporte (support-vector machines, SVM)** son clasificadores binarios y, en principio, lineales, las cuales, a diferencia del perceptrón, sí contemplan la optimización de la frontera de decisión que permite separar las dos clases: se busca aquel hiperplano que esté lo más separado posible de los puntos más cercanos al mismo. Dichos puntos son conocidos como **vectores de soporte**, mientras que el espacio entre ellos y el hiperplano se conoce como **margen**. Con esto presente, podemos decir que el algoritmo SVM encuentra el hiperplano que devuelve el mayor margen entre sí mismo y los vectores de soporte. Por esta razón es que este tipo de clasificador a veces es conocido como **clasificador por márgenes (margin classifier)**. Se dice que en principio es un clasificador lineal ya que es excelente para claificar conjuntos que son linealmente separables. Sin embargo, cuando las clases no son estrictamente linealmente separables, pero presentan un solapamiento moderado, se define una **tolerancia (C)** al error: permitimos que haya datos que caigan dentro del margen de error o incluso que estén del lado incorrecto del hiperplano. Existen problemas de clasificación complejos imposibles de resolver usando funciones lineales, incluso tomando una tolerancia. En estos casos, se puede generalizar SVM para que considere funciones no lineales sobre el espacio de parámetros original. Esto se logra al recurrir al **Kernel trick**: mapear los datos sobre un espacio de mayores dimensiones donde se espera, con alta probabilidad, que las clases sí sean linealmente separables. 

Teniendo todo esto en cuenta, se hizo un barrido de hiperparámetros, variando: `C` (tolerancia), `Kernel` (Kernel a utilizar en el mapeo), `class_weight` (si considera o no un balanceo entre las clases presentes) y `degree` (grado del polinomio, en caso de un Kernel polinómico). La combinación óptima resultó ser: 
* `C` = 1E3
* `Kernel` = 'poly'
* `class_weight` = None (default)
* `degree` = 3 (default)

Con estos hiperparámetros, se los valores de accuracy y la matriz de confusión obtenidos son:
* Entrenamiento: 96.916%
* Validación: 97.032%

![](confusion_svm.png)

Al submitir las predicciones de este modelo en la competencia Kaggle, se obtuvo un score de 97.133%, estando 0.133% por debajo del baseline.

## Conclusiones

cierre del entregable