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

En este entregable se pretende participar de una competencia de Kaggle, donde dataset es una recopilación de datos méedicos y demográficos de pacientes junto a su estado de diabetes.

El dataset consta de un total de 100000 registros: 95000 se utilizan en la etapa de entrenamiento, mientras que los otros 5000 son para la evaluación del modelo. Cada registro tiene 10 variables: además del ID del paciente y su estado de diabetes, hay 2 variables categóricas (sexo e historial de fumador) y 6 variables numéricas (edad, hipertensión, enfermedades cardíacas, índice de masa corporal, hemoglobina glicosilada y glucemia).

El objetivo es construir modelos que predigan si una persona tiene o no diabetes. Teniendo como referencia un árbol de decisión cuya exactitud es del 0.97266, al menos uno de nuestros modelos debe superar dicho valor.

## Exploración 

### Datos faltantes

Salvo las etiquetas que fueron eliminadas a propósito en el conjunto de datos de evaluación, no se tienen valores faltantes.

### Variables categóricas

Se puede observar que cada variable categórica posee pocas categorías. Además se observa que son variables nominales.

La variable del `sexo` tiene 3 opciones posibles: varón, mujer y otro. La variable `historial de fumador` tiene 6 opciones posibles: sin información, nunca, fumaba, fuma actualmente, no fuma actualmente y siempre.

Respecto a `sexo` hay casi un 60% de mujeres y un 40% de varones, por lo que no hay un gran desbalance entre estas dos categorías. Sí está extremadamente desbalanceado con otros: hay mucho menos que el 1%.

Respecto a `historial de fumador` las dos categorías más preponderantes con un 35% cada una son nunca y sin información, llevándose prácticamente el 70% de los datos. Las otras cuatro categorías ocupan el 30% restante de una manera más o menos equitativa.

### Variables numéricas

En ningún caso se observan outliers significativos, por lo que se decide no realizar ninguna eliminación ni imputación.

Las variables `hipertensión` y `enfermedad cardíaca` son en realidad respuestas binarias: 0 es que no tiene y 1 es que sí. En ambas variables hay desbalance: la enorme mayoría tiene valor 0.

La `edad` ocupa desde 0 a 80 años, habiendo un pico distintivo en torno a los 80, pero el resto de las edades están bastante distribuidas.

Tanto la `hemoglobina glicosilada` como la `glucemia` presentan unos pocos valores elevados, pero los demás tiene una buena distribución. La mayoría están alrededor de 6 para la `hemoglobina glicosilada`, lo que significa que muchos se encuentran en la zona umbral. Luego, la mayoría están alrededor de 150 para la `glucemia`, encontrándose ya unos 25 puntos por encima del umbral.

Finalmente, el `índice de masa corporal`va desde 10 hasta 96, aunque el 75% de los valores se encuentra por debajo de los 30, ubicándose entre valores normales y sobrepeso.

### Variable objetivo

    Las clases están super desbalanceadas: más del 80% corresponden a personas no diabéticas.

## Preprocesamiento

pipeline y qué hicimos

## XGBoost

Marce. Comentar un poco sobre el clasificador, que se hace, resultados, etc

## Redes neuronales

Santi. Comentar un poco sobre el clasificador, que se hace, resultados, etc

## Máquinas de vectores de soporte

Cheva. Comentar un poco sobre el clasificador, que se hace, resultados, etc

## Bosque aleatorio

Fede. Comentar un poco sobre el clasificador, que se hace, resultados, etc

## Conclusiones

cierre del entregable