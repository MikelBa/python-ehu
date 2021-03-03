---
title: "Aprendizaje supervisado: Clasificación y regresión"
linktitle: 6.1 Supervisado
type: book
weight: 2
---


El **aprendizaje sueprvisado** se denomina así porque tenemos un conjunto de datos que está formado por instancias con sus características (*features*) y sus etiquetas (*labels*). La tarea en este tipo de problemas trata de construir un algoritmo de estimación a partir de los datos etiquetados, de tal manera que si le pasamos una instancia nueva para la que no conocemos su etiqueta, el algoritmo sea capaz de estimarla correctamente.

Algunos ejemplos de problemas de aprendizaje supervisado son:
- predecir la especie de lirio a partir de un conjunto de medidas de la flor.
- dada una fotografía de una persona, identificar a la persona en la foto.
- dada una lista de películas que una persona ha visto y evaluado, recomendar una lista de películas que no haya visto y que le puedan gustar (el llamdo sistema de recomendación)

Hay dos tipos de problemas dentro del aprendizaje supervisado: **clasificación** y **regresión**. En los problemas de clasificación, las instancias tienen asociada una etiqueta discreta, mientras que en los problemas de regresión, la etiqueta es un valor continuo. Por ejemplo, en el caso de los lirios, la tarea de determinar la especie es un problema de clasificación. Por otro lado, puede que querramos estimar el peso de una persona a partir de una serie de observaciones y en este caso tendríamos un problema de regresión porque la etiqueta (el peso) es un valor continuo.


```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
```

## Ejemplo de problema de clasificación

El método de los **k vecinos más cercanos** o **k-nearest neighbors** (KNN) es uno de las estrategias más simples de aprendizaje: dada una nueva observación, buscamos en nuestro conjunto de datos que instancias tienen las características más parecidas y le asignamos la clase predonimante de entre esas instancias cercanas.

vamos a probar este algoritmo en nuestro problema de clasificación de los lirios:  


```python
from sklearn import neighbors, datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

# creamos el modelo
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

# ajustamos el modelo a nuestros datos
knn.fit(X, y)

# ¿Que especie de lirios tienen los sépalos de 3cm x 5cm y los pétalos de 4cm x 2cm?
result = knn.predict([[3, 5, 4, 2]])

print(iris.target_names[result])
```

    ['versicolor']
    

También podemos hacer prediciones de las probabilidades de que nuestra nueva instancia a cada clase:


```python
knn.predict_proba([[3, 5, 4, 2]])
```




    array([[0. , 0.8, 0.2]])




```python
from fig_code import plot_iris_knn
plot_iris_knn()
```


    
![svg](02_Supervisado_files/02_Supervisado_6_0.svg)
    


---

### Ejercicio

Usa un estimador diferente en el mismo problema: `sklearn.svm.SVC`.

*Ten en cuenta que no tienes que saber que algoritmo es y como funciona para poder usarlo. Simplemente estamos prbando la interfaz de `scikit-learn`.*

*Si quieres profundizar más, intenta replicar la figura de arriba con el estimador SVC.*


```python
from sklearn.svm import SVC
```


```python

```

---

## Ejemplo de problema de regresión

Como ya hemos visto antes, uno de los ejemplos más simples del problema de regresión es ajustar una linea a los datos. Pero `scikit-learn` también incluye algoritmos de regresión más complicados.


```python
# Crear algunos datos
import numpy as np
np.random.seed(0)
X = np.random.random(size=(20, 1))
y = 3 * X.squeeze() + 2 + np.random.randn(20)

plt.plot(X.squeeze(), y, 'o');
```


    
![svg](02_Supervisado_files/02_Supervisado_12_0.svg)
    


Vamos a ajustar un modelo de regresión lineal a este conjunto de datos:


```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

# Dibuja los datos y el modelo de predicción
X_fit = np.linspace(0, 1, 100)[:, np.newaxis]
y_fit = model.predict(X_fit)

plt.plot(X.squeeze(), y, 'o')
plt.plot(X_fit.squeeze(), y_fit);
```


    
![svg](02_Supervisado_files/02_Supervisado_14_0.svg)
    


Scikit-learn también tiene modelos más sofisticados:


```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X, y)

# Dibuja los datos y el modelo de predicción
X_fit = np.linspace(0, 1, 100)[:, np.newaxis]
y_fit = model.predict(X_fit)

plt.plot(X.squeeze(), y, 'o')
plt.plot(X_fit.squeeze(), y_fit);
```


    
![svg](02_Supervisado_files/02_Supervisado_16_0.svg)
    


La discusión de si este modelo es mejor que el anterior o no depende de un número de factores que estudiaremos más adelante.
