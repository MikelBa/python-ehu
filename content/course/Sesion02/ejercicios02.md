---
title: Ejercicios
linktitle: Ejercicios
type: book
weight: 2
---

# 1- Ejercicio

Importa los paquetes `numpy` y `pandas`.


```python

```

# 2- Ejercicio

Crea el siguiente `dataframe` y guardalo en tu directorio actual como `menu.csv`.

|         |       Juan       | Ana      |
|---------|:----------------:|----------|
| Plato 1 | Ensaladilla Rusa | Menestra |
| Plato 2 |      Merluza     | Entrecot |
| Postre  |      Cuajada     | Natillas |


```python

```

# 3- Ejercicio

Primero de todo, abre el conjunto de datos `winemag-data-130k-v2.csv` y guardalo en la variable reviews. Luego, crea una nueva variable llamada `df` que contenga las columnas `country`, `province`, `region_1`, y `region_2` de los indices `0`, `1`, `10`, y `100`:

![](https://i.imgur.com/FUCGiKP.png)


```python

```

# 4- Ejercicio

¿Cual es la mediana de la variable `points` en el conjunto de datos 'review'?


```python

```

# 5- Ejercicio

Vamos a estudiar la velocidad de las operaciones con el paquete `numpy`. Primero de todo, vamos a hacer la comparación con la operación de producto escalar:


```python
# importamos el módulo datetime del paquete datetime
from datetime import datetime

# creamos dos arrays de una dimensión
a = np.random.randn(100)
b = np.random.randn(100)
T= 100000

# definimos la función de producto escalar
def slow_dot_product(a, b):
    result = 0
    for e, f in zip(a, b):
        result += e*f
    return result

# medimos el tiempo necesario para hacer T=100000 operaciones con la función definida
t0 = datetime.now()
for t in range(T):
    slow_dot_product(a, b)
dt1 = datetime.now() - t0

# medimos el tiempo necesario para hacer T=100000 operaciones con la función dot de numpy
t0 = datetime.now()
for t in range(T):
    np.dot(a, b)
dt2 = datetime.now() - t0

print(f"dt1/dt2 = {dt1/dt2}")
```

Como podemos observar, hay una gran diferencia a favor de la función `dot` de `numpy`. Ahora, haz la misma comparación, pero para la operación de multiplicación entre matrices.

Define dos matrices con números aleatrorios con la función `np.random.randn` que tengan las dimensiones correctas para que la multiplicación sea una operación factible.


```python
A = np.random.randn(?, ?)
B = np.random.randn(?, ?)
```

Define la función `slow_matrix_mul`. Recuerda en la multiplicación de matrizes, el valor de la fila `i` y columna `j` de la matriz de salida viene del producto escalar entre la fila `i` de la matriz `a` con la columna `j` de la matriz `b`.


```python
def slow_matrix_mul(a, b):
    n = len(a)
    m = len(b[0])
    result = np.zeros((n, m))
    # Rellena aquí la función
    return result
```

Ahora, compara los tiempos:


```python
T = 100000

t0 = datetime.now()
for t in range(T):
    slow_matrix_mul(A, B)
dt1 = datetime.now() - t0

t0 = datetime.now()
for t in range(T):
    np.dot(A, B)
dt2 = datetime.now() - t0

print(f"dt1/dt2 = {dt1/dt2}")
```

Depende de las dimensiones de las matrizes `A` y `B`, y también el valor de la variable `T`; la ejecución de la celda anterior llevará un tiempo. Haz pruebas con diferentes valores para las dimensiones de 'A' y 'B'.
