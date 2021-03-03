---
title: Ejercicios
linktitle: Ejercicios
type: book
date: "2019-05-05T00:00:00+01:00"
# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 2
---

### 1- Ejercicio


```python
pi = 3.14159 # aproximación
diametro = 3
```

Crea una variable llamada `radio` igual a la mitad del diametro:


```python

```

Crea una variable llamada `area` usando la formula del area para un circulo: pi x radio al cuadrado


```python

```

### 2- Ejercicio


```python
a = [1, 2, 3]
b = [3, 2, 1]
```

Intercambia los valores de las variables `a` y `b`. Es decir, haz que la variable `a` coja el valor de `b` y vicerversa.

(Pista: Se puede hacer en una linea usando [esta idea](https://stackoverflow.com/questions/14836228/is-there-a-standardized-method-to-swap-two-variables-in-python))


```python

```

### 3- Ejercicio

Pon los parentesis necesarios ha la siguiente expresión para que el resultado sea 1.


```python
5 - 3 // 2
```

Pon los parentesis necesarios ha la siguiente expresión para que el resultado sea 0.


```python
8 - 3 * 2 - 1 + 1
```

### 4- Ejercicio

Alicia, Jorge y Carolina han acordado repartirse los dulces que han recaudado en Hallowen por partes iguales. Por lo tanto, por el bien de su amistad, han decidido regalar los dules restantes. Por ejemplo, si recolectaran 91 dulces, cada uno recibiría 20 dulces y el 1 restante lo regalaría.

Escribe una expresión aritmética para calcular cuantos dulces deberían regalar dependiendo de cuantos dulces hayan recogido.


```python
dulces_alicia = 121
dulces_jorge = 77
dulces_carolina = 109
```


```python
para_regalar = -1 # Completa la expresión aquí
```

### 5- Ejercicio

Completa la función de Python para que devuelva el número que se le pasa redondeado a dos decimales. Por ejemplo:
```
>>> redondear(3.14159)
3.14
````

(Pista: Python tiene una función incorporada llamada `round`)

(`pass` es una palabra clave que no hace literalmente nada. La usamos como un marcador de posición porque después de comenzar un bloque de código, Python requiere al menos una línea de código)


```python
def redondear(num):
    pass
```

### 6- Ejercicio

La ayuda para `round` dice que `ndigits` (el segundo argumento) puede ser negativo.
¿Qué crees que pasará cuando lo sea? Prueba algunos ejemplos en la siguiente celda.

¿Puedes pensar en un caso en el que esto sea útil?


```python

```

### 7- Ejercicio

En el 4º ejercicio, los amigos que compartían caramelos intentaron repartir los caramelos a partes iguales. Por el bien de su amistad decidieron que los caramelos sobrantes los regalarían. Por ejemplo, si colectivamente reuniesen 91 caramelos, se llevarán 30 cada uno y regalarían uno.

Abajo hay una simple función que calculará el número de caramelos a regalar para cualquier número de caramelos totales.

Modifíquela para que opcionalmente tome un segundo argumento que represente el número de amigos entre los que se dividen los caramelos. Si no se proporciona un segundo argumento, debería asumir 3 amigos, como antes.


```python
def pa_regalar(total_dulces):
    return total_dulces % 3
```

### 8- Ejercicio

Las siguientes expresiones de este ejercicio contienen algún error. Antes de ejecutarlos y que salga el mensaje de error intenta adivinar que pasará. Luego ejecutalos y mira los mensajes de error que salen (es importante entender los mensajes de error que nos devuelve Python). Para finalizar, corrige los errores y vuelve a ejecutar las celdas.


```python
redondear_(9.99999999)
```


```python
x = -10
y = 5
smallest_abs = min(abs(x, y))
```


```python
def f(x):
    y = abs(x)
return y

print(f(5)=
```

### 9- Ejercicio

Muchos lenguajes de programación tienen el signo disponible como una función incorporada. Python no, pero podemos definir el nuestro.

En la celda de abajo, define una función llamada signo que toma un argumento numérico y devuelve -1 si es negativo, 1 si es positivo, y 0 si es 0.


```python

```

### 10- Ejercicio

Hemos decidido añadir un mensaje de inicio a nuestra función `pa_regalar` del ejercicio anterior.


```python
def pa_regalar(total_dulces):
    print("Repartiendo", total_dulces, "dulces")
    return total_dulces % 3

pa_regalar(91)
```

¿Qué sucede si llamamos a la función con `total_dulces = 1`?


```python
pa_regalar(1)
```

Aunque funciona correctamente, el mensaje no es correcto gramaticalmente. Si tan solo hay un caramelo, el mensaje debería de decir "Repartiendo 1 dulce" en vez de "dulces". Arreglalo.


```python
def pa_regalar(total_dulces):
    print("Repartiendo", total_dulces, "dulces")
    return total_dulces % 3

pa_regalar(91)
pa_regalar(1)
```

### 11- Ejercicio

Supongamos que al salir de casa quiero saber si voy bien preparado para el clima o no. Voy bien preparado si...
  * llevo un paraguas...
  * o si llueve suavemente (zirimiri) y llevo un gorro...
  * en cualquier otro caso, estoy bien a menos que este lloviendo y sea un día laboral

La función `preparado_para_clima` no funciona perfectamente. En algún caso devuelve el valor incorrecto. Encuentra la configuración para la que el resultado no es el que debería:


```python
def preparado_para_clima(paraguas, intensidad_lluvia, gorro, dia_laboral):
    # No cambies el código de esta función, el objetivo es encontrar el bug, no arreglarlo
    return paraguas or intensidad_lluvia < 5 and gorro or not intensidad_lluvia > 0 and dia_laboral

# Cambia estos valores para encontrar el caso incorrecto
paraguas = True
intensidad_lluvia = 0.0
gorro = True
dia_laboral = True

# Checkea la salida de la función
actual = preparado_para_clima(paraguas, intensidad_lluvia, gorro, dia_laboral)
print(actual)
```

Ahora soluciona el error de la función y prueba que todo funcione bien:


```python
def preparado_para_clima(paraguas, intensidad_lluvia, gorro, dia_laboral):
    # ARRGLA el código
    return paraguas or intensidad_lluvia < 5 and gorro or not intensidad_lluvia > 0 and dia_laboral
```


```python
# Cambia estos valores para ver que todo funciona bien
paraguas = True
intensidad_lluvia = 0.0
gorro = True
dia_laboral = True

# Checkea la salida de la función
actual = preparado_para_clima(paraguas, intensidad_lluvia, gorro, dia_laboral)
print(actual)
```

### 12- Ejercicio

Las variables booleanas `ketchup`, `mostaza` y `cebolla` representan si un cliente quiere un determinado acompañamiento para su perrito caliente. Queremos implementar una serie de funciones booleanas que corresponden a algunas preguntas de sí o no sobre el pedido del cliente. Por ejemplo:


```python
def sin_cebolla(ketchup, ):
    """Devuelve true si el cliente no quiere cebolla
    """
    pass
```


```python
def lo_quire_todo(ketchup, mostaza, cebolla):
    """Devuelve true si el cliente lo quiere con todo
    """
    pass
```


```python
def solo(ketchup, mostaza, cebolla):
    """Devuelve true si el cliente quiere un perrito sin nada
    """
    pass
```


```python
def solo_salsa(ketchup, mostaza, cebolla):
    """Devuelve true si el cliente quiere ketchup o mostaza, pero no los dos a la vez
    """
    pass
```

### 13- Ejercicio

Completa la siguiente función para que devuelva el segundo elemento de la lista que recibe como entrada. Si la lista no tiene segundo elemento, devuelve `None`.


```python
def select_second(lista):
    pass
```


```python
lista = range(10)
select_second(lista)
```

### 14- Ejercicio

Estás analizando los equipos deportivos.  Los miembros de cada equipo se almacenan en una lista. El entrenador es el primer nombre de la lista, el capitán es el segundo nombre de la lista, y los demás jugadores están en la lista después de eso. 

Estas listas se almacenan en otra lista, que comienza con el mejor equipo y continúa a través de la lista hasta el último equipo.  Completa la siguiente función para seleccionar el **capitán** del peor equipo.


```python
def capitan_ultimo_equipo(equipos):
    pass
```


```python
equipos = [['Paul', 'John', 'Ringo', 'George']]
capitan_ultimo_equipo(equipos)
```

### 15- Ejercicio

La próxima iteración de Mario Kart incluirá un nuevo elemento extra-excitante, la *Concha Púrpura*. Cuando se usa, deforma el último lugar en el primer lugar y el primer lugar en el último. Completa la siguiente función para implementar el efecto de la *Concha Púrpura*.


```python
def concha_purpura(corredores):
    pass
```


```python
r = ["Mario", "Bowser", "Luigi"]
concha_purpura(r)
```

### 16- Ejercicio

¿Cuál es la longitud de las siguientes listas? Rellena la variable `longitudes` con tus predicciones. (Intenta hacer una predicción para cada lista *sin* usar `len()` en ella.)


```python
a = [1, 2, 3]
b = [1, [2, 3]]
c = []
d = [1, 2, 3][1:]

# Rellena la lista longitudes con tus 4 predicciones
longitudes = []
```


```python
longitudes == [len(a), len(b), len(c), len(d)]
```

### 17- Ejercicio

Estamos usando listas para registrar a la gente que asistió a nuestra fiesta y en qué orden llegaron. Por ejemplo, la siguiente lista representa una fiesta con 7 invitados, en la que Adela apareció primero y Ford fue el último en llegar:

```
    asistentes = ['Adela', 'Fleda', 'Owen', 'May', 'Mona', 'Gilbert', 'Ford']
```

Se considera que un invitado llega "elegantemente tarde" si llegó después de la mitad de los invitados de la fiesta. Sin embargo, no deben ser el último invitado (eso es ir demasiado lejos). En el ejemplo anterior, Mona y Gilbert son los únicos invitados que llegaron elegantemente tarde.

Completa la siguiente función que toma una lista de los asistentes a la fiesta así como una persona, y nos dice si esa persona llega elegantemente tarde.


```python
def elegantemente_tarde(asistentes, nombre):
    pass
```

### 18- Ejercicio

¿Cual crees que será la salida de la siguiente celda?


```python
[1, 2, 3, 4] > 2
```

R y Python tienen algunas bibliotecas (como numpy y pandas) que comparan cada elemento de la lista con el 2 (es decir, hacen una comparación "por elementos") y nos dan una lista de booleanos como `[Falso, Falso, Verdadero, Verdadero]`.

Implementa una función que reproduzca este comportamiento, devolviendo una lista de booleans correspondiente a si el elemento correspondiente es mayor que n.


```python
def elementos_mayores(L, umbral):
    """Devuelve una lista con la misma longitud que L, donde el valor en el índice i es 
    True si L[i] es mayor que la varible umbral, y falso si no.
    """
    pass
```


```python
elementos_mayores([1, 2, 3, 4], 2)
```


```python
elementos_mayores([10, -5, 3, 2, 0, -2], 0)
```

### 19- Ejercicio

Crea una función que dada una lista de comidas servidas durante algún período de tiempo, devuelve `True` si la misma comida ha sido servida dos días seguidos, y `False` de otra manera.


```python
def menu_aburrido(meals):
    pass
```


```python
menu_aburrido(['ensalada', 'pasta', 'humus', 'tortilla', 'pasta', 'verduras'])
```


```python
menu_aburrido(['ensalada', 'filete', 'filete', 'tortilla', 'pasta', 'verduras'])
```

### 20- Ejercicios

Hay un dicho que dice: "Los científicos de datos pasan el 80% de su tiempo limpiando datos, y el 20% de su tiempo quejándose de la limpieza de datos". Veamos si puedes escribir una función para ayudar a limpiar los datos del código postal de los EE.UU. Dada una cadena, debería devolver si esa cadena representa o no un código postal válido. En este caso, un código postal válido es cualquier cadena que consista exactamente de 5 dígitos.

SUGERENCIA: `str` tiene un método que será útil aquí para confirmar que nos están pasando un string con solo digitos. Usa `help(str)` para revisar una lista de métodos de `str`.


```python
def zip_valido(zip_code):
    pass
```


```python
zip_valido('51432')
```


```python
zip_valido('4as31')
```


```python
zip_valido('3')
```


```python
zip_valido('37408917586')
```
