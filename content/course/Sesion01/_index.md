---
# Title, summary, and page position.
linktitle: Sesión 01
weight: 1
icon: book
icon_pack: fas

# Page metadata.
title: Introducción a Python
type: book  # Do not modify.
---

## 1- Python en modo interprete (calculadora)

 - En jupyter notebooks escribimos código en cada **celda**.
 - Hay celdas de tipo **código** y tipo **texto**.
 - Lo ejecutamos con **ctrl + Intro**


```python
2+3
```




    5




```python
2**3
```




    8




```python
( (1+8)**(1/2) )*0.25
```




    0.75




```python
'hola' + ' ' + 'mundo'
```




    'hola mundo'




```python
print('hola mundo')
print('hola mundo'*2)
print("hola mundo"*2)
```

    hola mundo
    hola mundohola mundo
    hola mundohola mundo
    

**¿Cómo introducimos palabras entre comillas dentro de un string?**


```python
print("hola 'mundo'")
print('hola \'mundo\'')
```

    hola 'mundo'
    hola 'mundo'
    


```python
type(2) # Ajam, ajam! Probando, probando!

# Esto es un comentario! Se ignora a la hora de ejecución (y compilación).



# type nos devuelve el tipo de variable introducido
```




    int




```python
print('Hay 4 tipos de variables básicas en Python:')
print('\t - int: variable de número entero (integer)')
print('\t - float: variable de número real (float point)')
print('\t - str: variable string, de carácteres de texto (string)')
print('\t - bool: variable binaria, True o False (boolean)')
print()
print('Estos tipos de variable pueden asignarse a nombres, para trabajar en modo \'compinador\'')

# Aquí estamos devolviendo en la pantalla la explicación de las variables
#  básicas de Python... quizá un poco aburrido ;)

print('\n \'print()\' devuelve en la pantalla variables de tipo str, int, float, bool.')
```

    Hay 4 tipos de variables básicas en Python:
    	 - int: variable de número entero (integer)
    	 - float: variable de número real (float point)
    	 - str: variable string, de carácteres de texto (string)
    	 - bool: variable binaria, True o False (boolean)
    
    Estos tipos de variable pueden asignarse a nombres, para trabajar en modo 'compinador'
    
     'print()' devuelve en la pantalla variables de tipo str, int, float, bool.
    


```python
este_curso_es_de_python = True
print(este_curso_es_de_python)
```

    True
    


```python
#'HOLA'.lower() # Cada variable tiene su juego, lo veremos más adelante.
'hola'.upper()
```




    'HOLA'




```python
str(2.0) # Las variables admiten conversión
```




    '2.0'




```python
int(2.0) # Las variables admiten conversión
```




    2



## 2- Python en modo compilador (programación estructurada)

Cada línea es una orden independiente

No hace falta marcar el **final de linea**, como en otros lenguajes de programación (Matlab, Java, C)


```python
a = 12
b = 10
c = a*b
print(c)

d = a/b
print(d)


e = a//b # Redondeo a número entero, hacia abajo
print(e)

f = a%b  # Resto
print(f)


resto = f
cociente = e
print(cociente*b+resto) # Igual a la variable 'a' (dividendo)

```

    120
    1.2
    1
    2
    12
    


```python
#del a
```


```python
master = 'Modelización'

print(master*b)
```

    ModelizaciónModelizaciónModelizaciónModelizaciónModelizaciónModelizaciónModelizaciónModelizaciónModelizaciónModelización
    


```python
curso = 2020
print(f'El master es sobre {master} - {curso}')
```

    El master es sobre Modelización - 2020
    

## 3 - Variables compuestas: listas, tuplas, diccionarios

 - Estas variables contienen un número de variables dentro de sí.

 - Las variables contenidas pueden ser de diferentes tipos

### 3.1 - Listas (list)


```python
numeros = [1.0, 2, 3.0, 4, 5]
nombres = ['Gorka', 'Irene', 'Pedro', 'Ana', 'Leire']
datos = [0, True, '1999', 3.1415]
```


```python
type(numeros)
```




    list



Cada elemento tiene una posición asignada: 0,1,2,...

Para obtener el elemento en alguna de las posiciones se utilizan corchetes '[0]'


```python
print(numeros[0])
print(nombres[3])
datos[1:3] # Desde el elemento 1 hasta el 2
```

    1.0
    Ana
    




    [True, '1999']



Los valores pueden sustituirse (se dice que las listas son mutables)


```python
nombres[0] = 'Calabacín'
print(nombres) # Imprimimos toda la lista, esta se convierte a tipo string implicitamente
```

    ['Calabacín', 'Irene', 'Pedro', 'Ana', 'Leire']
    

El número de elementos se obtiene mediante 'len(nombres)'.

Nuevos elementos pueden añadirse mediante '.append()'


```python
len(nombres)
```




    5




```python
print(len(nombres))
nombres.append('Gorka')
print(len(nombres)) # Un elemento más que antes
print(nombres)
```

    5
    6
    ['Calabacín', 'Irene', 'Pedro', 'Ana', 'Leire', 'Gorka']
    

Las listas pueden 'pegarse' mediante el oprador de suma (+)


```python
datos2 = numeros + datos
print(datos2)
```

    [1.0, 2, 3.0, 4, 5, 0, True, '1999', 3.1415]
    

- **más opciones: slicing, sum, listas de listas**


```python
# Seleccionar elementos hasta
datos2[1:6] # Del segundo elemento hasta el 6to (recordad: empieza de 0)
```




    [2, 3.0, 4, 5, 0]




```python
# Sumar elementos en la lista
sum(datos2[1:6])
```




    14.0




```python
# Esto devuelve error
#sum(nombres)
```


```python
# Podemos creas listas con listas
lista_de_listas = [datos2, datos]
print(lista_de_listas)
```

    [[1.0, 2, 3.0, 4, 5, 0, True, '1999', 3.1415], [0, True, '1999', 3.1415]]
    

### 3.2 - Tuplas (tuple)

Como las listas, pero en este caso son inmutables, las **variables que la componen no pueden alterarse**.

Se utilizan los paréntesis


```python
jupyter = ('J', 'U', 'P', 'Y', 'T', 'E', 'R')
```


```python
jupyter[3]
```




    'Y'




```python
jupyter[3] = 'A' # Esto nos devuelve un error (TypeError)
#                    devido a la propiedad inmutable de las tuplas
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-30-76ddbd2c5ec1> in <module>
    ----> 1 jupyter[3] = 'A' # Esto nos devuelve un error (TypeError)
          2 #                    devido a la propiedad inmutable de las tuplas
    

    TypeError: 'tuple' object does not support item assignment



```python
jupyter[3:7] + jupyter[0:3] # Esto devuelve otra tupla
```




    ('Y', 'T', 'E', 'R', 'J', 'U', 'P')



**¿En que caso utlizarias tú una lista y en cual una tupla?**

### 3.3 - Diccionarios (dict)

Es una combinación de 'llave' y 'valor', se utilizan los corchetes **'{llave: valor}'**.


```python
# Definimos nuestro diccionario
constantes = {'pi': 3.14159265358979, 'phi': ( 1+5**(1/2) )/2}
```


```python
# Accedemos al valor (value) de la llave (key) 'phi'
constantes['phi']
```




    1.618033988749895



**Se pueden añadir elementos**


```python
# Key  valor
n=99999999  # Un valor muy alto, 'infinito'
constantes['e'] = round((1+1/n)**n, 10) # Con el método round podemos redondear las decimales
```


```python
print(type(constantes))
print(constantes)
```

    <class 'dict'>
    {'pi': 3.14159265358979, 'phi': 1.618033988749895, 'e': 2.7182818315}
    

- **Llaves** (keys) del diccionario


```python
constantes.keys()
```




    dict_keys(['pi', 'phi', 'e'])



- **Valores** (values) del diccionario


```python
# Los valores (values) del diccionario
constantes.values()
```




    dict_values([3.14159265358979, 1.618033988749895, 2.7182818315])



- **Elementos** (items) del diccionario


```python
# Los elementos del diccionario
constantes.items()
```




    dict_items([('pi', 3.14159265358979), ('phi', 1.618033988749895), ('e', 2.7182818315)])



**Ejemplo: sala de cine**

El cine [Golem de la Alhondiga](https://golem.es/golemv9/cine.php?idCine=9) de Bilbo tiene una sala pequeña con **8 filas numeradas** y **5 columnas con letras**.

Queremos definir una estructura que nos permita acceder a los asientos, guardar su estado y consultarlo. Éste último debe ser variable, puesto que las entradas se pondrán a la venta mañana a las 10:00.


```python
asientos_cine = {(1, 'A'): 'Libre', (1, 'B'): 'Libre', (1, 'C'): 'Libre', (1, 'D'): 'Libre', (1, 'E'): 'Libre',
                 (2, 'A'): 'Libre', (2, 'B'): 'Libre', (2, 'C'): 'Libre', (2, 'D'): 'Libre', (2, 'E'): 'Libre',
                 (3, 'A'): 'Libre', (3, 'B'): 'Libre', (3, 'C'): 'Libre', (3, 'D'): 'Libre', (3, 'E'): 'Libre',
                 (4, 'A'): 'Libre', (4, 'B'): 'Libre', (4, 'C'): 'Libre', (4, 'D'): 'Libre', (4, 'E'): 'Libre',
                 (5, 'A'): 'Libre', (5, 'B'): 'Libre', (5, 'C'): 'Libre', (5, 'D'): 'Libre', (5, 'E'): 'Libre',
                 (6, 'A'): 'Libre', (6, 'B'): 'Libre', (6, 'C'): 'Libre', (6, 'D'): 'Libre', (6, 'E'): 'Libre',
                 (7, 'A'): 'Libre', (7, 'B'): 'Libre', (7, 'C'): 'Libre', (7, 'D'): 'Libre', (7, 'E'): 'Libre',
                 (8, 'A'): 'Libre', (8, 'B'): 'Libre', (8, 'C'): 'Libre', (8, 'D'): 'Libre', (8, 'E'): 'Libre'}
```


```python
asientos_cine[(4,'B')]
```




    'Libre'




```python
asientos_cine[(4,'C')]
```




    'Libre'




```python
# A las 10:06 compramos dos tickets en asientos contiguos
asientos_cine[(4,'B')] = 'Ocupado'
asientos_cine[(4,'C')] = 'Ocupado'
```


```python
asientos_cine
```




    {(1, 'A'): 'Libre',
     (1, 'B'): 'Libre',
     (1, 'C'): 'Libre',
     (1, 'D'): 'Libre',
     (1, 'E'): 'Libre',
     (2, 'A'): 'Libre',
     (2, 'B'): 'Libre',
     (2, 'C'): 'Libre',
     (2, 'D'): 'Libre',
     (2, 'E'): 'Libre',
     (3, 'A'): 'Libre',
     (3, 'B'): 'Libre',
     (3, 'C'): 'Libre',
     (3, 'D'): 'Libre',
     (3, 'E'): 'Libre',
     (4, 'A'): 'Libre',
     (4, 'B'): 'Ocupado',
     (4, 'C'): 'Ocupado',
     (4, 'D'): 'Libre',
     (4, 'E'): 'Libre',
     (5, 'A'): 'Libre',
     (5, 'B'): 'Libre',
     (5, 'C'): 'Libre',
     (5, 'D'): 'Libre',
     (5, 'E'): 'Libre',
     (6, 'A'): 'Libre',
     (6, 'B'): 'Libre',
     (6, 'C'): 'Libre',
     (6, 'D'): 'Libre',
     (6, 'E'): 'Libre',
     (7, 'A'): 'Libre',
     (7, 'B'): 'Libre',
     (7, 'C'): 'Libre',
     (7, 'D'): 'Libre',
     (7, 'E'): 'Libre',
     (8, 'A'): 'Libre',
     (8, 'B'): 'Libre',
     (8, 'C'): 'Libre',
     (8, 'D'): 'Libre',
     (8, 'E'): 'Libre'}



**Vuelve a ejecutar la celda debajo de *'Consultamos la disponibilidad'*. ¿Qué ha cambiado?**

## 4 - Estructuras de programación

 - Ciclos loop: **for**
 
 - Condicionales **if, elif** y **else**
 
 - Ciclos loop: **while / for**
     - desviaciones del ciclo **(continue, break, exit)**


### 4.1 - Ciclos (primeros loops)

**¿Qué pasa si ejecutamos una y otra vez la quinta celda de la sección *4.1 - Ciclos*?**

#### For


```python
for num in [1,2,3]: # Este for recorre los elementos de la lista, asignandole sus valores a num
    print(num)
```

    1
    2
    3
    


```python
number = 2
for number in [1,2,3,4,5]: # La variable number queda asociada al último ciclo del bucle
    print(number)
print()
print(number)
```

    1
    2
    3
    4
    5
    
    5
    

**range()**

nos sirve para recorrer números en loops


```python
for tiempo in range(10): # Empieza en 0, termina en 9
    print(tiempo)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    


```python
for tiempo in range(1,11): # Empieza en 1, termina en 10
    print(tiempo)
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    


```python
salto = 2
for tiempo in range(1,11,salto): # Empieza en 1 y hace saltos dependiendo de la variable salto devolviendo valores menores que 11
    print(tiempo)
```

    1
    3
    5
    7
    9
    


```python
salto = 2
for fila in range(1,6): # los bucles pueden anidarse (nested)
    for columna in range(1,6):
        print(fila, columna) # Estamos utilizando 'print' con dos variables separadas por coma
```

    1 1
    1 2
    1 3
    1 4
    1 5
    2 1
    2 2
    2 3
    2 4
    2 5
    3 1
    3 2
    3 3
    3 4
    3 5
    4 1
    4 2
    4 3
    4 4
    4 5
    5 1
    5 2
    5 3
    5 4
    5 5
    

**len()**

podemos utilizarlo dentro de **range**


```python
filas_list = ('A', 'B', 'C', 'D') # Creamos una tupla 

for fila_idx in range(len(filas_list)):
    print(fila_idx, filas_list[fila_idx]) # Imprimimos el indice y la variable asociada a esa posicion
```

    0 A
    1 B
    2 C
    3 D
    


```python
butacas = list(range(1,11)) # list() convierte el rango en una lista
print(butacas)

butacas = tuple(range(1,11)) # tuple() convierte el rango en una tupla (inmutable)
print(butacas)
```

    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    

 **¿Como guardarias el estado de las butacas de cada fila de un cine?**
 
<details>
<summary> ->Pista</summary>
Enciclopedia
</details>



### 4.2 - Condicionales

Ejecutar unas ordenes cuando una condición se cumple

Las variables booleanas se establecen muchas veces mediante los operadores de condición:

- '==' es igual a
- '!=' es diferente de
- '>=' es mayor o igual a
- '<=' es menor o igual a
- '>'  es mayor que, '<' es menor que

### If


```python
if este_curso_es_de_python: # Asegurate de que la celda 'In [8]' se ha ejecutado
    print('Hey Pythoner!')
```

    Hey Pythoner!
    


```python
if 15.22222 >= 39**(1/2):
    print('La condición se cumple.')
else:
    print('La condición no se cumple.')
```

    La condición se cumple.
    


```python
if 15.22222 >= 500**(1/2):
    print('La condición se cumple.')
    es_mayor_o_igual = True
else:                                        # En caso de que la expresión de arriba no se cumpla ejecuta las ordenes de debajo
    print('La condición no se cumple.')
    es_mayor_o_igual = False

print(es_mayor_o_igual)
```

    La condición no se cumple.
    False
    

**Múliples condiciones**


```python
orden_planetas_dict = {'Mercurio':1, 'Venus' : 2, 'Tierra': 3, 'Marte': 4, 'Júpiter': 5, 'Saturno': 6, 'Urano': 7, 'Neptuno': 8}

planeta = 'Marte'

if orden_planetas_dict[planeta] > orden_planetas_dict['Urano']:
    print(planeta, 'está más lejos del sol que Urano.')
elif orden_planetas_dict[planeta] > orden_planetas_dict['Marte']:
    print(planeta, 'está más lejos del sol que Marte.')
elif orden_planetas_dict[planeta] > orden_planetas_dict['Venus']:
    print(planeta, 'está más lejos del sol que venus.')
else:
    print(planeta, 'está más cerca del sol que la Tierra.')

```

    Marte está más lejos del sol que venus.
    

Una vez se cumple una condición el resto del bloque condicional se ignora.

   En este caso se imprimira un planeta más cercano al sol que el planeta especificado mediante la variable 'planeta'.

### 4.3 - Ciclos while


```python
numero  = 1
while numero < 15:
    print(numero, numero**2) 
    numero = numero +1 # Actualizar variable
```

    1 1
    2 4
    3 9
    4 16
    5 25
    6 36
    7 49
    8 64
    9 81
    10 100
    11 121
    12 144
    13 169
    14 196
    


```python
curso_activo = True
mes = 10
while curso_activo:
    print(mes)
    if mes > 12:
        curso_activo = False
    
    mes  = mes + 1 # Actualización de variable mes

print()
print(mes)
```

    10
    11
    12
    13
    
    14
    


```python
# Fibonacci 
ii0 = 0
ii1 = 1
while ii0 < 100:
    print(ii0)
    ii0, ii1 = ii1, ii0+ii1 # Reasignar valores en una sola linea, sin variables auxiliares (Truco de python)
```

    0
    1
    1
    2
    3
    5
    8
    13
    21
    34
    55
    89
    


```python
# Fibonacci 
ii0 = 1
ii1 = 1
tolerancia = 1e-12
contador = 1
while abs(ii1/ii0 - constantes['phi']) > tolerancia: # Asegurate de ejecutar la celda 'In [12]'
    ii0, ii1 = ii1, ii0+ii1 # Reasignar valores en una sola linea, sin variables auxiliares (Truco de python)
    # Contador de la serie
    contador += 1 # Equivalente a 'contador = contador + 1'

    #print(ii0, ii1, ii1/ii0)
# Cuando el bucle termina sabemos que ha convergido    
print(f'Se han necesitado {contador} iteraciones `para llegar a la convergencia con una tolerancia de {tolerancia}.\n\t {ii1}/{ii0} ~ Phi')
    
```

    Se han necesitado 30 iteraciones `para llegar a la convergencia con una tolerancia de 1e-12.
    	 1346269/832040 ~ Phi
    


```python
import math # De esta forma importamos librerias (conjunto de codigo que podemos usar)
# Pi (Leibniz)

contador = 0
potencia = 3
tolerancia = 10**(-potencia)
pi = 0
while abs(pi - math.pi) > tolerancia:
    signo = (-1)**(contador) # Comienza en positivo
    pi = pi + 4*(signo*(1.0/(2*contador+1))) # Comienza en 1
    contador += 1

print(f'Se han necesitado {contador} pasos para llegar a la convergencia con una tolerancia de {tolerancia}.\n\t pi~{round(pi,potencia)}')

```

    Se han necesitado 1000 pasos para llegar a la convergencia con una tolerancia de 0.001.
    	 pi~3.141
    

**Curiosidad:** En 1949 un ENIAC fue capaz de romper el récord, obteniendo 2037 cifras decimales en 70 horas. Para esto se utilizan series de convergencia más rápidas que la de Leibniz (http://crd.lbl.gov/~dhbailey/dhbpapers/dhb-kanada.pdf). Lista de records históricos actualizada en wikipedia (https://es.wikipedia.org/wiki/N%C3%BAmero_%CF%80).

Ordenes **break, continue y pass**.


```python
for ii in [0,3,1,-2,5,5,2,7,8]:
    print(ii)
    if ii==2: 
        break
        
print('El blucle ha finalizado!')
```

    0
    3
    1
    -2
    5
    5
    2
    El blucle ha finalizado!
    


```python
for ii in [0,3,1,-2,5,5,2,7,8]:
    if ii==2: 
        break
    print(ii)
    
print('El blucle ha finalizado!')
```

    0
    3
    1
    -2
    5
    5
    El blucle ha finalizado!
    

**pass** no hace nada...


```python
for ii in [0,3,1,-2,5,5,2,7,8]:
    if ii==2: 
        pass
    print(ii)
```

    0
    3
    1
    -2
    5
    5
    2
    7
    8
    

## 5 - Funciones

 - Son estructuras de código para **operaciones específicas**.
 - Se escriben **para evitar repeticiones** en el código.


 - Aportan **claridad y orden** mientras se programa y en sus futuras lecturas. (Bloques)
 - Su uso puede extenderse a otros problemas, **universalidad**.
 
 Los dos últimos puntos requieren un esfuerzo por parte de la persona programadora: nombres claros, comentarios explicativos, dividir un proceso en diferentes funciones más especificas, etc. Siempre con cierto balance entre esfuerzo y claridad.
 
 Podemos creas funciones que hagan calculos especificos, *desde simples sumas de dos números hasta procesamiento de texto/imagenes o calculos de simulaciones físicas...*


```python
def hola_mundo():
    print('hola mundo!')

hola_mundo()
```

    hola mundo!
    

### 5.1 - Variables de entrada y salida



```python
def print_numeros_entre(a, b):
    numero = a
    print(f'Imprimiendo numeros entre {a} y {b}')
    while numero+1 < b:
        numero +=1
        print(numero)

print_numeros_entre(0,6)
```

    Imprimiendo numeros entre 0 y 6
    1
    2
    3
    4
    5
    

Las variables definidas dentro de una función solo se reconocen dentro de la función, son variables **locales**.

**¿Como harias esto de forma más simple? (con range)**


```python
def print_numeros_entre(a,b): # a y b son variables de entrada
    
    # Inicializar lista vacia
    numeros_entre_list = [] 
    
    numero = a
    while numero+1 < b:
        numero +=1
        # Añadir numero a la lista
        numeros_entre_list.append(numero) 
    
    # Variable a devolver (salida)
    return(numeros_entre_list)

```


```python
# Llamamos tantas veces como queramos a la función, asignando su salida a una variable
lista1 = print_numeros_entre(-6,6)
lista2 = print_numeros_entre(100,110)

# Imprimimos, para comprobar el resultado y el tipo de variable que devuelve la función
print(lista1, type(lista1))
print()
print(lista2, type(lista2))
```

    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] <class 'list'>
    
    [101, 102, 103, 104, 105, 106, 107, 108, 109] <class 'list'>
    

### 5.2 - Variables de entrada opcionales


**Retomando el ejemplo anterior...**



```python
def print_numeros_entre_con_salto(a,b, salto=1): # a y b son variables de entrada, *salto* es una variable de entrada opcional
    
    # Inicializar lista vacia
    numeros_entre_list = [] 
    
    # Iniciar comienzo
    numero = a
    while numero+1 < b:
        # Avanzar un paso
        numero += salto
        # Añadir numero a la lista
        numeros_entre_list.append(numero) 
    
    # Variable a devolver (salida)
    return numeros_entre_list # Puede darse sin paréntesis 

```


```python
# Nuevamente... 

#Llamamos tantas veces como queramos a la función, asignando su salida a una variable
lista1 = print_numeros_entre_con_salto(-6,6)
lista2 = print_numeros_entre_con_salto(-100, 100, salto=20)

# Imprimimos, para comprobar el resultado
print(lista1)
print()
print(lista2)
```

    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    
    [-80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
    

Las variables opcionales siempre van después de variables no opcionales

- Ejemplo: 
 -  (a,b,c,..., z=1)    correcto ;)
 -  (a,b,c,...,y=1,z) incorrecto :(
 


**Listas de listas**



```python
def matriz_identidad(n=10):
    
    # Inicializar lista
    matriz = []
    
    # Avanzar por cada fila
    for ii in range(n):
        valores_fila = [] #
        # Obtener valores de cada columna
        for jj in range(n):
            if ii == jj:
                valores_fila.append(1.0)
            else:
                valores_fila.append(0.0)
        
        matriz.append(valores_fila) # *valores_fila* es una lista que añadimos a la lista *matriz*
    
    # Devolvemos la 'lista de listas'
    return(matriz)
                
```


```python
matriz_identidad(n=10)
```




    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]



**Funciones que llaman a otras funciones**



```python
def condicion_unidad(tipo_matriz, ii_fila, jj_columna):
    # Esta funcion devuelve 'True' si el elemento de fila, columna = ii_fila, jj_columna ha de ser la unidad
    #  dependiendo del tipo de matriz que queremos, introducido mediante *tipo_matrix*
    
    # *tipo_matrix* es una variable de tipo string
    #  *ii_fila* y *jj_columna* son variables de tipo entero int, especifican las coordenadas del elemento de la matriz
    #  que queremos saber si es nula (devolver False) o no (devolver True)
    

    # Introducimos las condiciones como hemos visto anteriormente.
    if tipo_matriz == 'identidad':
        return(ii_fila == jj_columna)
    elif tipo_matriz == 'triangular_inferior':
        return(ii_fila >= jj_columna)
    elif tipo_matriz == 'triangular_superior':
        return(ii_fila <= jj_columna)
    else:
        print('No se ha introducido un *tipo_matriz* válido.')
        return(False) # Devolvemos False por defecto
    
def matriz(tipo='identidad', n=10):
    # Esta funcion devuelve una matriz cuadrada de tipo *identidad* y dimension *n*
    # Se recorren las filas y columnas rellenandose con unos los valores necesarios,
    #  siguiendo la función *condición_unidad*
    
    # Inicializar lista
    matriz = []
    
    # Avanzar por cada fila
    for ii in range(n): # ii va de 0 a  n-1
        valores_fila = [] # Inicializar valores de la fila *ii*
        
        # Obtener valores de cada columna
        for jj in range(n):
            
            # Dejamos el rellenado a cargo de la función *condicion_unidad* mediante este bloque condicional
            if condicion_unidad(tipo, ii, jj): 
                valores_fila.append(1.0)
            else:
                valores_fila.append(0.0)
        matriz.append(valores_fila) # *valores_fila* es una lista que añadimos a la lista *matriz*
    
    # Devolvemos la 'lista de listas'
    return(matriz)
```


```python
matriz(tipo='triangular_inferior', n=6)
```




    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]



 - **¿Qué te parece el nombre de la función de condición '*condicion_unidad()*' ? ¿Cómo la llamarias tú?**


 - **¿Cómo harias una función que tome dos matrices y devuelva su suma?**

 - **¿Qué pasaría si la función '*condición_unidad()*' estuviera dentro de la función '*matriz*'?**



### 5.3 - Recursividad


```python
def factorial(n):
    if n == 1:
        return(1)
    else:
        return(n*factorial(n-1)) # La salida requiere de una llamada de la misma función
    
factorial(7)
```




    5040



*Nota:* Como **return** finaliza la función podemos obviar el **else**.


```python
def factorial(n):
    if n == 1: return(1) # Ifs permiten la escritura en una sola linea
    return(n*factorial(n-1)) # La salida requiere de una llamada de la misma función
    
factorial(7)
```




    5040



### 5.4 - \*args y \*\*kwargs

Son variables opcionales sin nombre asociado.

- **\* args (arguments)** es una variable iterable: los valores se acceden mediante la posición del elemento. (ejemplo: lista, tuple)

- **\*\* kargs (key arguments)** es un diccionario con *key:value* arbitrarias para la llamada de la función.



```python
def invitados_a_la_fiezta(titulo_str, *invitados):
    
    print( '-'*30 + '\n\t'+ titulo_str + '\n' + '-'*30)
    for numero_invitado in range(len(invitados)):
            print(numero_invitado+1, invitados[numero_invitado])
    print('-'*30)
    
    # Añadimos un texto tan solo en el caso de que un invitado, 'Barney', asistirá a la fiesta
    if 'Barney' in invitados:
        print('\n\t¡Ponte traje!\n')

    
lista_invitados = ['Lily', 'Ted', 'Barney', 'Marshall', 'Robin']
invitados_a_la_fiezta('Lista de invitados',*lista_invitados)
```

    ------------------------------
    	Lista de invitados
    ------------------------------
    1 Lily
    2 Ted
    3 Barney
    4 Marshall
    5 Robin
    ------------------------------
    
    	¡Ponte traje!
    
    


```python
invitados_a_la_fiezta('Lista de invitados', 'Mikel', 'Juan', 'Leire')
```

    ------------------------------
    	Lista de invitados
    ------------------------------
    1 Mikel
    2 Juan
    3 Leire
    ------------------------------
    


```python
# Probamos a no pasar *args
invitados_a_la_fiezta('Fiesta de Hawking')
```

    ------------------------------
    	Fiesta de Hawking
    ------------------------------
    ------------------------------
    


```python
def lista_invitados(titulo_str, **invitados_encargo):
    
    print( '-'*50 + '\n\t Lista de invitados y encargo \n' + '-'*50)
    numero_invitado = 1
    for invitado, encargo in invitados_encargo.items():
            print(numero_invitado,'-', invitado,'\t', encargo)
            numero_invitado += 1
    
    # Comprobamos si algún invitado traerá bebida (para esto comprobamos en los valores del diccionario asociado a los **kwargs)
    if 'Bebida' in invitados_encargo.values():
        print('\n   ¡Va a ser... \n\t                    ... LEGENDARIO !\n')

    print('-'*50)
    
invitados_encargo = {'Lily': 'Tarta', 'Ted': 'Bebida', 'Barney': 'Nada', 'Marshall': 'Juegos de mesa', 'Robin': 'Sticks'}
lista_invitados('Lista de invitados y encargo',**invitados_encargo)
```

    --------------------------------------------------
    	 Lista de invitados y encargo 
    --------------------------------------------------
    1 - Lily 	 Tarta
    2 - Ted 	 Bebida
    3 - Barney 	 Nada
    4 - Marshall 	 Juegos de mesa
    5 - Robin 	 Sticks
    
       ¡Va a ser... 
    	                    ... LEGENDARIO !
    
    --------------------------------------------------
    


```python
lista_invitados('Lista de invitados y encargo', Lily='Tarta', Ted='Refrescos', Barney='Nada', Marshall='Juegos de mesa')
```

    --------------------------------------------------
    	 Lista de invitados y encargo 
    --------------------------------------------------
    1 - Lily 	 Tarta
    2 - Ted 	 Refrescos
    3 - Barney 	 Nada
    4 - Marshall 	 Juegos de mesa
    --------------------------------------------------
    

# Resumen

Hemos visto:

- Dos diferentes **modos de programar**: 
 - *Interprete*: Consiste en introducir unos ordenes sobre la marcha
 - *Compilador*: Consiste en introducir un código estructurado que será comprobado, y ejecutado secuencialmente en caso de no tener errores de escritura. Es éste el que nos interesa.
- **Variables**
 - 4 tipos básicos: *int, float, string, boolean*
 - 3 compuestas principales: *list, tuple, dict* (Podeís mirarlo [aquí](https://docs.python.org/3/tutorial/datastructures.html))
- **Condicionales**
 - *if, elif* y *else*
 - Su uso con condiciones definidas por operadores de condición (==, !=, >=, <=, ...)
- **Ciclos**
 - **for** y su uso con **range()**
 - **while** y su uso con condiciones definidas por operadores de condición (==, !=, >=, <=, ...)
 - Como cambiar el curso de un ciclo mediante las ordenes **break**, **continue**
- **Funciones**
 - Variables de **entrada** y de **salida**.
 - **Variables opcionales**
 - **Variables adicionales *args y** ****kwargs**, nombre de variable no necesariamente conocido
- **Imprimir**
 - Formateado para impresión de diferentes variables

# Ejercicio de ejemplo final

Retomamos el ejemplo del cine Golem de la Alhondiga de Bilbao. 

Ahora, en vez de añadir manualmente los asientos, utilizaremos ciclos 
que iteren sobre las filas y columnas de los asientos del cine. Guardaremos
 la disponibilidad como valor de un diccionario, de forma tal que pueda ser modificada.


```python
 # Creamos una tupla a partir del rango para las filas (1,2,3,...,8)
filas = tuple(range(1,9))

# Creamos una tupla para las columnas
columnas = ('A', 'B', 'C', 'D', 'E') 

# Inicializamos el diccionario en el que almacenaremos los asientos y su estado
asientos_cine = {} 

# Recorremos todas las filas
for ii_fila in filas:
    # Recorremos todas las columnas para la fila *ii_fila**
    for jj_columna in columnas:
        
        # Creamos un elemento en el diccionario con la tupla (fila, columna) como key
        #   y el estado 'Libre' como value
        asientos_cine[(ii_fila, jj_columna)] = 'Libre'

```

Ahora imprimimos el estado de cada asiento mediante otro ciclo **for**


```python
if asientos_cine[(4,'B')] == 'Libre':
    print('Ocupamos el asiento ', (4,'B'))
    asientos_cine[(4,'B')] = 'Ocupado'
else:
    print('El asiento ', (4,'B'), 'está ocupado.')
```

    Ocupamos el asiento  (4, 'B')
    

Ahora imprimimos el estado de cada asiento mediante otro ciclo **for**


```python
for asiento, estado in asientos_cine.items(): 
    # Recorremos los asientos (fila, columna) y su estado, es decir, 
    #  la llave(key) y valor (value) del diccionario con el método .items()
    
    # Imprimimos con formato          fila                     columna
    print(f'Asiento de la fila \'{asiento[0]}\' y columna \'{asiento[1]}\' \t -> \t {estado}')
```

    Asiento de la fila '1' y columna 'A' 	 -> 	 Libre
    Asiento de la fila '1' y columna 'B' 	 -> 	 Libre
    Asiento de la fila '1' y columna 'C' 	 -> 	 Libre
    Asiento de la fila '1' y columna 'D' 	 -> 	 Libre
    Asiento de la fila '1' y columna 'E' 	 -> 	 Libre
    Asiento de la fila '2' y columna 'A' 	 -> 	 Libre
    Asiento de la fila '2' y columna 'B' 	 -> 	 Libre
    Asiento de la fila '2' y columna 'C' 	 -> 	 Libre
    Asiento de la fila '2' y columna 'D' 	 -> 	 Libre
    Asiento de la fila '2' y columna 'E' 	 -> 	 Libre
    Asiento de la fila '3' y columna 'A' 	 -> 	 Libre
    Asiento de la fila '3' y columna 'B' 	 -> 	 Libre
    Asiento de la fila '3' y columna 'C' 	 -> 	 Libre
    Asiento de la fila '3' y columna 'D' 	 -> 	 Libre
    Asiento de la fila '3' y columna 'E' 	 -> 	 Libre
    Asiento de la fila '4' y columna 'A' 	 -> 	 Libre
    Asiento de la fila '4' y columna 'B' 	 -> 	 Ocupado
    Asiento de la fila '4' y columna 'C' 	 -> 	 Libre
    Asiento de la fila '4' y columna 'D' 	 -> 	 Libre
    Asiento de la fila '4' y columna 'E' 	 -> 	 Libre
    Asiento de la fila '5' y columna 'A' 	 -> 	 Libre
    Asiento de la fila '5' y columna 'B' 	 -> 	 Libre
    Asiento de la fila '5' y columna 'C' 	 -> 	 Libre
    Asiento de la fila '5' y columna 'D' 	 -> 	 Libre
    Asiento de la fila '5' y columna 'E' 	 -> 	 Libre
    Asiento de la fila '6' y columna 'A' 	 -> 	 Libre
    Asiento de la fila '6' y columna 'B' 	 -> 	 Libre
    Asiento de la fila '6' y columna 'C' 	 -> 	 Libre
    Asiento de la fila '6' y columna 'D' 	 -> 	 Libre
    Asiento de la fila '6' y columna 'E' 	 -> 	 Libre
    Asiento de la fila '7' y columna 'A' 	 -> 	 Libre
    Asiento de la fila '7' y columna 'B' 	 -> 	 Libre
    Asiento de la fila '7' y columna 'C' 	 -> 	 Libre
    Asiento de la fila '7' y columna 'D' 	 -> 	 Libre
    Asiento de la fila '7' y columna 'E' 	 -> 	 Libre
    Asiento de la fila '8' y columna 'A' 	 -> 	 Libre
    Asiento de la fila '8' y columna 'B' 	 -> 	 Libre
    Asiento de la fila '8' y columna 'C' 	 -> 	 Libre
    Asiento de la fila '8' y columna 'D' 	 -> 	 Libre
    Asiento de la fila '8' y columna 'E' 	 -> 	 Libre
    

**¿Qué pasa si utilizamos listas [4,'B'] como key en el diccionario de los asientos de cine?** <details>
<summary> ->Esto:</summary>
Nos devolverá error, puesto que las listas son mutables. Los key de los diccionarios deben ser inmutables.
</details>

**¿Cómo extenderías estos procesos que hemos seguido para la gestión de más salas del cine?**

**¿Utilizarias una, o más funciones?**

Golem necesita tu ayuda!
