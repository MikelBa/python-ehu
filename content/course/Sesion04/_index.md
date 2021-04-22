---
# Title, summary, and page position.
linktitle: Sesión 04
weight: 1
icon: book
icon_pack: fas

# Page metadata.
title: "Classes: programación dirigida a objetos"
type: book  # Do not modify.
---

{{< youtube Zj8hKoSe-8s >}}

**Librerias a importar**


```python
import numpy as np
```

## Estructura de clases - primeros ejemplos

Las clases se componen de **atributos** (datos) y **métodos** (funciones)


```python
class Pokemon(object): 
    
    edicion = 'rojo_fuego'
    def __init__(self, nombre_pokemon, tipo_pokemon, tamaño):
        
        # ATRIBUTOS
        # > Nombre Pokemon
        self.nombre = nombre_pokemon
        # > Tipo Pokemon
        self.tipo = tipo_pokemon
        # > Tamaño del Pokemon
        self.tamaño = tamaño
        
        # > Nivel
        self.nivel = np.random.randint(3,7) 
        
```

En este ejemplo hay un método, típico en las clases, que es el '\_\_init\_\_()' (constructor), que inicializa los atributos de la clase cuando una instancia de esta se crea.

Este no es necesario, pero se recomienda puesto que las variables fuera del '__init__()' tienen el peligro de ser accedidas antes de ser definidas.

Creamos dos instancias (objetos) de la clase creada, pasando los argumentos requeridos por el *constructor*, \_\_ init\_\_()


```python
pikachu = Pokemon('Pikachu', 'Eléctrico', 'pequeño')
charmander = Pokemon('Charmander', 'Fuego', 'pequeño')
```

Accedemos a los atributos mediante un punto

*El tipo*


```python
print(pikachu.tipo)
print(charmander.tipo)
```

    Eléctrico
    Fuego
    

*El tamaño*


```python
print(pikachu.tamaño)
print(charmander.tipo)
```

    pequeño
    Fuego
    

*El nivel*


```python
print(pikachu.nivel)
print(charmander.nivel)
```

    5
    3
    

*El artributo de clase:* edición


```python
print(pikachu.edicion)
```

    rojo_fuego
    

Se pueden definir funciones dentro del espacio (namespace) de la clase, estas son consideradas **métodos**


```python
class Pokemon(object): 
    
    edicion = 'rojo_fuego'
    def __init__(self, nombre_pokemon, tipo_pokemon, tamaño):
        
        # ATRIBUTOS
        # > Nombre Pokemon
        self.nombre = nombre_pokemon
        # > Tipo Pokemon
        self.tipo = tipo_pokemon
        # > Tamaño del Pokemon
        self.tamaño = tamaño
        
        # > Nivel
        self.nivel = np.random.randint(3,7) 
        
    def subir_nivel(self):
        # Subimos nivel si es posible
        if self.nivel < 100:
            print(f'{self.nombre} ha subido del Nº{self.nivel} al Nº{self.nivel+1}.')
            self.nivel += 1
        else:
            print(f'{self.nombre} no puede subir de nivel.')
```

El **método** 'subir\_nivel()' no utiliza ninguna variable de entrada, pero hemos introducido la palabra **'self'**:

    Ciertamente, todos los métodos, includio el *init*, requieren pasar el objeto en cuestión como primer parámetro. Esto se hace por convención utilizando la palabra *self*.



```python
pikachu = Pokemon('Pikachu', 'Eléctrico', 'pequeño')
charmander = Pokemon('Charmander', 'Fuego', 'pequeño')
```


```python
pikachu.subir_nivel()
```

    Pikachu ha subido del Nº5 al Nº6.
    

**Nota:**

    A diferencia de otros lenguajes dirigidos a objetos, Python no tiene concepto de elementos 'privados'. 

    Todo se comporta como público por defecto, es decir, puede accederse fuera del entorno de la clase. Por ejemplo, hemos obtenido los valores de *tipo*, *nombre*, *tipo* y nivel desde el objeto; al igual que el método 'subir\_nivel()'. 

**Métodos mágicos**

 Hay métodos que se dan en la forma '\_\_*nombre\_de\_método*\_\_'  (como el init). Estos son especiales en Python.
 Por ejemplo, los operadores de una clase. Estos comparan dos instancias 


```python
class Pokemon(object): 
    
    edicion = 'rojo_fuego'
    def __init__(self, nombre_pokemon, tipo_pokemon, tamaño):
        
        # ATRIBUTOS
        # > Nombre Pokemon
        self.nombre = nombre_pokemon
        # > Tipo Pokemon
        self.tipo = tipo_pokemon
        # > Tamaño del Pokemon
        self.tamaño = tamaño
        
        # > Nivel
        self.nivel = np.random.randint(3,7) 
        
    def subir_nivel(self):
        # Subimos nivel si es posible
        if self.nivel < 100:
            print(f'{self.nombre} ha subido del Nº{self.nivel} al Nº{self.nivel+1}.')
            self.nivel += 1
        else:
            print(f'{self.nombre} no puede subir de nivel.')
    
    # Incluimos OPERADORES que usen el atributo 'nivel' como referencia
    def __gt__(self, other):
        return self.nivel > other.nivel 
    def __lt__(self, other):
        return self.nivel < other.nivel
    def __ge__(self, other):
        return self.nivel >= other.nivel
    def __le__(self, other):
        return self.nivel <= other.nivel # Recordad que el return no requiere de paréntesis
    
    # Incluimos un 'método mágico' de imprimir
    def __str__(self):
        # Imprimimos atributos
        return(f'{self.nombre} (Nº_{self.nivel} | {self.tipo} | {self.tamaño})')
```

Creamos las instancias de los Pokemon


```python
pikachu = Pokemon('Pikachu', 'Eléctrico', 'pequeño')
charmander = Pokemon('Charmander', 'Fuego', 'pequeño')
```


```python
print(pikachu.nivel)
print(charmander.nivel)
```

    5
    6
    

Comparamos los Pokemon, es decir, su nivel


```python
pikachu >= charmander
```




    False



Imprimimos un Pokemon, es decir su nombre y atributos


```python
print(pikachu)
```

    Pikachu (Nº_5 | Eléctrico | pequeño)
    

**dir(), hasattr()**


```python
pikachu.nuevo_atributo = 'podemos_añadirlo'

hasattr(pikachu, 'nuevo_atributo')
```




    True



## **Bounded method, [Class method, static methods](https://stackoverflow.com/questions/12179271/meaning-of-classmethod-and-staticmethod-for-beginner)**


```python

```

## Inheritance (Basic & multiple)

Monkey patching

## Properties


```python
RECAP slides
```


      File "<ipython-input-403-ef798a623fd6>", line 1
        RECAP slides
                   ^
    SyntaxError: invalid syntax
    

