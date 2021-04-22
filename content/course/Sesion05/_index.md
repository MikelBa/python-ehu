---
# Title, summary, and page position.
linktitle: Sesión 05
weight: 1
icon: book
icon_pack: fas

# Page metadata.
title: "Algunas cosas"
type: book  # Do not modify.
---
    
{{< youtube 8M7HPLNgtxI >}}

# `os` module

El módulo `os` nos da funcionalidades para hacer tareas en el sistema operativo (Operative System = os). Algunas de esas tareas son las siguientes:

* Navegar por el sistema operativo
* Crear archivos y carpetas
* Eliminar archivos y carpetas
* Modificar archivos y carpetas


```python
import os

dir(os)
```




    ['DirEntry',
     'F_OK',
     'MutableMapping',
     'O_APPEND',
     'O_BINARY',
     'O_CREAT',
     'O_EXCL',
     'O_NOINHERIT',
     'O_RANDOM',
     'O_RDONLY',
     'O_RDWR',
     'O_SEQUENTIAL',
     'O_SHORT_LIVED',
     'O_TEMPORARY',
     'O_TEXT',
     'O_TRUNC',
     'O_WRONLY',
     'P_DETACH',
     'P_NOWAIT',
     'P_NOWAITO',
     'P_OVERLAY',
     'P_WAIT',
     'PathLike',
     'R_OK',
     'SEEK_CUR',
     'SEEK_END',
     'SEEK_SET',
     'TMP_MAX',
     'W_OK',
     'X_OK',
     '_Environ',
     '__all__',
     '__builtins__',
     '__cached__',
     '__doc__',
     '__file__',
     '__loader__',
     '__name__',
     '__package__',
     '__spec__',
     '_execvpe',
     '_exists',
     '_exit',
     '_fspath',
     '_get_exports_list',
     '_putenv',
     '_unsetenv',
     '_wrap_close',
     'abc',
     'abort',
     'access',
     'altsep',
     'chdir',
     'chmod',
     'close',
     'closerange',
     'cpu_count',
     'curdir',
     'defpath',
     'device_encoding',
     'devnull',
     'dup',
     'dup2',
     'environ',
     'error',
     'execl',
     'execle',
     'execlp',
     'execlpe',
     'execv',
     'execve',
     'execvp',
     'execvpe',
     'extsep',
     'fdopen',
     'fsdecode',
     'fsencode',
     'fspath',
     'fstat',
     'fsync',
     'ftruncate',
     'get_exec_path',
     'get_handle_inheritable',
     'get_inheritable',
     'get_terminal_size',
     'getcwd',
     'getcwdb',
     'getenv',
     'getlogin',
     'getpid',
     'getppid',
     'isatty',
     'kill',
     'linesep',
     'link',
     'listdir',
     'lseek',
     'lstat',
     'makedirs',
     'mkdir',
     'name',
     'open',
     'pardir',
     'path',
     'pathsep',
     'pipe',
     'popen',
     'putenv',
     'read',
     'readlink',
     'remove',
     'removedirs',
     'rename',
     'renames',
     'replace',
     'rmdir',
     'scandir',
     'sep',
     'set_handle_inheritable',
     'set_inheritable',
     'spawnl',
     'spawnle',
     'spawnv',
     'spawnve',
     'st',
     'startfile',
     'stat',
     'stat_result',
     'statvfs_result',
     'strerror',
     'supports_bytes_environ',
     'supports_dir_fd',
     'supports_effective_ids',
     'supports_fd',
     'supports_follow_symlinks',
     'symlink',
     'sys',
     'system',
     'terminal_size',
     'times',
     'times_result',
     'truncate',
     'umask',
     'uname_result',
     'unlink',
     'urandom',
     'utime',
     'waitpid',
     'walk',
     'write']



¿Cual es nuestro directorio de trabajo?


```python
os.getcwd()
```




    'C:\\Users\\mikel\\Desktop\\curso-de-python-mimec\\Sesion 05'



Cambiar el directorio de trabajo:


```python
os.chdir("./Sesion 05")

os.getcwd()
```




    'C:\\Users\\mikel\\Desktop\\curso-de-python-mimec\\Sesion 05'



Devolver lista de archivos y directorios:


```python
os.listdir(".")
```




    ['.ipynb_checkpoints', 'clase05.ipynb', 'Population_Data']




```python
os.listdir(os.getcwd())
```




    ['.ipynb_checkpoints', 'clase05.ipynb', 'Population_Data']



Crear carpetas:


```python
os.chdir(".")
```


```python
os.mkdir("prueba1")
```


```python
os.listdir()
```




    ['.ipynb_checkpoints', 'clase05.ipynb', 'Population_Data', 'prueba1']




```python
os.mkdir("prueba2/subprueba")
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-11-1806715c7d58> in <module>
    ----> 1 os.mkdir("prueba2/subprueba")
    

    FileNotFoundError: [WinError 3] El sistema no puede encontrar la ruta especificada: 'prueba2/subprueba'



```python
os.makedirs("prueba2/subpruebas")
```


```python
for root, dirs, files in os.walk("."):
    if dir!= '.git':
        level = root.replace(".", '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
```

    ./
        clase05.ipynb
        .ipynb_checkpoints/
            clase05-checkpoint.ipynb
        Population_Data/
            Alabama/
                Alabama_population.csv
            Alaska/
                Alaska_population.csv
            Arizona/
                Arizona_population.csv
            Arkansas/
                Arkansas_population.csv
            California/
                California_population.csv
            Colorado/
                Colorado_population.csv
            Connecticut/
                Connecticut_population.csv
            Delaware/
                Delaware_population.csv
    

Eliminar carpetas:


```python
os.rmdir("prueba1")
```


```python
os.listdir()
```




    ['.ipynb_checkpoints', 'clase05.ipynb', 'Population_Data', 'prueba2']




```python
os.rmdir("prueba2")
```


    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-17-c5532e03c0a9> in <module>
    ----> 1 os.rmdir("prueba2")
    

    OSError: [WinError 145] El directorio no está vacío: 'prueba2'



```python
os.removedirs("prueba2")
```


    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-18-9a0376c8d9c5> in <module>
    ----> 1 os.removedirs("prueba2")
    

    ~\miniconda3\lib\os.py in removedirs(name)
        237 
        238     """
    --> 239     rmdir(name)
        240     head, tail = path.split(name)
        241     if not tail:
    

    OSError: [WinError 145] El directorio no está vacío: 'prueba2'



```python
os.removedirs("prueba2/subpruebas/")
```


```python
os.listdir()
```




    ['.ipynb_checkpoints', 'clase05.ipynb', 'Population_Data']



Comprobar si un archivo o directorio existe:


```python
os.path.isdir("./prueba2")
```




    False




```python
os.path.isdir("Population_Data")
```




    True




```python
os.path.isfile("Population_Data/Alaska")
```




    False




```python
os.path.isfile("Population_Data/Alaska/Alaska_population.csv")
```




    True



Ejemplo útil de procesamiento de datos con el módulo `os`:


```python
os.getcwd()
```




    'C:\\Users\\mikel\\Desktop\\curso-de-python-mimec\\Sesion 05'




```python
for root, dirs, files in os.walk("."):
    level = root.replace(".", '').count(os.sep)
    indent = ' ' * 4 * (level)
    print('{}{}/'.format(indent, os.path.basename(root)))
    subindent = ' ' * 4 * (level + 1)
    for f in files:
        print('{}{}'.format(subindent, f))
```

    ./
        Alabama/
            Alabama_population.csv
        Alaska/
            Alaska_population.csv
        Arizona/
            Arizona_population.csv
        Arkansas/
            Arkansas_population.csv
        California/
            California_population.csv
        Colorado/
            Colorado_population.csv
        Connecticut/
            Connecticut_population.csv
        Delaware/
            Delaware_population.csv
    


```python
os.chdir("Population_Data/")
```


```python
import pandas as pd

# create a list to hold the data from each state
list_states = []

# iteratively loop over all the folders and add their data to the list
for root, dirs, files in os.walk(os.getcwd()):
    print(root)
    print(dirs)
    print(files)
    if files:
        list_states.append(pd.read_csv(root+'/'+files[0], index_col=0))

# merge the dataframes into a single dataframe using Pandas library
merge_data = pd.concat(list_states[1:], sort=False)
merge_data
```

    C:\Users\mikel\Desktop\curso-de-python-mimec\Sesion 05\Population_Data
    ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware']
    []
    C:\Users\mikel\Desktop\curso-de-python-mimec\Sesion 05\Population_Data\Alabama
    []
    ['Alabama_population.csv']
    C:\Users\mikel\Desktop\curso-de-python-mimec\Sesion 05\Population_Data\Alaska
    []
    ['Alaska_population.csv']
    C:\Users\mikel\Desktop\curso-de-python-mimec\Sesion 05\Population_Data\Arizona
    []
    ['Arizona_population.csv']
    C:\Users\mikel\Desktop\curso-de-python-mimec\Sesion 05\Population_Data\Arkansas
    []
    ['Arkansas_population.csv']
    C:\Users\mikel\Desktop\curso-de-python-mimec\Sesion 05\Population_Data\California
    []
    ['California_population.csv']
    C:\Users\mikel\Desktop\curso-de-python-mimec\Sesion 05\Population_Data\Colorado
    []
    ['Colorado_population.csv']
    C:\Users\mikel\Desktop\curso-de-python-mimec\Sesion 05\Population_Data\Connecticut
    []
    ['Connecticut_population.csv']
    C:\Users\mikel\Desktop\curso-de-python-mimec\Sesion 05\Population_Data\Delaware
    []
    ['Delaware_population.csv']
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>city_ascii</th>
      <th>state_id</th>
      <th>state_name</th>
      <th>county_fips</th>
      <th>county_name</th>
      <th>lat</th>
      <th>lng</th>
      <th>population</th>
      <th>density</th>
      <th>source</th>
      <th>military</th>
      <th>incorporated</th>
      <th>timezone</th>
      <th>ranking</th>
      <th>zips</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>184</th>
      <td>Anchorage</td>
      <td>Anchorage</td>
      <td>AK</td>
      <td>Alaska</td>
      <td>2020</td>
      <td>Anchorage</td>
      <td>61.1508</td>
      <td>-149.1091</td>
      <td>247949</td>
      <td>65.0</td>
      <td>polygon</td>
      <td>False</td>
      <td>True</td>
      <td>America/Anchorage</td>
      <td>2</td>
      <td>99518 99515 99517 99516 99513 99540 99567 9958...</td>
      <td>1840023385</td>
    </tr>
    <tr>
      <th>773</th>
      <td>Fairbanks</td>
      <td>Fairbanks</td>
      <td>AK</td>
      <td>Alaska</td>
      <td>2090</td>
      <td>Fairbanks North Star</td>
      <td>64.8353</td>
      <td>-147.6534</td>
      <td>63245</td>
      <td>375.0</td>
      <td>polygon</td>
      <td>False</td>
      <td>True</td>
      <td>America/Anchorage</td>
      <td>3</td>
      <td>99701 99703 99707</td>
      <td>1840023463</td>
    </tr>
    <tr>
      <th>1916</th>
      <td>Juneau</td>
      <td>Juneau</td>
      <td>AK</td>
      <td>Alaska</td>
      <td>2110</td>
      <td>Juneau</td>
      <td>58.4546</td>
      <td>-134.1739</td>
      <td>25085</td>
      <td>4.0</td>
      <td>polygon</td>
      <td>False</td>
      <td>True</td>
      <td>America/Juneau</td>
      <td>2</td>
      <td>99824 99801 99802 99803 99811 99812 99821 99850</td>
      <td>1840023306</td>
    </tr>
    <tr>
      <th>2516</th>
      <td>Badger</td>
      <td>Badger</td>
      <td>AK</td>
      <td>Alaska</td>
      <td>2090</td>
      <td>Fairbanks North Star</td>
      <td>64.8006</td>
      <td>-147.3877</td>
      <td>18792</td>
      <td>110.0</td>
      <td>polygon</td>
      <td>False</td>
      <td>False</td>
      <td>America/Anchorage</td>
      <td>3</td>
      <td>99705 99711</td>
      <td>1840023690</td>
    </tr>
    <tr>
      <th>2674</th>
      <td>Knik-Fairview</td>
      <td>Knik-Fairview</td>
      <td>AK</td>
      <td>Alaska</td>
      <td>2170</td>
      <td>Matanuska-Susitna</td>
      <td>61.4964</td>
      <td>-149.6535</td>
      <td>17513</td>
      <td>81.0</td>
      <td>polygon</td>
      <td>False</td>
      <td>True</td>
      <td>America/Anchorage</td>
      <td>3</td>
      <td>99654</td>
      <td>1840075080</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>23924</th>
      <td>Woodside</td>
      <td>Woodside</td>
      <td>DE</td>
      <td>Delaware</td>
      <td>10001</td>
      <td>Kent</td>
      <td>39.0712</td>
      <td>-75.5667</td>
      <td>193</td>
      <td>527.0</td>
      <td>polygon</td>
      <td>False</td>
      <td>True</td>
      <td>America/New_York</td>
      <td>3</td>
      <td>19943 19904 19980</td>
      <td>1840005821</td>
    </tr>
    <tr>
      <th>24553</th>
      <td>Viola</td>
      <td>Viola</td>
      <td>DE</td>
      <td>Delaware</td>
      <td>10001</td>
      <td>Kent</td>
      <td>39.0429</td>
      <td>-75.5714</td>
      <td>165</td>
      <td>359.0</td>
      <td>polygon</td>
      <td>False</td>
      <td>True</td>
      <td>America/New_York</td>
      <td>3</td>
      <td>19979</td>
      <td>1840003807</td>
    </tr>
    <tr>
      <th>25084</th>
      <td>Henlopen Acres</td>
      <td>Henlopen Acres</td>
      <td>DE</td>
      <td>Delaware</td>
      <td>10005</td>
      <td>Sussex</td>
      <td>38.7257</td>
      <td>-75.0849</td>
      <td>144</td>
      <td>217.0</td>
      <td>polygon</td>
      <td>False</td>
      <td>True</td>
      <td>America/New_York</td>
      <td>3</td>
      <td>19971</td>
      <td>1840006067</td>
    </tr>
    <tr>
      <th>25654</th>
      <td>Farmington</td>
      <td>Farmington</td>
      <td>DE</td>
      <td>Delaware</td>
      <td>10001</td>
      <td>Kent</td>
      <td>38.8699</td>
      <td>-75.5790</td>
      <td>122</td>
      <td>647.0</td>
      <td>polygon</td>
      <td>False</td>
      <td>True</td>
      <td>America/New_York</td>
      <td>3</td>
      <td>19950</td>
      <td>1840005805</td>
    </tr>
    <tr>
      <th>26837</th>
      <td>Hartly</td>
      <td>Hartly</td>
      <td>DE</td>
      <td>Delaware</td>
      <td>10001</td>
      <td>Kent</td>
      <td>39.1684</td>
      <td>-75.7127</td>
      <td>76</td>
      <td>518.0</td>
      <td>polygon</td>
      <td>False</td>
      <td>True</td>
      <td>America/New_York</td>
      <td>3</td>
      <td>19953</td>
      <td>1840005808</td>
    </tr>
  </tbody>
</table>
<p>3398 rows × 17 columns</p>
</div>




```python
os.chdir("..")
```

# `pathlib` module

`pathlib` es una libreria de Python que se utiliza para trabajar con _paths_. Pero, ¿que es un _path_? _path_ (ruta en castellano) es la forma de referenciar un archivo informático o directorio en un sistema de archivos de un sistema operativo determinado.

Hay dos tipos de _paths_:
* **Absolute paths**: Señalan la ubicación de un archivo o directorio desde el directorio raíz del sistema de archivos.
* **Relative paths**: Señalan la ubicación de un archivo o directorio a partir de la posición actual del sistema operativo en el sistema de archivos.

`pathlib` proporciona una forma más legible y fácil de construir *paths* representando las rutas del sistema de archivos como objetos adecuados.


```python
os.chdir("..")
```


```python
from pathlib import Path

absolute_path = Path.cwd() / "Population_Data"
relative_path = Path("Population_Data")
print(f"Absolute path: {absolute_path}")
print(f"Relative path: {relative_path}")
```

    Absolute path: C:\Users\mikel\Desktop\curso-de-python-mimec\Sesion 05\Population_Data
    Relative path: Population_Data
    


```python
absolute_path.is_dir()
```




    True




```python
relative_path.is_dir()
```




    True



### ¿Qué ventajas tiene `pathlib` respecto a `os.path`?


```python
alaska_file_os = os.path.join(os.getcwd(), 'Population_Data', "Alaska", "Alaska_population.csv")
alaska_file_os
```




    'C:\\Users\\mikel\\Desktop\\curso-de-python-mimec\\Sesion 05\\Population_Data\\Alaska\\Alaska_population.csv'




```python
alaska_file_os = "C:/Users/"
```


```python
alaska_file = Path.cwd() / "Population_Data" / "Alaska" / "Alaska_population.csv"
alaska_file
```




    WindowsPath('C:/Users/mikel/Desktop/curso-de-python-mimec/Sesion 05/Population_Data/Alaska/Alaska_population.csv')



Como podemos observar, el ejemplo de `pathlib` es más claro que el de `os.path`. Además, con `pathlib` se crea un objeto `Path`, que tiene asociado métodos.


```python
os.path.isfile(alaska_file_os)
```




    True




```python
alaska_file.is_file()
```




    True




```python
current_dir_os = os.getcwd()
current_dir = Path.cwd()

print(current_dir_os)
print(current_dir)
```

    C:\Users\mikel\Desktop\curso-de-python-mimec\Sesion 05
    C:\Users\mikel\Desktop\curso-de-python-mimec\Sesion 05
    


```python
os.mkdir(os.path.join(current_dir_os, "pruebaos"))
(current_dir / "pruebalib").mkdir()
```


```python
os.rmdir(os.path.join(current_dir_os, "pruebaos"))
(current_dir / "pruebalib").rmdir()
```

La conclusióne es que si podeís usar `pathlib` lo utilizeis porque aunque se puede obtener el mismo resultado con `os.path`, el código es más fácil de leer con `pathlib`.

# Input/Output files

Si ya hemos visto que los módulos como `numpy` o `pandas` tienen funciones para abrir archivos de diferentes tipos, ¿por qué nos interesa ahora aprender otra manera de trabajar con archivos?

Con esas librerias, los archivos que leiamos tenían que tener un tipo de estructura clara. En cambio, con estos métodos que vamos a proporner, no necesitamos que el archivo que vayamos a leer tenga una estructura tan clara.

Además, saber leer, escribir y guardar nuestras salidas en archivos puede ser útil. Aunque con `prints` podriamos hacer lo mismo, el problema es que lo que printeamos con print se guarda en la RAM y cuando cerramos Python, todos lo que habiamos mostrado desaparece.

Para abrir un archivo usaremos la función `open()`. Hay dos formas de usar la función:


```python
nombre = "Juan"
edad = 22

with open("texto.txt", "w", encoding="UTF-8") as f:
    f.write(f"Mi nombre es {nombre} y tengo {edad} años")
```


```python
nombre = "Ana"
edad = 23

f = open("texto.txt", "a", encoding="UTF-8")
f.write(f"\nMi nombre es {nombre} y tengo {edad} años")
f.close()
```

Estamos pasandole dos argumentos a la función `open()`. El primer argumento es una cadena que contiene el nombre del fichero. El segundo argumento es otra cadena que contiene unos pocos caracteres describiendo la forma en que el fichero será usado. mode puede ser `'r'` cuando el fichero solo se leerá, `'w'` para solo escritura (un fichero existente con el mismo nombre se borrará) y `'a'` abre el fichero para agregar.; cualquier dato que se escribe en el fichero se añade automáticamente al final. `'r+'` abre el fichero tanto para lectura como para escritura. El argumento mode es opcional; se asume que se usará `'r'` si se omite. 

Además de esos dos argumentos, también le podemos pasar otros argumentos importantes como `encoding` por ejemplo.

Al usar el primer método, no tenemos porque cerrar el archivo expicitamente porque con el `with` Python se encarga de cerrarlo. En cambio, si usamos el segundo método tenemos que cerrarlo nosotros con el método `close()`.

Con el método `write` hemos añadido el texto al fichero que hemos abierto con un modo que nos permite escribir en el.

Para leer el contenido del archivo usamos el método `read()`.


```python
f = open("texto.txt", encoding="UTF-8")
text = f.read(10)
text
```




    'Mi nombre '




```python
text = f.read(20)
text
```




    'tengo 22 años\nMi nom'




```python
f.close()
```


```python
with open("texto.txt", encoding="UTF-8") as f:
    text = f.read()
text
```




    'Mi nombre es Juan y tengo 22 años\nMi nombre es Ana y tengo 23 años'




```python
print(text)
```

    Mi nombre es Juan y tengo 22 años
    Mi nombre es Ana y tengo 23 años
    


```python
with open("texto.txt") as f:
    for i, line in enumerate(f):
        print(f"{i+1}ª linea: {line}")
```

    1ª linea: Mi nombre es Juan y tengo 22 aÃ±os
    
    2ª linea: Mi nombre es Ana y tengo 23 aÃ±os
    


```python
f = open("texto.txt")
f.readline()
```




    'Mi nombre es Juan y tengo 22 aÃ±os\n'




```python
dir(f)
```




    ['_CHUNK_SIZE',
     '__class__',
     '__del__',
     '__delattr__',
     '__dict__',
     '__dir__',
     '__doc__',
     '__enter__',
     '__eq__',
     '__exit__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getstate__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__lt__',
     '__ne__',
     '__new__',
     '__next__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     '_checkClosed',
     '_checkReadable',
     '_checkSeekable',
     '_checkWritable',
     '_finalizing',
     'buffer',
     'close',
     'closed',
     'detach',
     'encoding',
     'errors',
     'fileno',
     'flush',
     'isatty',
     'line_buffering',
     'mode',
     'name',
     'newlines',
     'read',
     'readable',
     'readline',
     'readlines',
     'reconfigure',
     'seek',
     'seekable',
     'tell',
     'truncate',
     'writable',
     'write',
     'write_through',
     'writelines']




```python
f.readline()
```




    'Mi nombre es Ana y tengo 23 aÃ±os'




```python
f.close()
```

¡IMPORTANTE!
No reeinventeis la rueda. Si vais a leer un tipo de archivo estructurado para el que ya existen funciones programadas en Python para leerlo, usar estas funciones y no os compliquéis la cabeza.

Algunas librerías para trabajar con diferentes tipos de archivos:
* wave (audio)
* aifc (audio)
* tarfile
* zipfile
* xml.etree.ElementTree
* PyPDF2
* xlwings (Excel)
* Pillow (imágenes)

# Módulo `pickle`

Pickle se utiliza para serializar y des-serializar las estructuras de los objetos de Python. 

Pickle es muy útil para cuando se trabaja con algoritmos de aprendizaje automático, en los que se requiere guardar los modelos para poder hacer nuevas predicciones más adelante, sin tener que reescribir todo o entrenar el modelo de nuevo.


```python
import pickle
```


```python
def preprocesamiento(x):
    return x/10
```


```python
def classificador(x):
    if x < 0:
        return 0
    else:
        return 1
```


```python
modelo = { 'preprocess': preprocesamiento, 'model': classificador, 'accuracy': 0.9}
```


```python
modelo['preprocess'](20)
```




    2.0




```python
filename = 'modelo.pickle'
outfile = open(filename,'wb')
pickle.dump(modelo, outfile)
outfile.close()
```


```python
infile = open(filename,'rb')
new_dict = pickle.load(infile)
infile.close()
```


```python
new_dict
```




    {'preprocess': <function __main__.preprocesamiento(x)>,
     'model': <function __main__.classificador(x)>,
     'accuracy': 0.9}




```python
new_dict['model'](-2)
```




    0



# `try\except`

En Python podemos controlar los errores que sabemos de antemano que pueden ocurrir en nuestros programas. Podeís encontrar una lista de errores definidos en Python [aquí](https://docs.python.org/es/3.7/library/exceptions.html#bltin-exceptions).


```python
2/0
```


    ---------------------------------------------------------------------------

    ZeroDivisionError                         Traceback (most recent call last)

    <ipython-input-85-e8326a161779> in <module>
    ----> 1 2/0
    

    ZeroDivisionError: division by zero



```python
2 + "a"
```


```python
while True:
    try:
        n = int(input("Elige un número entero: "))
        print(f"Tu número entero es : {n}")
        break
    except ValueError:
        print("Vuelve a intentarlo...")
    except KeyboardInterrupt:
        print("Saliendo...")
        break
```

    Elige un número entero: 2
    Tu número entero es : 2
    

Podemos definir nuestros propios errores.


```python
class Error(Exception):
    """Base class for other exceptions"""
    pass


class ValueTooSmallError(Error):
    """Raised when the input value is too small"""
    pass


class ValueTooLargeError(Error):
    """Raised when the input value is too large"""
    pass
```

`raise` se utiliza para devolver errores


```python
# numero que quermos predecir
number = 10

# el usuario dice un numero y le decimos si el nuestro es mayor o menor para que lo intente adivinar
while True:
    try:
        i_num = int(input("Enter a number: "))
        if i_num < number:
            raise ValueTooSmallError
        elif i_num > number:
            raise ValueTooLargeError
        break
    except ValueTooSmallError:
        print("This value is too small, try again!")
        print()
    except ValueTooLargeError:
        print("This value is too large, try again!")
        print()

print("Congratulations! You guessed it correctly.")
```

    Enter a number: 10
    Congratulations! You guessed it correctly.
    

`else` y `finally`:


```python
x = 0

try:
    10/x

except ZeroDivisionError:
    print("Has dividido por cero")
except:
    print("El error ha sido otro")
else:
    print("No ha habido error de dvidir entre 0")
    

finally:
    print("Lo has intentado")
```

    Has dividido por cero
    Lo has intentado
    

# Buenas prácticas con Python

El Zen de Python (PEP 20) es una colección de 20 () principios de software que influyen en el diseño del Lenguaje de Programación Python:


```python
from pandas import read_csv
```


```python
import this
```

    The Zen of Python, by Tim Peters
    
    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!
    

En [este enlace](https://pybaq.co/blog/el-zen-de-python-explicado/) podeis encontrar explicado cada principio.

El [PEP 8](https://www.python.org/dev/peps/pep-0008/) proporciona la guía de estilo para código de Python.

### Algunas curiosidades y funcionalidades útiles:

* Enumerate:


```python
z = [ 'a', 'b', 'c', 'd' ]

i = 0
while i < len(z):
    print(i, z[i])
    i += 1
```

    1 b
    2 c
    3 d
    


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-101-89c49a98421e> in <module>
          4 while i < len(z):
          5     i += 1
    ----> 6     print(i, z[i])
          7 
    

    IndexError: list index out of range



```python
for i in range(0, len(z)):
    print(i, z[i])
```

    0 a
    1 b
    2 c
    3 d
    


```python
for i, item in enumerate(z):
    print(i, item)
```

    0 a
    1 b
    2 c
    3 d
    


```python
?enumerate
```


```python
list(enumerate(z))
```




    [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')]



* zip


```python
z_inv = ['z', 'y', 'x', 'w', 'v']
z_inv
```




    ['z', 'y', 'x', 'w', 'v']




```python
for i in range(len(z_inv)):
    print(z[i], z_inv[i])
```

    a z
    b y
    c x
    d w
    


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-108-a1716adda375> in <module>
          1 for i in range(len(z_inv)):
    ----> 2     print(z[i], z_inv[i])
    

    IndexError: list index out of range



```python
for i, item in zip(z, z_inv):
    print(i, item)
```

    a z
    b y
    c x
    d w
    


```python
?zip
```


```python
list(zip(z, z_inv))
```




    [('a', 'z'), ('b', 'y'), ('c', 'x'), ('d', 'w')]



* itertools: Esto ya es un módulo propio con diferentes métodos.


```python
import itertools
```


```python
dir(itertools)
```




    ['__doc__',
     '__loader__',
     '__name__',
     '__package__',
     '__spec__',
     '_grouper',
     '_tee',
     '_tee_dataobject',
     'accumulate',
     'chain',
     'combinations',
     'combinations_with_replacement',
     'compress',
     'count',
     'cycle',
     'dropwhile',
     'filterfalse',
     'groupby',
     'islice',
     'permutations',
     'product',
     'repeat',
     'starmap',
     'takewhile',
     'tee',
     'zip_longest']




```python
abc = ['a', 'b', 'c', 'd', 'e']
num = [1, 2, 3, 4]
```


```python
l = []
cont = 0
for elem in num:
    cont += elem
    l.append(cont)
```


```python
list(itertools.accumulate(num))
```




    [1, 3, 6, 10]




```python
for comb in itertool.com
```


```python
list(itertools.combinations(abc, 5))
```




    [('a', 'b', 'c', 'd', 'e')]




```python
list(itertools.permutations(num))
```




    [(1, 2, 3, 4),
     (1, 2, 4, 3),
     (1, 3, 2, 4),
     (1, 3, 4, 2),
     (1, 4, 2, 3),
     (1, 4, 3, 2),
     (2, 1, 3, 4),
     (2, 1, 4, 3),
     (2, 3, 1, 4),
     (2, 3, 4, 1),
     (2, 4, 1, 3),
     (2, 4, 3, 1),
     (3, 1, 2, 4),
     (3, 1, 4, 2),
     (3, 2, 1, 4),
     (3, 2, 4, 1),
     (3, 4, 1, 2),
     (3, 4, 2, 1),
     (4, 1, 2, 3),
     (4, 1, 3, 2),
     (4, 2, 1, 3),
     (4, 2, 3, 1),
     (4, 3, 1, 2),
     (4, 3, 2, 1)]




```python
list(itertools.product(num, abc))
```




    [(1, 'a'),
     (1, 'b'),
     (1, 'c'),
     (1, 'd'),
     (1, 'e'),
     (2, 'a'),
     (2, 'b'),
     (2, 'c'),
     (2, 'd'),
     (2, 'e'),
     (3, 'a'),
     (3, 'b'),
     (3, 'c'),
     (3, 'd'),
     (3, 'e'),
     (4, 'a'),
     (4, 'b'),
     (4, 'c'),
     (4, 'd'),
     (4, 'e')]




```python
for number, letter in itertools.product(num, abc):
    print(number, letter)
```

    1 a
    1 b
    1 c
    1 d
    1 e
    2 a
    2 b
    2 c
    2 d
    2 e
    3 a
    3 b
    3 c
    3 d
    3 e
    4 a
    4 b
    4 c
    4 d
    4 e
    

* List comprehension:


```python
z = []

for i in range(0, 5):
    if i%2 == 0:
        z.append(i**2)
        np.random.randn(i,i)
    
z
```




    [0, 4, 16]




```python
z = [[] for i in range(0, 5)]
z
```




    [[], [], [], [], []]




```python
z = [ i**2 for i in range(0, 10) if i % 2 == 0 elif ]
z
```




    [0, 4, 16, 36, 64]



* Dict comprehension:


```python
d = {'a': 1, 'b': 2, 'c': 3}
d
```




    {'a': 1, 'b': 2, 'c': 3}




```python
d_inv = {valor:llave for llave, valor in d.items()}
d_inv
```




    {1: 'a', 2: 'b', 3: 'c'}



* La barra baja `_`: Si no vamos a utilizar una variable, se pone la barra baja para no gastar memoria


```python
a, b = (1, 2)
print(a)
```

    1
    


```python
a, _ = (1, 2)
print(a)
```

    1
    

Y cuando no sabemos cuantas variables va a tener el objeto que nos van a devolver usamos `*`:


```python
a, b = (1, 2, 3, 4, 5)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-132-1d8a5f8a6c69> in <module>
    ----> 1 a, b = (1, 2, 3, 4, 5)
    

    ValueError: too many values to unpack (expected 2)



```python
a, b, *c = (1, 2, 3, 4, 5)
print(a)
print(b)
print(c)
```

    1
    2
    [3, 4, 5]
    


```python
a, b, *_ = (1, 2, 3, 4, 5)
print(a)
print(b)
```

    1
    2
    


```python
a, b, *c, d = (1, 2, 3, 4, 5)
print(a)
print(b)
print(c)
print(d)
```

    1
    2
    [3, 4]
    5
    

Estos conceptos son parecidos a los de `*args` y `**kwargs` de como argumentos de funciones en Python.

### `lambda`, `map` y `filter`

`lambda` se usa para crear funciones pequeñas sin nombre, para usar en la ejecución del programa. Se suele utilizar en conjunto con `map` y `filter`.


```python
suma = lambda x, y: x + y
```


```python
suma(3, 4)
```




    7




```python
?map
```


```python
list(map(lambda x: x**2, [1, 2, 3]))
```




    [1, 4, 9]




```python
for i in  map(lambda x: x**2, [1, 2, 3]):
    print(i)
```

    1
    4
    9
    


```python
for i in map(lambda x,y: x + y, [1, 2, 3], [4, 5, 6]):
    print(i)
```

    5
    7
    9
    


```python
m = map(lambda x: x**2, [1,2,3])
```


```python
m[1]
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-145-c36dbd6b589c> in <module>
    ----> 1 m[1]
    

    TypeError: 'map' object is not subscriptable



```python
for i in m:
    print(i)
    break
```

    1
    


```python
list(m)
```




    [4, 9]




```python
list(m)
```




    []




```python
?filter
```


```python
for i in filter(lambda x: x%2 == 0, [1,2,3,4,5,6,7,8,9]):
    print(i)
```

    2
    4
    6
    8
    


```python

```
