---
# Title, summary, and page position.
linktitle: Sesión 02
weight: 1
icon: book
icon_pack: fas

# Page metadata.
title: Scientific computing in Python
type: book  # Do not modify.
---

{{< youtube GbfydqUdICw >}}
<br>
{{< youtube qkSMqwg8ldg >}}

### Instalar paquetes/librerias

Los paquetes que vamos a usar en este curso ya los tenenemos instalados con Anaconda en el entorno `base`. Para ver que paquetes tenemos instalados en el entorno `base` podemos mirarlo por el entorno gráfico que nos proporciona Anaconda, o a traves de la linea de comandos con el siguiente comando:

```
conda list
```

Para instalar paquetes que no vengan ya de serie, de nuevo dos opciones, usar la interfaz gráfica o la linea de comandos con alguno de los siguientes comandos:
```
conda install scikit-learn
pip install scikit-learn
```

## Numpy

* Numpy (numerical python) es uno de los paquetes más importantes de Python. 

* Muchos otros paquetes usan funcionalidades de este paquete de base. Por ese motivo, es importante conocer los conceptos básicos de Numpy.

* Numpy tiene un tipo de estructura especifico denominado `ndarray` que hace referencia a un vector/matríz N-dimensional.

Para importar el módulo Numpy a la sesión de Python en la que estamos trabajando:


```python
import numpy as np
```

Importante: Los import de los módulos se suelen hacer al principio en los script de python.

Podemos ver todos los métodods asociados que tiene Numpy con la función `dir`:


```python
dir(np)
```




    ['ALLOW_THREADS',
     'AxisError',
     'BUFSIZE',
     'CLIP',
     'ComplexWarning',
     'DataSource',
     'ERR_CALL',
     'ERR_DEFAULT',
     'ERR_IGNORE',
     'ERR_LOG',
     'ERR_PRINT',
     'ERR_RAISE',
     'ERR_WARN',
     'FLOATING_POINT_SUPPORT',
     'FPE_DIVIDEBYZERO',
     'FPE_INVALID',
     'FPE_OVERFLOW',
     'FPE_UNDERFLOW',
     'False_',
     'Inf',
     'Infinity',
     'MAXDIMS',
     'MAY_SHARE_BOUNDS',
     'MAY_SHARE_EXACT',
     'MachAr',
     'ModuleDeprecationWarning',
     'NAN',
     'NINF',
     'NZERO',
     'NaN',
     'PINF',
     'PZERO',
     'RAISE',
     'RankWarning',
     'SHIFT_DIVIDEBYZERO',
     'SHIFT_INVALID',
     'SHIFT_OVERFLOW',
     'SHIFT_UNDERFLOW',
     'ScalarType',
     'Tester',
     'TooHardError',
     'True_',
     'UFUNC_BUFSIZE_DEFAULT',
     'UFUNC_PYVALS_NAME',
     'VisibleDeprecationWarning',
     'WRAP',
     '_NoValue',
     '_UFUNC_API',
     '__NUMPY_SETUP__',
     '__all__',
     '__builtins__',
     '__cached__',
     '__config__',
     '__dir__',
     '__doc__',
     '__file__',
     '__getattr__',
     '__git_revision__',
     '__loader__',
     '__mkl_version__',
     '__name__',
     '__package__',
     '__path__',
     '__spec__',
     '__version__',
     '_add_newdoc_ufunc',
     '_distributor_init',
     '_globals',
     '_mat',
     '_pytesttester',
     'abs',
     'absolute',
     'add',
     'add_docstring',
     'add_newdoc',
     'add_newdoc_ufunc',
     'alen',
     'all',
     'allclose',
     'alltrue',
     'amax',
     'amin',
     'angle',
     'any',
     'append',
     'apply_along_axis',
     'apply_over_axes',
     'arange',
     'arccos',
     'arccosh',
     'arcsin',
     'arcsinh',
     'arctan',
     'arctan2',
     'arctanh',
     'argmax',
     'argmin',
     'argpartition',
     'argsort',
     'argwhere',
     'around',
     'array',
     'array2string',
     'array_equal',
     'array_equiv',
     'array_repr',
     'array_split',
     'array_str',
     'asanyarray',
     'asarray',
     'asarray_chkfinite',
     'ascontiguousarray',
     'asfarray',
     'asfortranarray',
     'asmatrix',
     'asscalar',
     'atleast_1d',
     'atleast_2d',
     'atleast_3d',
     'average',
     'bartlett',
     'base_repr',
     'binary_repr',
     'bincount',
     'bitwise_and',
     'bitwise_not',
     'bitwise_or',
     'bitwise_xor',
     'blackman',
     'block',
     'bmat',
     'bool',
     'bool8',
     'bool_',
     'broadcast',
     'broadcast_arrays',
     'broadcast_to',
     'busday_count',
     'busday_offset',
     'busdaycalendar',
     'byte',
     'byte_bounds',
     'bytes0',
     'bytes_',
     'c_',
     'can_cast',
     'cast',
     'cbrt',
     'cdouble',
     'ceil',
     'cfloat',
     'char',
     'character',
     'chararray',
     'choose',
     'clip',
     'clongdouble',
     'clongfloat',
     'column_stack',
     'common_type',
     'compare_chararrays',
     'compat',
     'complex',
     'complex128',
     'complex64',
     'complex_',
     'complexfloating',
     'compress',
     'concatenate',
     'conj',
     'conjugate',
     'convolve',
     'copy',
     'copysign',
     'copyto',
     'core',
     'corrcoef',
     'correlate',
     'cos',
     'cosh',
     'count_nonzero',
     'cov',
     'cross',
     'csingle',
     'ctypeslib',
     'cumprod',
     'cumproduct',
     'cumsum',
     'datetime64',
     'datetime_as_string',
     'datetime_data',
     'deg2rad',
     'degrees',
     'delete',
     'deprecate',
     'deprecate_with_doc',
     'diag',
     'diag_indices',
     'diag_indices_from',
     'diagflat',
     'diagonal',
     'diff',
     'digitize',
     'disp',
     'divide',
     'divmod',
     'dot',
     'double',
     'dsplit',
     'dstack',
     'dtype',
     'e',
     'ediff1d',
     'einsum',
     'einsum_path',
     'emath',
     'empty',
     'empty_like',
     'equal',
     'errstate',
     'euler_gamma',
     'exp',
     'exp2',
     'expand_dims',
     'expm1',
     'extract',
     'eye',
     'fabs',
     'fastCopyAndTranspose',
     'fft',
     'fill_diagonal',
     'find_common_type',
     'finfo',
     'fix',
     'flatiter',
     'flatnonzero',
     'flexible',
     'flip',
     'fliplr',
     'flipud',
     'float',
     'float16',
     'float32',
     'float64',
     'float_',
     'float_power',
     'floating',
     'floor',
     'floor_divide',
     'fmax',
     'fmin',
     'fmod',
     'format_float_positional',
     'format_float_scientific',
     'format_parser',
     'frexp',
     'frombuffer',
     'fromfile',
     'fromfunction',
     'fromiter',
     'frompyfunc',
     'fromregex',
     'fromstring',
     'full',
     'full_like',
     'fv',
     'gcd',
     'generic',
     'genfromtxt',
     'geomspace',
     'get_array_wrap',
     'get_include',
     'get_printoptions',
     'getbufsize',
     'geterr',
     'geterrcall',
     'geterrobj',
     'gradient',
     'greater',
     'greater_equal',
     'half',
     'hamming',
     'hanning',
     'heaviside',
     'histogram',
     'histogram2d',
     'histogram_bin_edges',
     'histogramdd',
     'hsplit',
     'hstack',
     'hypot',
     'i0',
     'identity',
     'iinfo',
     'imag',
     'in1d',
     'index_exp',
     'indices',
     'inexact',
     'inf',
     'info',
     'infty',
     'inner',
     'insert',
     'int',
     'int0',
     'int16',
     'int32',
     'int64',
     'int8',
     'int_',
     'intc',
     'integer',
     'interp',
     'intersect1d',
     'intp',
     'invert',
     'ipmt',
     'irr',
     'is_busday',
     'isclose',
     'iscomplex',
     'iscomplexobj',
     'isfinite',
     'isfortran',
     'isin',
     'isinf',
     'isnan',
     'isnat',
     'isneginf',
     'isposinf',
     'isreal',
     'isrealobj',
     'isscalar',
     'issctype',
     'issubclass_',
     'issubdtype',
     'issubsctype',
     'iterable',
     'ix_',
     'kaiser',
     'kron',
     'lcm',
     'ldexp',
     'left_shift',
     'less',
     'less_equal',
     'lexsort',
     'lib',
     'linalg',
     'linspace',
     'little_endian',
     'load',
     'loads',
     'loadtxt',
     'log',
     'log10',
     'log1p',
     'log2',
     'logaddexp',
     'logaddexp2',
     'logical_and',
     'logical_not',
     'logical_or',
     'logical_xor',
     'logspace',
     'long',
     'longcomplex',
     'longdouble',
     'longfloat',
     'longlong',
     'lookfor',
     'ma',
     'mafromtxt',
     'mask_indices',
     'mat',
     'math',
     'matmul',
     'matrix',
     'matrixlib',
     'max',
     'maximum',
     'maximum_sctype',
     'may_share_memory',
     'mean',
     'median',
     'memmap',
     'meshgrid',
     'mgrid',
     'min',
     'min_scalar_type',
     'minimum',
     'mintypecode',
     'mirr',
     'mkl',
     'mod',
     'modf',
     'moveaxis',
     'msort',
     'multiply',
     'nan',
     'nan_to_num',
     'nanargmax',
     'nanargmin',
     'nancumprod',
     'nancumsum',
     'nanmax',
     'nanmean',
     'nanmedian',
     'nanmin',
     'nanpercentile',
     'nanprod',
     'nanquantile',
     'nanstd',
     'nansum',
     'nanvar',
     'nbytes',
     'ndarray',
     'ndenumerate',
     'ndfromtxt',
     'ndim',
     'ndindex',
     'nditer',
     'negative',
     'nested_iters',
     'newaxis',
     'nextafter',
     'nonzero',
     'not_equal',
     'nper',
     'npv',
     'numarray',
     'number',
     'obj2sctype',
     'object',
     'object0',
     'object_',
     'ogrid',
     'oldnumeric',
     'ones',
     'ones_like',
     'os',
     'outer',
     'packbits',
     'pad',
     'partition',
     'percentile',
     'pi',
     'piecewise',
     'place',
     'pmt',
     'poly',
     'poly1d',
     'polyadd',
     'polyder',
     'polydiv',
     'polyfit',
     'polyint',
     'polymul',
     'polynomial',
     'polysub',
     'polyval',
     'positive',
     'power',
     'ppmt',
     'printoptions',
     'prod',
     'product',
     'promote_types',
     'ptp',
     'put',
     'put_along_axis',
     'putmask',
     'pv',
     'quantile',
     'r_',
     'rad2deg',
     'radians',
     'random',
     'rate',
     'ravel',
     'ravel_multi_index',
     'real',
     'real_if_close',
     'rec',
     'recarray',
     'recfromcsv',
     'recfromtxt',
     'reciprocal',
     'record',
     'remainder',
     'repeat',
     'require',
     'reshape',
     'resize',
     'result_type',
     'right_shift',
     'rint',
     'roll',
     'rollaxis',
     'roots',
     'rot90',
     'round',
     'round_',
     'row_stack',
     's_',
     'safe_eval',
     'save',
     'savetxt',
     'savez',
     'savez_compressed',
     'sctype2char',
     'sctypeDict',
     'sctypeNA',
     'sctypes',
     'searchsorted',
     'select',
     'set_numeric_ops',
     'set_printoptions',
     'set_string_function',
     'setbufsize',
     'setdiff1d',
     'seterr',
     'seterrcall',
     'seterrobj',
     'setxor1d',
     'shape',
     'shares_memory',
     'short',
     'show_config',
     'sign',
     'signbit',
     'signedinteger',
     'sin',
     'sinc',
     'single',
     'singlecomplex',
     'sinh',
     'size',
     'sometrue',
     'sort',
     'sort_complex',
     'source',
     'spacing',
     'split',
     'sqrt',
     'square',
     'squeeze',
     'stack',
     'std',
     'str',
     'str0',
     'str_',
     'string_',
     'subtract',
     'sum',
     'swapaxes',
     'sys',
     'take',
     'take_along_axis',
     'tan',
     'tanh',
     'tensordot',
     'test',
     'testing',
     'tile',
     'timedelta64',
     'trace',
     'tracemalloc_domain',
     'transpose',
     'trapz',
     'tri',
     'tril',
     'tril_indices',
     'tril_indices_from',
     'trim_zeros',
     'triu',
     'triu_indices',
     'triu_indices_from',
     'true_divide',
     'trunc',
     'typeDict',
     'typeNA',
     'typecodes',
     'typename',
     'ubyte',
     'ufunc',
     'uint',
     'uint0',
     'uint16',
     'uint32',
     'uint64',
     'uint8',
     'uintc',
     'uintp',
     'ulonglong',
     'unicode',
     'unicode_',
     'union1d',
     'unique',
     'unpackbits',
     'unravel_index',
     'unsignedinteger',
     'unwrap',
     'use_hugepage',
     'ushort',
     'vander',
     'var',
     'vdot',
     'vectorize',
     'version',
     'void',
     'void0',
     'vsplit',
     'vstack',
     'warnings',
     'where',
     'who',
     'zeros',
     'zeros_like']



Defininimos un `ndarray` con la función `np.array`:


```python
a = np.array([1, 2, 3])
a
```




    array([1, 2, 3])




```python
type(a)
```




    numpy.ndarray




```python
dir(a)
```




    ['T',
     '__abs__',
     '__add__',
     '__and__',
     '__array__',
     '__array_finalize__',
     '__array_function__',
     '__array_interface__',
     '__array_prepare__',
     '__array_priority__',
     '__array_struct__',
     '__array_ufunc__',
     '__array_wrap__',
     '__bool__',
     '__class__',
     '__complex__',
     '__contains__',
     '__copy__',
     '__deepcopy__',
     '__delattr__',
     '__delitem__',
     '__dir__',
     '__divmod__',
     '__doc__',
     '__eq__',
     '__float__',
     '__floordiv__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__gt__',
     '__hash__',
     '__iadd__',
     '__iand__',
     '__ifloordiv__',
     '__ilshift__',
     '__imatmul__',
     '__imod__',
     '__imul__',
     '__index__',
     '__init__',
     '__init_subclass__',
     '__int__',
     '__invert__',
     '__ior__',
     '__ipow__',
     '__irshift__',
     '__isub__',
     '__iter__',
     '__itruediv__',
     '__ixor__',
     '__le__',
     '__len__',
     '__lshift__',
     '__lt__',
     '__matmul__',
     '__mod__',
     '__mul__',
     '__ne__',
     '__neg__',
     '__new__',
     '__or__',
     '__pos__',
     '__pow__',
     '__radd__',
     '__rand__',
     '__rdivmod__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__rfloordiv__',
     '__rlshift__',
     '__rmatmul__',
     '__rmod__',
     '__rmul__',
     '__ror__',
     '__rpow__',
     '__rrshift__',
     '__rshift__',
     '__rsub__',
     '__rtruediv__',
     '__rxor__',
     '__setattr__',
     '__setitem__',
     '__setstate__',
     '__sizeof__',
     '__str__',
     '__sub__',
     '__subclasshook__',
     '__truediv__',
     '__xor__',
     'all',
     'any',
     'argmax',
     'argmin',
     'argpartition',
     'argsort',
     'astype',
     'base',
     'byteswap',
     'choose',
     'clip',
     'compress',
     'conj',
     'conjugate',
     'copy',
     'ctypes',
     'cumprod',
     'cumsum',
     'data',
     'diagonal',
     'dot',
     'dtype',
     'dump',
     'dumps',
     'fill',
     'flags',
     'flat',
     'flatten',
     'getfield',
     'imag',
     'item',
     'itemset',
     'itemsize',
     'max',
     'mean',
     'min',
     'nbytes',
     'ndim',
     'newbyteorder',
     'nonzero',
     'partition',
     'prod',
     'ptp',
     'put',
     'ravel',
     'real',
     'repeat',
     'reshape',
     'resize',
     'round',
     'searchsorted',
     'setfield',
     'setflags',
     'shape',
     'size',
     'sort',
     'squeeze',
     'std',
     'strides',
     'sum',
     'swapaxes',
     'take',
     'tobytes',
     'tofile',
     'tolist',
     'tostring',
     'trace',
     'transpose',
     'var',
     'view']



Cada `ndarray` está asociado a un tipo de dato (float, float32, float64, int, ...) y todos los objetos que lo forman tienen que ser de ese mismo tipo. Podemos ver que tipo de dato esta asociado a un `ndarray` con `dtype`:


```python
a.dtype
```




    dtype('int32')



Vamos a definir ahora una bi dimensional: 


```python
b = np.array([[1.3, 2.4],[0.3, 4.1]])
b
```




    array([[1.3, 2.4],
           [0.3, 4.1]])




```python
b.dtype
```




    dtype('float64')




```python
b.shape
```




    (2, 2)




```python
b.ndim
```




    2




```python
b.size
```




    4



Podemos definir `ndarray`s con más tipos de elementos:


```python
c = np.array([['a', 'b'],['c', 'd']])
c
```




    array([['a', 'b'],
           ['c', 'd']], dtype='<U1')




```python
d = np.array([[1, 2, 3],[4, 5, 6]], dtype=complex)
d
```




    array([[1.+0.j, 2.+0.j, 3.+0.j],
           [4.+0.j, 5.+0.j, 6.+0.j]])



### Diferentes tipos de funciones para crear `ndarrays`:


```python
np.zeros((3, 3))
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])




```python
np.ones((3, 3))
```




    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])




```python
np.arange(0, 10)
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.arange(4, 10)
```




    array([4, 5, 6, 7, 8, 9])




```python
np.arange(0, 12, 3)
```




    array([0, 3, 6, 9])




```python
np.arange(0, 6, 0.6)
```




    array([0. , 0.6, 1.2, 1.8, 2.4, 3. , 3.6, 4.2, 4.8, 5.4])




```python
np.linspace(0, 10, 5)
```




    array([ 0. ,  2.5,  5. ,  7.5, 10. ])




```python
np.random.random(3)
```




    array([0.43122818, 0.29135981, 0.82425344])




```python
np.random.random((3, 3))
```




    array([[0.80545447, 0.39337949, 0.46895882],
           [0.37764337, 0.71051177, 0.45382529],
           [0.19417507, 0.16479578, 0.74048171]])



La función `reshape`:


```python
np.arange(0,12).reshape(3, 4)
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])



### Operaciones aritméticas


```python
a = np.arange(4)
a
```




    array([0, 1, 2, 3])




```python
a+4
```




    array([4, 5, 6, 7])




```python
a*2
```




    array([0, 2, 4, 6])




```python
b = np.arange(4, 8)
b
```




    array([4, 5, 6, 7])




```python
a + b
```




    array([ 4,  6,  8, 10])




```python
a - b
```




    array([-4, -4, -4, -4])




```python
a * b
```




    array([ 0,  5, 12, 21])




```python
A = np.arange(0, 9).reshape(3, 3)
A
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
B = np.ones((3, 3))
B
```




    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])




```python
A * B
```




    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])



### Multiplicación de matrices

Hasta ahora, solo hemos hecho operaciones por elementos de los `ndarray`s. Ahora vamos a ver otro tipo de operaciones, como la multiplicación de matrices.


```python
np.dot(A, B)
```




    array([[ 3.,  3.,  3.],
           [12., 12., 12.],
           [21., 21., 21.]])




```python
A.dot(B)
```




    array([[ 3.,  3.,  3.],
           [12., 12., 12.],
           [21., 21., 21.]])




```python
np.dot(B, A)
```




    array([[ 9., 12., 15.],
           [ 9., 12., 15.],
           [ 9., 12., 15.]])



### Más funciones para las `ndarrays`


```python
a = np.arange(1, 10)
a
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.sqrt(a)
```




    array([1.        , 1.41421356, 1.73205081, 2.        , 2.23606798,
           2.44948974, 2.64575131, 2.82842712, 3.        ])




```python
np.log(a)
```




    array([0.        , 0.69314718, 1.09861229, 1.38629436, 1.60943791,
           1.79175947, 1.94591015, 2.07944154, 2.19722458])




```python
np.sin(a)
```




    array([ 0.84147098,  0.90929743,  0.14112001, -0.7568025 , -0.95892427,
           -0.2794155 ,  0.6569866 ,  0.98935825,  0.41211849])




```python
a.sum()
```




    45




```python
a.min()
```




    1




```python
a.max()
```




    9




```python
a.mean()
```




    5.0




```python
a.std()
```




    2.581988897471611



### Manipular vectores y matrices


```python
a = np.arange(1, 10)
a
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
a[4]
```




    5




```python
a[-4]
```




    6




```python
a[:5]
```




    array([1, 2, 3, 4, 5])




```python
a[[1,2,8]]
```




    array([2, 3, 9])




```python
A = np.arange(1, 10).reshape((3, 3))
A
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
A[1, 2]
```




    6




```python
A[:,0]
```




    array([1, 4, 7])




```python
A[0:2, 0:2]
```




    array([[1, 2],
           [4, 5]])




```python
A[[0,2], 0:2]
```




    array([[1, 2],
           [7, 8]])



### Iterando un array


```python
a
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
for i in a:
    print(i)
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
    


```python
A
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
for row in A:
    print(row)
```

    [1 2 3]
    [4 5 6]
    [7 8 9]
    


```python
for i in A.flat:
    print(i)
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
    

Si necesitamos aplicar una función en las columnas o filas de una matriz, hay una forma más elegante y eficaz de hacerlo que usando un `for`.


```python
np.apply_along_axis(np.mean, axis=0, arr=A)
```




    array([4., 5., 6.])




```python
np.apply_along_axis(np.mean, axis=1, arr=A)
```




    array([2., 5., 8.])



### Condiciones y arrays de Booleanos


```python
A = np.random.random((4, 4))
A
```




    array([[0.85602406, 0.17439647, 0.82691213, 0.48981487],
           [0.9042295 , 0.99104976, 0.50913057, 0.15598115],
           [0.34064124, 0.97923316, 0.74593774, 0.53230547],
           [0.80176434, 0.45300678, 0.57819088, 0.14618446]])




```python
A < 0.5
```




    array([[False,  True, False,  True],
           [False, False, False,  True],
           [ True, False, False, False],
           [False,  True, False,  True]])




```python
A[A < 0.5]
```




    array([0.17439647, 0.48981487, 0.15598115, 0.34064124, 0.45300678,
           0.14618446])



### Unir arrays


```python
A = np.ones((3, 3))
B = np.zeros((3, 3))
```


```python
np.vstack((A, B))
```




    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.],
           [0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])




```python
np.hstack((A, B))
```




    array([[1., 1., 1., 0., 0., 0.],
           [1., 1., 1., 0., 0., 0.],
           [1., 1., 1., 0., 0., 0.]])



Funciones más especificas para unir arrays de una sola dimensión y crear así arrays bidimensionales:


```python
a = np.array([0, 1, 2])
b = np.array([3, 4, 5])
c = np.array([6, 7, 8])
```


```python
np.column_stack((a, b, c))
```




    array([[0, 3, 6],
           [1, 4, 7],
           [2, 5, 8]])




```python
np.row_stack((a, b, c))
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])



### Importante: Copía o vista de un elemento `ndarray`


```python
a = np.array([1, 2, 3, 4])
a
```




    array([1, 2, 3, 4])




```python
b = a
b
```




    array([1, 2, 3, 4])




```python
a[2] = 0
a
```




    array([1, 2, 0, 4])




```python
b
```




    array([1, 2, 0, 4])




```python
c = a[0:2]
c
```




    array([1, 2])




```python
a[0] = 0
c
```




    array([0, 2])



Evitamos esto usando la funcion `copy`.


```python
a = np.array([1, 2, 3, 4])
b = a.copy()
b
```




    array([1, 2, 3, 4])




```python
a[2] = 0
b
```




    array([1, 2, 3, 4])




```python
c = a[0:2].copy()
c
```




    array([1, 2])




```python
a[0] = 0
c
```




    array([1, 2])



## Pandas

* Pandas es el paquete de referencia para el análisis de datos en Python.

* Pandas proporciona estructuras de datos complejas y funciones especificas para trabajar con ellas.

* El concenpto fundamental de Pandas son los `DataFrame`, una estructura de datos con dos dimensiones. También están las `Series`, que son de una dimensión.

* Pandas usa Numpy


```python
import numpy as np
import pandas as pd
```

### DataFrame

Un DataFrame es basicamente una tabla. Esta formado por filas y columnas, que son arrays con valores individuales (pueden ser números o no).


```python
pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
```




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
      <th>Yes</th>
      <th>No</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>131</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame({'Juan': ['Sopa', 'Pescado'], 'Ana': ['Pasta', 'Solomillo']})
```




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
      <th>Juan</th>
      <th>Ana</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sopa</td>
      <td>Pasta</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pescado</td>
      <td>Solomillo</td>
    </tr>
  </tbody>
</table>
</div>



Estamos usando `pd.DataFrame()` para construir objetos `DataFrame`. Como argumento le pasamos un diccionario con los `keys` `['Juan', 'Ana']` y sus respectivos valores. Aunque este es el método más común para construir un objeto `DataFrame`, no es el único.

El método para construir `DataFrames` que hemos usado le asigna una etiqueta a cada columna que va desde el 0 hasta el número de columnas ascendentemente. Algunas veces esto está bien, pero otras veces puede que queramos asignar una etiqueta específica a cada columna.


```python
pd.DataFrame({'Juan': ['Sopa', 'Pescado', 'Yogurt'], 'Ana': ['Pasta', 'Solomillo', 'Fruta']}, index=['1 Plato', '2 Plato', 'Postre'])
```




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
      <th>Juan</th>
      <th>Ana</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 Plato</th>
      <td>Sopa</td>
      <td>Pasta</td>
    </tr>
    <tr>
      <th>2 Plato</th>
      <td>Pescado</td>
      <td>Solomillo</td>
    </tr>
    <tr>
      <th>Postre</th>
      <td>Yogurt</td>
      <td>Fruta</td>
    </tr>
  </tbody>
</table>
</div>



### Series

Las `Series` son una sequencia de datos. Si los `DataFrames` son tablas de datos, las `Series` son listas de datos.


```python
pd.Series([1, 2, 3, 4, 5])
```




    0    1
    1    2
    2    3
    3    4
    4    5
    dtype: int64




```python
pd.Series([30, 35, 40], index=['2015 matriculas', '2016 matriculas', '2017 matriculas'], name='Matriculas en máster de modelización')
```




    2015 matriculas    30
    2016 matriculas    35
    2017 matriculas    40
    Name: Matriculas en máster de modelización, dtype: int64



Las `Series` y los `DataFrames` están estrechamente relacionados. De hecho, podemos pensas que los `DataFrames` son simplemente un puñado de `Series` juntados.

### Leer ficheros de datos

Aunque exista la opción de crear los `DataFrames` y las `Series` a mano, lo más habitual va a ser que trabajemos con datos que ya existen y están recogidos en algún tipo de fichero (.xls, .csv, .json, ...)

El formato más habitual para guardar datos el el CSV. Los ficheros CSV contienen valores separados por comas.


```python
reviews = pd.read_csv("data/winemag-data-130k-v2.csv")
```


```python
reviews.shape
```




    (129971, 14)




```python
reviews.head()
```




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
      <th>Unnamed: 0</th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
    </tr>
  </tbody>
</table>
</div>




```python
reviews = pd.read_csv("data/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
    </tr>
  </tbody>
</table>
</div>



### Seleccionar subconjuntos del `DataFrame` o `Series`

Podemos seleccionar los valores de una o varias columnas de varias maneras.


```python
reviews.country
```




    0            Italy
    1         Portugal
    2               US
    3               US
    4               US
                ...   
    129966     Germany
    129967          US
    129968      France
    129969      France
    129970      France
    Name: country, Length: 129971, dtype: object




```python
reviews['country']
```




    0            Italy
    1         Portugal
    2               US
    3               US
    4               US
                ...   
    129966     Germany
    129967          US
    129968      France
    129969      France
    129970      France
    Name: country, Length: 129971, dtype: object




```python
reviews[['country', 'province']]
```




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
      <th>country</th>
      <th>province</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Sicily &amp; Sardinia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>Douro</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Oregon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Michigan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>Oregon</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>129966</th>
      <td>Germany</td>
      <td>Mosel</td>
    </tr>
    <tr>
      <th>129967</th>
      <td>US</td>
      <td>Oregon</td>
    </tr>
    <tr>
      <th>129968</th>
      <td>France</td>
      <td>Alsace</td>
    </tr>
    <tr>
      <th>129969</th>
      <td>France</td>
      <td>Alsace</td>
    </tr>
    <tr>
      <th>129970</th>
      <td>France</td>
      <td>Alsace</td>
    </tr>
  </tbody>
</table>
<p>129971 rows × 2 columns</p>
</div>




```python
reviews[['country', 'province']][:5]
```




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
      <th>country</th>
      <th>province</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Sicily &amp; Sardinia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>Douro</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Oregon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Michigan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>Oregon</td>
    </tr>
  </tbody>
</table>
</div>



También podemos usar los indices para seleccionar los subconjuntos usando el método `iloc`.


```python
reviews.iloc[0]
```




    country                                                              Italy
    description              Aromas include tropical fruit, broom, brimston...
    designation                                                   Vulkà Bianco
    points                                                                  87
    price                                                                  NaN
    province                                                 Sicily & Sardinia
    region_1                                                              Etna
    region_2                                                               NaN
    taster_name                                                  Kerin O’Keefe
    taster_twitter_handle                                         @kerinokeefe
    title                                    Nicosia 2013 Vulkà Bianco  (Etna)
    variety                                                        White Blend
    winery                                                             Nicosia
    Name: 0, dtype: object




```python
reviews.iloc[0,0]
```




    'Italy'




```python
reviews.iloc[:,-1]
```




    0                                          Nicosia
    1                              Quinta dos Avidagos
    2                                        Rainstorm
    3                                       St. Julian
    4                                     Sweet Cheeks
                                ...                   
    129966    Dr. H. Thanisch (Erben Müller-Burggraef)
    129967                                    Citation
    129968                             Domaine Gresser
    129969                        Domaine Marcel Deiss
    129970                            Domaine Schoffit
    Name: winery, Length: 129971, dtype: object




```python
reviews.iloc[-3:, :3]
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>129968</th>
      <td>France</td>
      <td>Well-drained gravel soil gives this wine its c...</td>
      <td>Kritt</td>
    </tr>
    <tr>
      <th>129969</th>
      <td>France</td>
      <td>A dry style of Pinot Gris, this is crisp with ...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>129970</th>
      <td>France</td>
      <td>Big, rich and off-dry, this is powered by inte...</td>
      <td>Lieu-dit Harth Cuvée Caroline</td>
    </tr>
  </tbody>
</table>
</div>




```python
reviews.iloc[[0, 10, 100], 0]
```




    0      Italy
    10        US
    100       US
    Name: country, dtype: object



Por último, tambien podemos usar el método `loc` para usar las etiquetas de las filas y columnas.


```python
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
```




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
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>87</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>87</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>129966</th>
      <td>Anna Lee C. Iijima</td>
      <td>NaN</td>
      <td>90</td>
    </tr>
    <tr>
      <th>129967</th>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>90</td>
    </tr>
    <tr>
      <th>129968</th>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>90</td>
    </tr>
    <tr>
      <th>129969</th>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>90</td>
    </tr>
    <tr>
      <th>129970</th>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
<p>129971 rows × 3 columns</p>
</div>



CUIDADO! En este caso, las etiquetas de las filas son números, pero `iloc` y `loc` no funciónan igual. 


```python
reviews.iloc[:5, 0]
```




    0       Italy
    1    Portugal
    2          US
    3          US
    4          US
    Name: country, dtype: object




```python
reviews.loc[:5, 'country']
```




    0       Italy
    1    Portugal
    2          US
    3          US
    4          US
    5       Spain
    Name: country, dtype: object



### Manipular el índice


```python
reviews.set_index("title")
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Nicosia 2013 Vulkà Bianco  (Etna)</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>Quinta dos Avidagos 2011 Avidagos Red (Douro)</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>Rainstorm 2013 Pinot Gris (Willamette Valley)</th>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>St. Julian 2013 Reserve Late Harvest Riesling (Lake Michigan Shore)</th>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>Sweet Cheeks 2012 Vintner's Reserve Wild Child Block Pinot Noir (Willamette Valley)</th>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
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
    </tr>
    <tr>
      <th>Dr. H. Thanisch (Erben Müller-Burggraef) 2013 Brauneberger Juffer-Sonnenuhr Spätlese Riesling (Mosel)</th>
      <td>Germany</td>
      <td>Notes of honeysuckle and cantaloupe sweeten th...</td>
      <td>Brauneberger Juffer-Sonnenuhr Spätlese</td>
      <td>90</td>
      <td>28.0</td>
      <td>Mosel</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Anna Lee C. Iijima</td>
      <td>NaN</td>
      <td>Riesling</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef)</td>
    </tr>
    <tr>
      <th>Citation 2004 Pinot Noir (Oregon)</th>
      <td>US</td>
      <td>Citation is given as much as a decade of bottl...</td>
      <td>NaN</td>
      <td>90</td>
      <td>75.0</td>
      <td>Oregon</td>
      <td>Oregon</td>
      <td>Oregon Other</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Pinot Noir</td>
      <td>Citation</td>
    </tr>
    <tr>
      <th>Domaine Gresser 2013 Kritt Gewurztraminer (Alsace)</th>
      <td>France</td>
      <td>Well-drained gravel soil gives this wine its c...</td>
      <td>Kritt</td>
      <td>90</td>
      <td>30.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Gewürztraminer</td>
      <td>Domaine Gresser</td>
    </tr>
    <tr>
      <th>Domaine Marcel Deiss 2012 Pinot Gris (Alsace)</th>
      <td>France</td>
      <td>A dry style of Pinot Gris, this is crisp with ...</td>
      <td>NaN</td>
      <td>90</td>
      <td>32.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Pinot Gris</td>
      <td>Domaine Marcel Deiss</td>
    </tr>
    <tr>
      <th>Domaine Schoffit 2012 Lieu-dit Harth Cuvée Caroline Gewurztraminer (Alsace)</th>
      <td>France</td>
      <td>Big, rich and off-dry, this is powered by inte...</td>
      <td>Lieu-dit Harth Cuvée Caroline</td>
      <td>90</td>
      <td>21.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Gewürztraminer</td>
      <td>Domaine Schoffit</td>
    </tr>
  </tbody>
</table>
<p>129971 rows × 12 columns</p>
</div>



### Selección condicional

Podemos buscar los vinos de Italia.


```python
reviews['country'] == 'Italy'
```




    0          True
    1         False
    2         False
    3         False
    4         False
              ...  
    129966    False
    129967    False
    129968    False
    129969    False
    129970    False
    Name: country, Length: 129971, dtype: bool



La anterior expresión nos ha devuelto una `Series` con los booleanos que nos dicen cuando el vino es Italiano. Para encontrar esas instancias devueltas por los booleanos hacemos:


```python
reviews.loc[reviews['country'] == 'Italy']
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Italy</td>
      <td>Here's a bright, informal red that opens with ...</td>
      <td>Belsito</td>
      <td>87</td>
      <td>16.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Vittoria</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Terre di Giurfo 2013 Belsito Frappato (Vittoria)</td>
      <td>Frappato</td>
      <td>Terre di Giurfo</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Italy</td>
      <td>This is dominated by oak and oak-driven aromas...</td>
      <td>Rosso</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Masseria Setteporte 2012 Rosso  (Etna)</td>
      <td>Nerello Mascalese</td>
      <td>Masseria Setteporte</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Italy</td>
      <td>Delicate aromas recall white flower and citrus...</td>
      <td>Ficiligno</td>
      <td>87</td>
      <td>19.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Baglio di Pianetto 2007 Ficiligno White (Sicilia)</td>
      <td>White Blend</td>
      <td>Baglio di Pianetto</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Italy</td>
      <td>Aromas of prune, blackcurrant, toast and oak c...</td>
      <td>Aynat</td>
      <td>87</td>
      <td>35.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Canicattì 2009 Aynat Nero d'Avola (Sicilia)</td>
      <td>Nero d'Avola</td>
      <td>Canicattì</td>
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
    </tr>
    <tr>
      <th>129929</th>
      <td>Italy</td>
      <td>This luminous sparkler has a sweet, fruit-forw...</td>
      <td>NaN</td>
      <td>91</td>
      <td>38.0</td>
      <td>Veneto</td>
      <td>Prosecco Superiore di Cartizze</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Col Vetoraz Spumanti NV  Prosecco Superiore di...</td>
      <td>Prosecco</td>
      <td>Col Vetoraz Spumanti</td>
    </tr>
    <tr>
      <th>129943</th>
      <td>Italy</td>
      <td>A blend of Nero d'Avola and Syrah, this convey...</td>
      <td>Adènzia</td>
      <td>90</td>
      <td>29.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Baglio del Cristo di Campobello 2012 Adènzia R...</td>
      <td>Red Blend</td>
      <td>Baglio del Cristo di Campobello</td>
    </tr>
    <tr>
      <th>129947</th>
      <td>Italy</td>
      <td>A blend of 65% Cabernet Sauvignon, 30% Merlot ...</td>
      <td>Symposio</td>
      <td>90</td>
      <td>20.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Terre Siciliane</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Feudo Principi di Butera 2012 Symposio Red (Te...</td>
      <td>Red Blend</td>
      <td>Feudo Principi di Butera</td>
    </tr>
    <tr>
      <th>129961</th>
      <td>Italy</td>
      <td>Intense aromas of wild cherry, baking spice, t...</td>
      <td>NaN</td>
      <td>90</td>
      <td>30.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>COS 2013 Frappato (Sicilia)</td>
      <td>Frappato</td>
      <td>COS</td>
    </tr>
    <tr>
      <th>129962</th>
      <td>Italy</td>
      <td>Blackberry, cassis, grilled herb and toasted a...</td>
      <td>Sàgana Tenuta San Giacomo</td>
      <td>90</td>
      <td>40.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Cusumano 2012 Sàgana Tenuta San Giacomo Nero d...</td>
      <td>Nero d'Avola</td>
      <td>Cusumano</td>
    </tr>
  </tbody>
</table>
<p>19540 rows × 13 columns</p>
</div>



Si además de que sea Italiano, también queremos que nuestro vino tenga una puntuación mayor o igual a 90:


```python
reviews.loc[(reviews['country'] == 'Italy') & (reviews['points'] >= 90)]
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>120</th>
      <td>Italy</td>
      <td>Slightly backward, particularly given the vint...</td>
      <td>Bricco Rocche Prapó</td>
      <td>92</td>
      <td>70.0</td>
      <td>Piedmont</td>
      <td>Barolo</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Ceretto 2003 Bricco Rocche Prapó  (Barolo)</td>
      <td>Nebbiolo</td>
      <td>Ceretto</td>
    </tr>
    <tr>
      <th>130</th>
      <td>Italy</td>
      <td>At the first it was quite muted and subdued, b...</td>
      <td>Bricco Rocche Brunate</td>
      <td>91</td>
      <td>70.0</td>
      <td>Piedmont</td>
      <td>Barolo</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Ceretto 2003 Bricco Rocche Brunate  (Barolo)</td>
      <td>Nebbiolo</td>
      <td>Ceretto</td>
    </tr>
    <tr>
      <th>133</th>
      <td>Italy</td>
      <td>Einaudi's wines have been improving lately, an...</td>
      <td>NaN</td>
      <td>91</td>
      <td>68.0</td>
      <td>Piedmont</td>
      <td>Barolo</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Poderi Luigi Einaudi 2003  Barolo</td>
      <td>Nebbiolo</td>
      <td>Poderi Luigi Einaudi</td>
    </tr>
    <tr>
      <th>135</th>
      <td>Italy</td>
      <td>The color is just beginning to show signs of b...</td>
      <td>Sorano</td>
      <td>91</td>
      <td>60.0</td>
      <td>Piedmont</td>
      <td>Barolo</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Giacomo Ascheri 2001 Sorano  (Barolo)</td>
      <td>Nebbiolo</td>
      <td>Giacomo Ascheri</td>
    </tr>
    <tr>
      <th>140</th>
      <td>Italy</td>
      <td>A big, fat, luscious wine with plenty of toast...</td>
      <td>Costa Bruna</td>
      <td>90</td>
      <td>26.0</td>
      <td>Piedmont</td>
      <td>Barbera d'Alba</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Poderi Colla 2005 Costa Bruna  (Barbera d'Alba)</td>
      <td>Barbera</td>
      <td>Poderi Colla</td>
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
    </tr>
    <tr>
      <th>129929</th>
      <td>Italy</td>
      <td>This luminous sparkler has a sweet, fruit-forw...</td>
      <td>NaN</td>
      <td>91</td>
      <td>38.0</td>
      <td>Veneto</td>
      <td>Prosecco Superiore di Cartizze</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Col Vetoraz Spumanti NV  Prosecco Superiore di...</td>
      <td>Prosecco</td>
      <td>Col Vetoraz Spumanti</td>
    </tr>
    <tr>
      <th>129943</th>
      <td>Italy</td>
      <td>A blend of Nero d'Avola and Syrah, this convey...</td>
      <td>Adènzia</td>
      <td>90</td>
      <td>29.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Baglio del Cristo di Campobello 2012 Adènzia R...</td>
      <td>Red Blend</td>
      <td>Baglio del Cristo di Campobello</td>
    </tr>
    <tr>
      <th>129947</th>
      <td>Italy</td>
      <td>A blend of 65% Cabernet Sauvignon, 30% Merlot ...</td>
      <td>Symposio</td>
      <td>90</td>
      <td>20.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Terre Siciliane</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Feudo Principi di Butera 2012 Symposio Red (Te...</td>
      <td>Red Blend</td>
      <td>Feudo Principi di Butera</td>
    </tr>
    <tr>
      <th>129961</th>
      <td>Italy</td>
      <td>Intense aromas of wild cherry, baking spice, t...</td>
      <td>NaN</td>
      <td>90</td>
      <td>30.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>COS 2013 Frappato (Sicilia)</td>
      <td>Frappato</td>
      <td>COS</td>
    </tr>
    <tr>
      <th>129962</th>
      <td>Italy</td>
      <td>Blackberry, cassis, grilled herb and toasted a...</td>
      <td>Sàgana Tenuta San Giacomo</td>
      <td>90</td>
      <td>40.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Cusumano 2012 Sàgana Tenuta San Giacomo Nero d...</td>
      <td>Nero d'Avola</td>
      <td>Cusumano</td>
    </tr>
  </tbody>
</table>
<p>6648 rows × 13 columns</p>
</div>



Si queremos un vino Italiano o con puntuación mayor o igual a 90:


```python
reviews.loc[(reviews['country'] == 'Italy') | (reviews['points'] >= 90)]
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Italy</td>
      <td>Here's a bright, informal red that opens with ...</td>
      <td>Belsito</td>
      <td>87</td>
      <td>16.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Vittoria</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Terre di Giurfo 2013 Belsito Frappato (Vittoria)</td>
      <td>Frappato</td>
      <td>Terre di Giurfo</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Italy</td>
      <td>This is dominated by oak and oak-driven aromas...</td>
      <td>Rosso</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Masseria Setteporte 2012 Rosso  (Etna)</td>
      <td>Nerello Mascalese</td>
      <td>Masseria Setteporte</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Italy</td>
      <td>Delicate aromas recall white flower and citrus...</td>
      <td>Ficiligno</td>
      <td>87</td>
      <td>19.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Baglio di Pianetto 2007 Ficiligno White (Sicilia)</td>
      <td>White Blend</td>
      <td>Baglio di Pianetto</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Italy</td>
      <td>Aromas of prune, blackcurrant, toast and oak c...</td>
      <td>Aynat</td>
      <td>87</td>
      <td>35.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Canicattì 2009 Aynat Nero d'Avola (Sicilia)</td>
      <td>Nero d'Avola</td>
      <td>Canicattì</td>
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
    </tr>
    <tr>
      <th>129966</th>
      <td>Germany</td>
      <td>Notes of honeysuckle and cantaloupe sweeten th...</td>
      <td>Brauneberger Juffer-Sonnenuhr Spätlese</td>
      <td>90</td>
      <td>28.0</td>
      <td>Mosel</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Anna Lee C. Iijima</td>
      <td>NaN</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef) 2013 ...</td>
      <td>Riesling</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef)</td>
    </tr>
    <tr>
      <th>129967</th>
      <td>US</td>
      <td>Citation is given as much as a decade of bottl...</td>
      <td>NaN</td>
      <td>90</td>
      <td>75.0</td>
      <td>Oregon</td>
      <td>Oregon</td>
      <td>Oregon Other</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Citation 2004 Pinot Noir (Oregon)</td>
      <td>Pinot Noir</td>
      <td>Citation</td>
    </tr>
    <tr>
      <th>129968</th>
      <td>France</td>
      <td>Well-drained gravel soil gives this wine its c...</td>
      <td>Kritt</td>
      <td>90</td>
      <td>30.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Gresser 2013 Kritt Gewurztraminer (Als...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Gresser</td>
    </tr>
    <tr>
      <th>129969</th>
      <td>France</td>
      <td>A dry style of Pinot Gris, this is crisp with ...</td>
      <td>NaN</td>
      <td>90</td>
      <td>32.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Marcel Deiss 2012 Pinot Gris (Alsace)</td>
      <td>Pinot Gris</td>
      <td>Domaine Marcel Deiss</td>
    </tr>
    <tr>
      <th>129970</th>
      <td>France</td>
      <td>Big, rich and off-dry, this is powered by inte...</td>
      <td>Lieu-dit Harth Cuvée Caroline</td>
      <td>90</td>
      <td>21.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Schoffit 2012 Lieu-dit Harth Cuvée Car...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Schoffit</td>
    </tr>
  </tbody>
</table>
<p>61937 rows × 13 columns</p>
</div>



Si queremos un vino Italiano o Español:


```python
reviews.loc[reviews['country'].isin(['Italy', 'Spain'])]
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spain</td>
      <td>Blackberry and raspberry aromas show a typical...</td>
      <td>Ars In Vitro</td>
      <td>87</td>
      <td>15.0</td>
      <td>Northern Spain</td>
      <td>Navarra</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Tandem 2011 Ars In Vitro Tempranillo-Merlot (N...</td>
      <td>Tempranillo-Merlot</td>
      <td>Tandem</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Italy</td>
      <td>Here's a bright, informal red that opens with ...</td>
      <td>Belsito</td>
      <td>87</td>
      <td>16.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Vittoria</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Terre di Giurfo 2013 Belsito Frappato (Vittoria)</td>
      <td>Frappato</td>
      <td>Terre di Giurfo</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Italy</td>
      <td>This is dominated by oak and oak-driven aromas...</td>
      <td>Rosso</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Masseria Setteporte 2012 Rosso  (Etna)</td>
      <td>Nerello Mascalese</td>
      <td>Masseria Setteporte</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Spain</td>
      <td>Desiccated blackberry, leather, charred wood a...</td>
      <td>Vendimia Seleccionada Finca Valdelayegua Singl...</td>
      <td>87</td>
      <td>28.0</td>
      <td>Northern Spain</td>
      <td>Ribera del Duero</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Pradorey 2010 Vendimia Seleccionada Finca Vald...</td>
      <td>Tempranillo Blend</td>
      <td>Pradorey</td>
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
    </tr>
    <tr>
      <th>129943</th>
      <td>Italy</td>
      <td>A blend of Nero d'Avola and Syrah, this convey...</td>
      <td>Adènzia</td>
      <td>90</td>
      <td>29.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Baglio del Cristo di Campobello 2012 Adènzia R...</td>
      <td>Red Blend</td>
      <td>Baglio del Cristo di Campobello</td>
    </tr>
    <tr>
      <th>129947</th>
      <td>Italy</td>
      <td>A blend of 65% Cabernet Sauvignon, 30% Merlot ...</td>
      <td>Symposio</td>
      <td>90</td>
      <td>20.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Terre Siciliane</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Feudo Principi di Butera 2012 Symposio Red (Te...</td>
      <td>Red Blend</td>
      <td>Feudo Principi di Butera</td>
    </tr>
    <tr>
      <th>129957</th>
      <td>Spain</td>
      <td>Lightly baked berry aromas vie for attention w...</td>
      <td>Crianza</td>
      <td>90</td>
      <td>17.0</td>
      <td>Northern Spain</td>
      <td>Rioja</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Viñedos Real Rubio 2010 Crianza  (Rioja)</td>
      <td>Tempranillo Blend</td>
      <td>Viñedos Real Rubio</td>
    </tr>
    <tr>
      <th>129961</th>
      <td>Italy</td>
      <td>Intense aromas of wild cherry, baking spice, t...</td>
      <td>NaN</td>
      <td>90</td>
      <td>30.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>COS 2013 Frappato (Sicilia)</td>
      <td>Frappato</td>
      <td>COS</td>
    </tr>
    <tr>
      <th>129962</th>
      <td>Italy</td>
      <td>Blackberry, cassis, grilled herb and toasted a...</td>
      <td>Sàgana Tenuta San Giacomo</td>
      <td>90</td>
      <td>40.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Cusumano 2012 Sàgana Tenuta San Giacomo Nero d...</td>
      <td>Nero d'Avola</td>
      <td>Cusumano</td>
    </tr>
  </tbody>
</table>
<p>26185 rows × 13 columns</p>
</div>



Si queremos deshacernos de las instancias en las que no tenemos el valor del precio:


```python
reviews.loc[reviews['price'].notnull()]
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spain</td>
      <td>Blackberry and raspberry aromas show a typical...</td>
      <td>Ars In Vitro</td>
      <td>87</td>
      <td>15.0</td>
      <td>Northern Spain</td>
      <td>Navarra</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Tandem 2011 Ars In Vitro Tempranillo-Merlot (N...</td>
      <td>Tempranillo-Merlot</td>
      <td>Tandem</td>
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
    </tr>
    <tr>
      <th>129966</th>
      <td>Germany</td>
      <td>Notes of honeysuckle and cantaloupe sweeten th...</td>
      <td>Brauneberger Juffer-Sonnenuhr Spätlese</td>
      <td>90</td>
      <td>28.0</td>
      <td>Mosel</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Anna Lee C. Iijima</td>
      <td>NaN</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef) 2013 ...</td>
      <td>Riesling</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef)</td>
    </tr>
    <tr>
      <th>129967</th>
      <td>US</td>
      <td>Citation is given as much as a decade of bottl...</td>
      <td>NaN</td>
      <td>90</td>
      <td>75.0</td>
      <td>Oregon</td>
      <td>Oregon</td>
      <td>Oregon Other</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Citation 2004 Pinot Noir (Oregon)</td>
      <td>Pinot Noir</td>
      <td>Citation</td>
    </tr>
    <tr>
      <th>129968</th>
      <td>France</td>
      <td>Well-drained gravel soil gives this wine its c...</td>
      <td>Kritt</td>
      <td>90</td>
      <td>30.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Gresser 2013 Kritt Gewurztraminer (Als...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Gresser</td>
    </tr>
    <tr>
      <th>129969</th>
      <td>France</td>
      <td>A dry style of Pinot Gris, this is crisp with ...</td>
      <td>NaN</td>
      <td>90</td>
      <td>32.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Marcel Deiss 2012 Pinot Gris (Alsace)</td>
      <td>Pinot Gris</td>
      <td>Domaine Marcel Deiss</td>
    </tr>
    <tr>
      <th>129970</th>
      <td>France</td>
      <td>Big, rich and off-dry, this is powered by inte...</td>
      <td>Lieu-dit Harth Cuvée Caroline</td>
      <td>90</td>
      <td>21.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Schoffit 2012 Lieu-dit Harth Cuvée Car...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Schoffit</td>
    </tr>
  </tbody>
</table>
<p>120975 rows × 13 columns</p>
</div>



### Añadir datos

Añadir datos a nuestros `DataFrames` es fácil. Por ejemplo, podemos asignar el mismo valor a todas las instancias con el siguiente comando:


```python
reviews['critic'] = 'everyone'
reviews['critic']
```




    0         everyone
    1         everyone
    2         everyone
    3         everyone
    4         everyone
                ...   
    129966    everyone
    129967    everyone
    129968    everyone
    129969    everyone
    129970    everyone
    Name: critic, Length: 129971, dtype: object




```python
reviews['index_backwards'] = range(len(reviews), 0, -1)
reviews['index_backwards']
```




    0         129971
    1         129970
    2         129969
    3         129968
    4         129967
               ...  
    129966         5
    129967         4
    129968         3
    129969         2
    129970         1
    Name: index_backwards, Length: 129971, dtype: int32



### Describir nuestro dataset

Pandas nos proporciona herramientas para facilmente conocer un poco por encima como es el dataset con el que estamos trabajando a traves de valores estadísticos.


```python
reviews.describe()
```




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
      <th>points</th>
      <th>price</th>
      <th>index_backwards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>129971.000000</td>
      <td>120975.000000</td>
      <td>129971.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>88.447138</td>
      <td>35.363389</td>
      <td>64986.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.039730</td>
      <td>41.022218</td>
      <td>37519.540256</td>
    </tr>
    <tr>
      <th>min</th>
      <td>80.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>86.000000</td>
      <td>17.000000</td>
      <td>32493.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>88.000000</td>
      <td>25.000000</td>
      <td>64986.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>91.000000</td>
      <td>42.000000</td>
      <td>97478.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>3300.000000</td>
      <td>129971.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
reviews.describe(include='all')
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
      <th>critic</th>
      <th>index_backwards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>129908</td>
      <td>129971</td>
      <td>92506</td>
      <td>129971.000000</td>
      <td>120975.000000</td>
      <td>129908</td>
      <td>108724</td>
      <td>50511</td>
      <td>103727</td>
      <td>98758</td>
      <td>129971</td>
      <td>129970</td>
      <td>129971</td>
      <td>129971</td>
      <td>129971.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>43</td>
      <td>119955</td>
      <td>37979</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>425</td>
      <td>1229</td>
      <td>17</td>
      <td>19</td>
      <td>15</td>
      <td>118840</td>
      <td>707</td>
      <td>16757</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>US</td>
      <td>Ripe plum, game, truffle, leather and menthol ...</td>
      <td>Reserve</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>California</td>
      <td>Napa Valley</td>
      <td>Central Coast</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Gloria Ferrer NV Sonoma Brut Sparkling (Sonoma...</td>
      <td>Pinot Noir</td>
      <td>Wines &amp; Winemakers</td>
      <td>everyone</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>54504</td>
      <td>3</td>
      <td>2009</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>36247</td>
      <td>4480</td>
      <td>11065</td>
      <td>25514</td>
      <td>25514</td>
      <td>11</td>
      <td>13272</td>
      <td>222</td>
      <td>129971</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>88.447138</td>
      <td>35.363389</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>64986.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.039730</td>
      <td>41.022218</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37519.540256</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>80.000000</td>
      <td>4.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>86.000000</td>
      <td>17.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32493.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>88.000000</td>
      <td>25.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>64986.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.000000</td>
      <td>42.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>97478.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.000000</td>
      <td>3300.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>129971.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
reviews.dtypes
```




    country                   object
    description               object
    designation               object
    points                     int64
    price                    float64
    province                  object
    region_1                  object
    region_2                  object
    taster_name               object
    taster_twitter_handle     object
    title                     object
    variety                   object
    winery                    object
    critic                    object
    index_backwards            int32
    dtype: object




```python
reviews['points'].mean()
```




    88.44713820775404




```python
reviews['points'].quantile(0.25)
```




    86.0




```python
reviews['country'].unique()
```




    array(['Italy', 'Portugal', 'US', 'Spain', 'France', 'Germany',
           'Argentina', 'Chile', 'Australia', 'Austria', 'South Africa',
           'New Zealand', 'Israel', 'Hungary', 'Greece', 'Romania', 'Mexico',
           'Canada', nan, 'Turkey', 'Czech Republic', 'Slovenia',
           'Luxembourg', 'Croatia', 'Georgia', 'Uruguay', 'England',
           'Lebanon', 'Serbia', 'Brazil', 'Moldova', 'Morocco', 'Peru',
           'India', 'Bulgaria', 'Cyprus', 'Armenia', 'Switzerland',
           'Bosnia and Herzegovina', 'Ukraine', 'Slovakia', 'Macedonia',
           'China', 'Egypt'], dtype=object)




```python
reviews['country'].value_counts()
```




    US                        54504
    France                    22093
    Italy                     19540
    Spain                      6645
    Portugal                   5691
    Chile                      4472
    Argentina                  3800
    Austria                    3345
    Australia                  2329
    Germany                    2165
    New Zealand                1419
    South Africa               1401
    Israel                      505
    Greece                      466
    Canada                      257
    Hungary                     146
    Bulgaria                    141
    Romania                     120
    Uruguay                     109
    Turkey                       90
    Slovenia                     87
    Georgia                      86
    England                      74
    Croatia                      73
    Mexico                       70
    Moldova                      59
    Brazil                       52
    Lebanon                      35
    Morocco                      28
    Peru                         16
    Ukraine                      14
    Macedonia                    12
    Serbia                       12
    Czech Republic               12
    Cyprus                       11
    India                         9
    Switzerland                   7
    Luxembourg                    6
    Bosnia and Herzegovina        2
    Armenia                       2
    Egypt                         1
    China                         1
    Slovakia                      1
    Name: country, dtype: int64



### Modificar los valores de una columna

Por ejemplo, vamos a normalizar los datos de la columna points.


```python
(reviews['points'] - reviews['points'].mean()) / reviews['points'].std()
```




    0        -0.476075
    1        -0.476075
    2        -0.476075
    3        -0.476075
    4        -0.476075
                ...   
    129966    0.510855
    129967    0.510855
    129968    0.510855
    129969    0.510855
    129970    0.510855
    Name: points, Length: 129971, dtype: float64




```python
reviews['province - region'] = reviews['province'] + ' - ' + reviews['region_1']
reviews
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
      <th>critic</th>
      <th>index_backwards</th>
      <th>province - region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
      <td>everyone</td>
      <td>129971</td>
      <td>Sicily &amp; Sardinia - Etna</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
      <td>everyone</td>
      <td>129970</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
      <td>everyone</td>
      <td>129969</td>
      <td>Oregon - Willamette Valley</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
      <td>everyone</td>
      <td>129968</td>
      <td>Michigan - Lake Michigan Shore</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
      <td>everyone</td>
      <td>129967</td>
      <td>Oregon - Willamette Valley</td>
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
    </tr>
    <tr>
      <th>129966</th>
      <td>Germany</td>
      <td>Notes of honeysuckle and cantaloupe sweeten th...</td>
      <td>Brauneberger Juffer-Sonnenuhr Spätlese</td>
      <td>90</td>
      <td>28.0</td>
      <td>Mosel</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Anna Lee C. Iijima</td>
      <td>NaN</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef) 2013 ...</td>
      <td>Riesling</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef)</td>
      <td>everyone</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>129967</th>
      <td>US</td>
      <td>Citation is given as much as a decade of bottl...</td>
      <td>NaN</td>
      <td>90</td>
      <td>75.0</td>
      <td>Oregon</td>
      <td>Oregon</td>
      <td>Oregon Other</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Citation 2004 Pinot Noir (Oregon)</td>
      <td>Pinot Noir</td>
      <td>Citation</td>
      <td>everyone</td>
      <td>4</td>
      <td>Oregon - Oregon</td>
    </tr>
    <tr>
      <th>129968</th>
      <td>France</td>
      <td>Well-drained gravel soil gives this wine its c...</td>
      <td>Kritt</td>
      <td>90</td>
      <td>30.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Gresser 2013 Kritt Gewurztraminer (Als...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Gresser</td>
      <td>everyone</td>
      <td>3</td>
      <td>Alsace - Alsace</td>
    </tr>
    <tr>
      <th>129969</th>
      <td>France</td>
      <td>A dry style of Pinot Gris, this is crisp with ...</td>
      <td>NaN</td>
      <td>90</td>
      <td>32.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Marcel Deiss 2012 Pinot Gris (Alsace)</td>
      <td>Pinot Gris</td>
      <td>Domaine Marcel Deiss</td>
      <td>everyone</td>
      <td>2</td>
      <td>Alsace - Alsace</td>
    </tr>
    <tr>
      <th>129970</th>
      <td>France</td>
      <td>Big, rich and off-dry, this is powered by inte...</td>
      <td>Lieu-dit Harth Cuvée Caroline</td>
      <td>90</td>
      <td>21.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Schoffit 2012 Lieu-dit Harth Cuvée Car...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Schoffit</td>
      <td>everyone</td>
      <td>1</td>
      <td>Alsace - Alsace</td>
    </tr>
  </tbody>
</table>
<p>129971 rows × 16 columns</p>
</div>



### Eliminar columnas


```python
reviews.columns
```




    Index(['country', 'description', 'designation', 'points', 'price', 'province',
           'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'title',
           'variety', 'winery', 'critic', 'index_backwards', 'province - region'],
          dtype='object')




```python
reviews.pop('province - region')
```




    0               Sicily & Sardinia - Etna
    1                                    NaN
    2             Oregon - Willamette Valley
    3         Michigan - Lake Michigan Shore
    4             Oregon - Willamette Valley
                           ...              
    129966                               NaN
    129967                   Oregon - Oregon
    129968                   Alsace - Alsace
    129969                   Alsace - Alsace
    129970                   Alsace - Alsace
    Name: province - region, Length: 129971, dtype: object




```python
reviews = reviews.drop(columns=['critic', 'index_backwards'])
reviews.columns
```




    Index(['country', 'description', 'designation', 'points', 'price', 'province',
           'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'title',
           'variety', 'winery'],
          dtype='object')



### Agrupar datos


```python
reviews.groupby('points')['points'].count()
```




    points
    80       397
    81       692
    82      1836
    83      3025
    84      6480
    85      9530
    86     12600
    87     16933
    88     17207
    89     12226
    90     15410
    91     11359
    92      9613
    93      6489
    94      3758
    95      1535
    96       523
    97       229
    98        77
    99        33
    100       19
    Name: points, dtype: int64



Lo que ha ocurrido es que la función `groupby()` ha creado diferentes grupos dependiendo de la puntuación y luego a contado cuantos vinos hay en cada grupo.

Ahora, vamos a calcular el precio medio de los vinos dependiendo la puntuación:


```python
reviews.groupby('points')['price'].mean()
```




    points
    80      16.372152
    81      17.182353
    82      18.870767
    83      18.237353
    84      19.310215
    85      19.949562
    86      22.133759
    87      24.901884
    88      28.687523
    89      32.169640
    90      36.906622
    91      43.224252
    92      51.037763
    93      63.112216
    94      81.436938
    95     109.235420
    96     159.292531
    97     207.173913
    98     245.492754
    99     284.214286
    100    485.947368
    Name: price, dtype: float64



Podemos agrupar usando más de un criterio y devolver más de un valor con `agg()`.


```python
reviews.groupby(['price', 'country']).agg(['count', 'min', 'mean', 'max'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="4" halign="left">points</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>min</th>
      <th>mean</th>
      <th>max</th>
    </tr>
    <tr>
      <th>price</th>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">4.0</th>
      <th>Argentina</th>
      <td>1</td>
      <td>84</td>
      <td>84.000000</td>
      <td>84</td>
    </tr>
    <tr>
      <th>Romania</th>
      <td>1</td>
      <td>86</td>
      <td>86.000000</td>
      <td>86</td>
    </tr>
    <tr>
      <th>Spain</th>
      <td>4</td>
      <td>82</td>
      <td>83.750000</td>
      <td>85</td>
    </tr>
    <tr>
      <th>US</th>
      <td>5</td>
      <td>83</td>
      <td>84.400000</td>
      <td>86</td>
    </tr>
    <tr>
      <th>5.0</th>
      <th>Argentina</th>
      <td>3</td>
      <td>80</td>
      <td>81.333333</td>
      <td>84</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1900.0</th>
      <th>France</th>
      <td>1</td>
      <td>98</td>
      <td>98.000000</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2000.0</th>
      <th>France</th>
      <td>2</td>
      <td>96</td>
      <td>96.500000</td>
      <td>97</td>
    </tr>
    <tr>
      <th>2013.0</th>
      <th>US</th>
      <td>1</td>
      <td>91</td>
      <td>91.000000</td>
      <td>91</td>
    </tr>
    <tr>
      <th>2500.0</th>
      <th>France</th>
      <td>2</td>
      <td>96</td>
      <td>96.000000</td>
      <td>96</td>
    </tr>
    <tr>
      <th>3300.0</th>
      <th>France</th>
      <td>1</td>
      <td>88</td>
      <td>88.000000</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
<p>2238 rows × 4 columns</p>
</div>



### Ordenar instancias


```python
reviews.sort_values(by='points')
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>118056</th>
      <td>US</td>
      <td>This wine has very little going on aromaticall...</td>
      <td>Reserve</td>
      <td>80</td>
      <td>26.0</td>
      <td>California</td>
      <td>Livermore Valley</td>
      <td>Central Coast</td>
      <td>Virginie Boone</td>
      <td>@vboone</td>
      <td>3 Steves Winery 2008 Reserve Cabernet Sauvigno...</td>
      <td>Cabernet Sauvignon</td>
      <td>3 Steves Winery</td>
    </tr>
    <tr>
      <th>35516</th>
      <td>US</td>
      <td>This Merlot has not fully ripened, with aromas...</td>
      <td>NaN</td>
      <td>80</td>
      <td>20.0</td>
      <td>Washington</td>
      <td>Horse Heaven Hills</td>
      <td>Columbia Valley</td>
      <td>Sean P. Sullivan</td>
      <td>@wawinereport</td>
      <td>James Wyatt 2013 Merlot (Horse Heaven Hills)</td>
      <td>Merlot</td>
      <td>James Wyatt</td>
    </tr>
    <tr>
      <th>11086</th>
      <td>France</td>
      <td>Picture grandma standing over a pot of stewed ...</td>
      <td>NaN</td>
      <td>80</td>
      <td>11.0</td>
      <td>Languedoc-Roussillon</td>
      <td>Fitou</td>
      <td>NaN</td>
      <td>Joe Czerwinski</td>
      <td>@JoeCz</td>
      <td>Mont Tauch 1998 Red (Fitou)</td>
      <td>Red Blend</td>
      <td>Mont Tauch</td>
    </tr>
    <tr>
      <th>11085</th>
      <td>France</td>
      <td>A white this age should be fresh and crisp; th...</td>
      <td>NaN</td>
      <td>80</td>
      <td>8.0</td>
      <td>Southwest France</td>
      <td>Bergerac</td>
      <td>NaN</td>
      <td>Joe Czerwinski</td>
      <td>@JoeCz</td>
      <td>Seigneurs de Bergerac 1999 White (Bergerac)</td>
      <td>White Blend</td>
      <td>Seigneurs de Bergerac</td>
    </tr>
    <tr>
      <th>102482</th>
      <td>US</td>
      <td>This wine is a medium cherry-red color, with s...</td>
      <td>Cabernet Franc</td>
      <td>80</td>
      <td>18.0</td>
      <td>Washington</td>
      <td>Columbia Valley (WA)</td>
      <td>Columbia Valley</td>
      <td>Sean P. Sullivan</td>
      <td>@wawinereport</td>
      <td>Tucannon 2014 Cabernet Franc Rosé (Columbia Va...</td>
      <td>Rosé</td>
      <td>Tucannon</td>
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
    </tr>
    <tr>
      <th>111756</th>
      <td>France</td>
      <td>A hugely powerful wine, full of dark, brooding...</td>
      <td>NaN</td>
      <td>100</td>
      <td>359.0</td>
      <td>Bordeaux</td>
      <td>Saint-Julien</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château Léoville Las Cases 2010  Saint-Julien</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château Léoville Las Cases</td>
    </tr>
    <tr>
      <th>89728</th>
      <td>France</td>
      <td>This latest incarnation of the famous brand is...</td>
      <td>Cristal Vintage Brut</td>
      <td>100</td>
      <td>250.0</td>
      <td>Champagne</td>
      <td>Champagne</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Louis Roederer 2008 Cristal Vintage Brut  (Cha...</td>
      <td>Champagne Blend</td>
      <td>Louis Roederer</td>
    </tr>
    <tr>
      <th>89729</th>
      <td>France</td>
      <td>This new release from a great vintage for Char...</td>
      <td>Le Mesnil Blanc de Blancs Brut</td>
      <td>100</td>
      <td>617.0</td>
      <td>Champagne</td>
      <td>Champagne</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Salon 2006 Le Mesnil Blanc de Blancs Brut Char...</td>
      <td>Chardonnay</td>
      <td>Salon</td>
    </tr>
    <tr>
      <th>118058</th>
      <td>US</td>
      <td>This wine dazzles with perfection. Sourced fro...</td>
      <td>La Muse</td>
      <td>100</td>
      <td>450.0</td>
      <td>California</td>
      <td>Sonoma County</td>
      <td>Sonoma</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Verité 2007 La Muse Red (Sonoma County)</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Verité</td>
    </tr>
    <tr>
      <th>111755</th>
      <td>France</td>
      <td>This is the finest Cheval Blanc for many years...</td>
      <td>NaN</td>
      <td>100</td>
      <td>1500.0</td>
      <td>Bordeaux</td>
      <td>Saint-Émilion</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château Cheval Blanc 2010  Saint-Émilion</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château Cheval Blanc</td>
    </tr>
  </tbody>
</table>
<p>129971 rows × 13 columns</p>
</div>




```python
reviews.sort_values(by='points', ascending=False).reset_index()
```




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
      <th>index</th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>114972</td>
      <td>Portugal</td>
      <td>A powerful and ripe wine, strongly influenced ...</td>
      <td>Nacional Vintage</td>
      <td>100</td>
      <td>650.0</td>
      <td>Port</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta do Noval 2011 Nacional Vintage  (Port)</td>
      <td>Port</td>
      <td>Quinta do Noval</td>
    </tr>
    <tr>
      <th>1</th>
      <td>89729</td>
      <td>France</td>
      <td>This new release from a great vintage for Char...</td>
      <td>Le Mesnil Blanc de Blancs Brut</td>
      <td>100</td>
      <td>617.0</td>
      <td>Champagne</td>
      <td>Champagne</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Salon 2006 Le Mesnil Blanc de Blancs Brut Char...</td>
      <td>Chardonnay</td>
      <td>Salon</td>
    </tr>
    <tr>
      <th>2</th>
      <td>113929</td>
      <td>US</td>
      <td>In 2005 Charles Smith introduced three high-en...</td>
      <td>Royal City</td>
      <td>100</td>
      <td>80.0</td>
      <td>Washington</td>
      <td>Columbia Valley (WA)</td>
      <td>Columbia Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Charles Smith 2006 Royal City Syrah (Columbia ...</td>
      <td>Syrah</td>
      <td>Charles Smith</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45781</td>
      <td>Italy</td>
      <td>This gorgeous, fragrant wine opens with classi...</td>
      <td>Riserva</td>
      <td>100</td>
      <td>550.0</td>
      <td>Tuscany</td>
      <td>Brunello di Montalcino</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Biondi Santi 2010 Riserva  (Brunello di Montal...</td>
      <td>Sangiovese</td>
      <td>Biondi Santi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>123545</td>
      <td>US</td>
      <td>Initially a rather subdued Frog; as if it has ...</td>
      <td>Bionic Frog</td>
      <td>100</td>
      <td>80.0</td>
      <td>Washington</td>
      <td>Walla Walla Valley (WA)</td>
      <td>Columbia Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Cayuse 2008 Bionic Frog Syrah (Walla Walla Val...</td>
      <td>Syrah</td>
      <td>Cayuse</td>
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
    </tr>
    <tr>
      <th>129966</th>
      <td>128255</td>
      <td>Argentina</td>
      <td>Severely compromised by green, minty, weedy ar...</td>
      <td>NaN</td>
      <td>80</td>
      <td>13.0</td>
      <td>Mendoza Province</td>
      <td>Mendoza</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Viniterra 2007 Malbec (Mendoza)</td>
      <td>Malbec</td>
      <td>Viniterra</td>
    </tr>
    <tr>
      <th>129967</th>
      <td>128254</td>
      <td>Argentina</td>
      <td>Disappointing considering the source. The nose...</td>
      <td>Gran Lurton</td>
      <td>80</td>
      <td>20.0</td>
      <td>Mendoza Province</td>
      <td>Mendoza</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>François Lurton 2006 Gran Lurton Cabernet Sauv...</td>
      <td>Cabernet Sauvignon</td>
      <td>François Lurton</td>
    </tr>
    <tr>
      <th>129968</th>
      <td>93686</td>
      <td>Peru</td>
      <td>Best on the nose, where apple and lemony aroma...</td>
      <td>Brut</td>
      <td>80</td>
      <td>15.0</td>
      <td>Ica</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Tacama 2010 Brut Sparkling (Ica)</td>
      <td>Sparkling Blend</td>
      <td>Tacama</td>
    </tr>
    <tr>
      <th>129969</th>
      <td>73865</td>
      <td>Chile</td>
      <td>There's not much point in making a reserve-sty...</td>
      <td>Prima Reserva</td>
      <td>80</td>
      <td>13.0</td>
      <td>Maipo Valley</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Joe Czerwinski</td>
      <td>@JoeCz</td>
      <td>De Martino 1999 Prima Reserva Merlot (Maipo Va...</td>
      <td>Merlot</td>
      <td>De Martino</td>
    </tr>
    <tr>
      <th>129970</th>
      <td>128256</td>
      <td>US</td>
      <td>A hot, harsh and unbalance Chardonnay. Barely ...</td>
      <td>NaN</td>
      <td>80</td>
      <td>24.0</td>
      <td>California</td>
      <td>Contra Costa County</td>
      <td>Central Coast</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bloomfield 2008 Chardonnay (Contra Costa County)</td>
      <td>Chardonnay</td>
      <td>Bloomfield</td>
    </tr>
  </tbody>
</table>
<p>129971 rows × 14 columns</p>
</div>



### Missing data

Tratar con los datos que faltan es muy importante. Pandas nos ofrece funciones como `isnull, notnull y fillna` para localizar y rellenar los valores perdidos.


```python
reviews['country'].isnull()
```




    0         False
    1         False
    2         False
    3         False
    4         False
              ...  
    129966    False
    129967    False
    129968    False
    129969    False
    129970    False
    Name: country, Length: 129971, dtype: bool




```python
reviews[reviews['country'].isnull()]
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>913</th>
      <td>NaN</td>
      <td>Amber in color, this wine has aromas of peach ...</td>
      <td>Asureti Valley</td>
      <td>87</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Mike DeSimone</td>
      <td>@worldwineguys</td>
      <td>Gotsa Family Wines 2014 Asureti Valley Chinuri</td>
      <td>Chinuri</td>
      <td>Gotsa Family Wines</td>
    </tr>
    <tr>
      <th>3131</th>
      <td>NaN</td>
      <td>Soft, fruity and juicy, this is a pleasant, si...</td>
      <td>Partager</td>
      <td>83</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Barton &amp; Guestier NV Partager Red</td>
      <td>Red Blend</td>
      <td>Barton &amp; Guestier</td>
    </tr>
    <tr>
      <th>4243</th>
      <td>NaN</td>
      <td>Violet-red in color, this semisweet wine has a...</td>
      <td>Red Naturally Semi-Sweet</td>
      <td>88</td>
      <td>18.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Mike DeSimone</td>
      <td>@worldwineguys</td>
      <td>Kakhetia Traditional Winemaking 2012 Red Natur...</td>
      <td>Ojaleshi</td>
      <td>Kakhetia Traditional Winemaking</td>
    </tr>
    <tr>
      <th>9509</th>
      <td>NaN</td>
      <td>This mouthwatering blend starts with a nose of...</td>
      <td>Theopetra Malagouzia-Assyrtiko</td>
      <td>92</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Susan Kostrzewa</td>
      <td>@suskostrzewa</td>
      <td>Tsililis 2015 Theopetra Malagouzia-Assyrtiko W...</td>
      <td>White Blend</td>
      <td>Tsililis</td>
    </tr>
    <tr>
      <th>9750</th>
      <td>NaN</td>
      <td>This orange-style wine has a cloudy yellow-gol...</td>
      <td>Orange Nikolaevo Vineyard</td>
      <td>89</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Jeff Jenssen</td>
      <td>@worldwineguys</td>
      <td>Ross-idi 2015 Orange Nikolaevo Vineyard Chardo...</td>
      <td>Chardonnay</td>
      <td>Ross-idi</td>
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
    </tr>
    <tr>
      <th>124176</th>
      <td>NaN</td>
      <td>This Swiss red blend is composed of four varie...</td>
      <td>Les Romaines</td>
      <td>90</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Jeff Jenssen</td>
      <td>@worldwineguys</td>
      <td>Les Frères Dutruy 2014 Les Romaines Red</td>
      <td>Red Blend</td>
      <td>Les Frères Dutruy</td>
    </tr>
    <tr>
      <th>129407</th>
      <td>NaN</td>
      <td>Dry spicy aromas of dusty plum and tomato add ...</td>
      <td>Reserve</td>
      <td>89</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>El Capricho 2015 Reserve Cabernet Sauvignon</td>
      <td>Cabernet Sauvignon</td>
      <td>El Capricho</td>
    </tr>
    <tr>
      <th>129408</th>
      <td>NaN</td>
      <td>El Capricho is one of Uruguay's more consisten...</td>
      <td>Reserve</td>
      <td>89</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>El Capricho 2015 Reserve Tempranillo</td>
      <td>Tempranillo</td>
      <td>El Capricho</td>
    </tr>
    <tr>
      <th>129590</th>
      <td>NaN</td>
      <td>A blend of 60% Syrah, 30% Cabernet Sauvignon a...</td>
      <td>Shah</td>
      <td>90</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Mike DeSimone</td>
      <td>@worldwineguys</td>
      <td>Büyülübağ 2012 Shah Red</td>
      <td>Red Blend</td>
      <td>Büyülübağ</td>
    </tr>
    <tr>
      <th>129900</th>
      <td>NaN</td>
      <td>This wine offers a delightful bouquet of black...</td>
      <td>NaN</td>
      <td>91</td>
      <td>32.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Mike DeSimone</td>
      <td>@worldwineguys</td>
      <td>Psagot 2014 Merlot</td>
      <td>Merlot</td>
      <td>Psagot</td>
    </tr>
  </tbody>
</table>
<p>63 rows × 13 columns</p>
</div>




```python
reviews['region_2'].fillna("Unknown")
```




    0                   Unknown
    1                   Unknown
    2         Willamette Valley
    3                   Unknown
    4         Willamette Valley
                    ...        
    129966              Unknown
    129967         Oregon Other
    129968              Unknown
    129969              Unknown
    129970              Unknown
    Name: region_2, Length: 129971, dtype: object




```python
reviews['price'][reviews['price'].isnull()]
```




    0        NaN
    13       NaN
    30       NaN
    31       NaN
    32       NaN
              ..
    129844   NaN
    129860   NaN
    129863   NaN
    129893   NaN
    129964   NaN
    Name: price, Length: 8996, dtype: float64




```python
reviews['price'].fillna(method='bfill')
```




    0         15.0
    1         15.0
    2         14.0
    3         13.0
    4         65.0
              ... 
    129966    28.0
    129967    75.0
    129968    30.0
    129969    32.0
    129970    21.0
    Name: price, Length: 129971, dtype: float64




```python
reviews['region_2'].fillna("Unknown")
```




    0                   Unknown
    1                   Unknown
    2         Willamette Valley
    3                   Unknown
    4         Willamette Valley
                    ...        
    129966              Unknown
    129967         Oregon Other
    129968              Unknown
    129969              Unknown
    129970              Unknown
    Name: region_2, Length: 129971, dtype: object



### Sustituir valores


```python
reviews['taster_twitter_handle'].replace("@kerinokeefe", "@kerino")
```




    0             @kerino
    1          @vossroger
    2         @paulgwine 
    3                 NaN
    4         @paulgwine 
                 ...     
    129966            NaN
    129967    @paulgwine 
    129968     @vossroger
    129969     @vossroger
    129970     @vossroger
    Name: taster_twitter_handle, Length: 129971, dtype: object



### Renombrar

Podemos renombrar los nombres de los indices o columnas con la función `rename`.


```python
reviews.rename(columns={'points': 'score'})
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>score</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
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
    </tr>
    <tr>
      <th>129966</th>
      <td>Germany</td>
      <td>Notes of honeysuckle and cantaloupe sweeten th...</td>
      <td>Brauneberger Juffer-Sonnenuhr Spätlese</td>
      <td>90</td>
      <td>28.0</td>
      <td>Mosel</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Anna Lee C. Iijima</td>
      <td>NaN</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef) 2013 ...</td>
      <td>Riesling</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef)</td>
    </tr>
    <tr>
      <th>129967</th>
      <td>US</td>
      <td>Citation is given as much as a decade of bottl...</td>
      <td>NaN</td>
      <td>90</td>
      <td>75.0</td>
      <td>Oregon</td>
      <td>Oregon</td>
      <td>Oregon Other</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Citation 2004 Pinot Noir (Oregon)</td>
      <td>Pinot Noir</td>
      <td>Citation</td>
    </tr>
    <tr>
      <th>129968</th>
      <td>France</td>
      <td>Well-drained gravel soil gives this wine its c...</td>
      <td>Kritt</td>
      <td>90</td>
      <td>30.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Gresser 2013 Kritt Gewurztraminer (Als...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Gresser</td>
    </tr>
    <tr>
      <th>129969</th>
      <td>France</td>
      <td>A dry style of Pinot Gris, this is crisp with ...</td>
      <td>NaN</td>
      <td>90</td>
      <td>32.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Marcel Deiss 2012 Pinot Gris (Alsace)</td>
      <td>Pinot Gris</td>
      <td>Domaine Marcel Deiss</td>
    </tr>
    <tr>
      <th>129970</th>
      <td>France</td>
      <td>Big, rich and off-dry, this is powered by inte...</td>
      <td>Lieu-dit Harth Cuvée Caroline</td>
      <td>90</td>
      <td>21.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Schoffit 2012 Lieu-dit Harth Cuvée Car...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Schoffit</td>
    </tr>
  </tbody>
</table>
<p>129971 rows × 13 columns</p>
</div>




```python
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>firstEntry</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>secondEntry</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
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
    </tr>
    <tr>
      <th>129966</th>
      <td>Germany</td>
      <td>Notes of honeysuckle and cantaloupe sweeten th...</td>
      <td>Brauneberger Juffer-Sonnenuhr Spätlese</td>
      <td>90</td>
      <td>28.0</td>
      <td>Mosel</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Anna Lee C. Iijima</td>
      <td>NaN</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef) 2013 ...</td>
      <td>Riesling</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef)</td>
    </tr>
    <tr>
      <th>129967</th>
      <td>US</td>
      <td>Citation is given as much as a decade of bottl...</td>
      <td>NaN</td>
      <td>90</td>
      <td>75.0</td>
      <td>Oregon</td>
      <td>Oregon</td>
      <td>Oregon Other</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Citation 2004 Pinot Noir (Oregon)</td>
      <td>Pinot Noir</td>
      <td>Citation</td>
    </tr>
    <tr>
      <th>129968</th>
      <td>France</td>
      <td>Well-drained gravel soil gives this wine its c...</td>
      <td>Kritt</td>
      <td>90</td>
      <td>30.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Gresser 2013 Kritt Gewurztraminer (Als...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Gresser</td>
    </tr>
    <tr>
      <th>129969</th>
      <td>France</td>
      <td>A dry style of Pinot Gris, this is crisp with ...</td>
      <td>NaN</td>
      <td>90</td>
      <td>32.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Marcel Deiss 2012 Pinot Gris (Alsace)</td>
      <td>Pinot Gris</td>
      <td>Domaine Marcel Deiss</td>
    </tr>
    <tr>
      <th>129970</th>
      <td>France</td>
      <td>Big, rich and off-dry, this is powered by inte...</td>
      <td>Lieu-dit Harth Cuvée Caroline</td>
      <td>90</td>
      <td>21.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Schoffit 2012 Lieu-dit Harth Cuvée Car...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Schoffit</td>
    </tr>
  </tbody>
</table>
<p>129971 rows × 13 columns</p>
</div>



### Combinar datasets


```python
df1 = reviews.iloc[:50000, :]
df2 = reviews.iloc[50000:, :]
df2.shape
```




    (79971, 13)




```python
pd.concat([df1, df2])
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
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
    </tr>
    <tr>
      <th>129966</th>
      <td>Germany</td>
      <td>Notes of honeysuckle and cantaloupe sweeten th...</td>
      <td>Brauneberger Juffer-Sonnenuhr Spätlese</td>
      <td>90</td>
      <td>28.0</td>
      <td>Mosel</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Anna Lee C. Iijima</td>
      <td>NaN</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef) 2013 ...</td>
      <td>Riesling</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef)</td>
    </tr>
    <tr>
      <th>129967</th>
      <td>US</td>
      <td>Citation is given as much as a decade of bottl...</td>
      <td>NaN</td>
      <td>90</td>
      <td>75.0</td>
      <td>Oregon</td>
      <td>Oregon</td>
      <td>Oregon Other</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Citation 2004 Pinot Noir (Oregon)</td>
      <td>Pinot Noir</td>
      <td>Citation</td>
    </tr>
    <tr>
      <th>129968</th>
      <td>France</td>
      <td>Well-drained gravel soil gives this wine its c...</td>
      <td>Kritt</td>
      <td>90</td>
      <td>30.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Gresser 2013 Kritt Gewurztraminer (Als...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Gresser</td>
    </tr>
    <tr>
      <th>129969</th>
      <td>France</td>
      <td>A dry style of Pinot Gris, this is crisp with ...</td>
      <td>NaN</td>
      <td>90</td>
      <td>32.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Marcel Deiss 2012 Pinot Gris (Alsace)</td>
      <td>Pinot Gris</td>
      <td>Domaine Marcel Deiss</td>
    </tr>
    <tr>
      <th>129970</th>
      <td>France</td>
      <td>Big, rich and off-dry, this is powered by inte...</td>
      <td>Lieu-dit Harth Cuvée Caroline</td>
      <td>90</td>
      <td>21.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Schoffit 2012 Lieu-dit Harth Cuvée Car...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Schoffit</td>
    </tr>
  </tbody>
</table>
<p>129971 rows × 13 columns</p>
</div>




```python
dfl = reviews.iloc[:, :8]
dfr = reviews.iloc[:, 8:]
dfr.shape
```




    (129971, 5)




```python
dfl.join(dfr)
```




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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
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
    </tr>
    <tr>
      <th>129966</th>
      <td>Germany</td>
      <td>Notes of honeysuckle and cantaloupe sweeten th...</td>
      <td>Brauneberger Juffer-Sonnenuhr Spätlese</td>
      <td>90</td>
      <td>28.0</td>
      <td>Mosel</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Anna Lee C. Iijima</td>
      <td>NaN</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef) 2013 ...</td>
      <td>Riesling</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef)</td>
    </tr>
    <tr>
      <th>129967</th>
      <td>US</td>
      <td>Citation is given as much as a decade of bottl...</td>
      <td>NaN</td>
      <td>90</td>
      <td>75.0</td>
      <td>Oregon</td>
      <td>Oregon</td>
      <td>Oregon Other</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Citation 2004 Pinot Noir (Oregon)</td>
      <td>Pinot Noir</td>
      <td>Citation</td>
    </tr>
    <tr>
      <th>129968</th>
      <td>France</td>
      <td>Well-drained gravel soil gives this wine its c...</td>
      <td>Kritt</td>
      <td>90</td>
      <td>30.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Gresser 2013 Kritt Gewurztraminer (Als...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Gresser</td>
    </tr>
    <tr>
      <th>129969</th>
      <td>France</td>
      <td>A dry style of Pinot Gris, this is crisp with ...</td>
      <td>NaN</td>
      <td>90</td>
      <td>32.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Marcel Deiss 2012 Pinot Gris (Alsace)</td>
      <td>Pinot Gris</td>
      <td>Domaine Marcel Deiss</td>
    </tr>
    <tr>
      <th>129970</th>
      <td>France</td>
      <td>Big, rich and off-dry, this is powered by inte...</td>
      <td>Lieu-dit Harth Cuvée Caroline</td>
      <td>90</td>
      <td>21.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Schoffit 2012 Lieu-dit Harth Cuvée Car...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Schoffit</td>
    </tr>
  </tbody>
</table>
<p>129971 rows × 13 columns</p>
</div>


