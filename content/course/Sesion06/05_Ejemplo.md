---
title: "Breve introducción a Machine Learning con Scikit-learn"
linktitle: 6.4 Ejemplos
type: book
weight: 2
---


Referencias

[Artificial intelligence MIT Course](https://www.youtube.com/watch?v=_PwhiWxHK8o)

[Kaggle - Elie Kawerk Glass Classification](https://www.kaggle.com/eliekawerk/glass-type-classification-with-machine-learning)

[Random Forest](https://towardsdatascience.com/random-forest-3a55c3aca46d)
Data

[UCI ML Repository - Glass](https://archive.ics.uci.edu/ml/datasets/Glass+Identification)


## Nociones de Machine Learning 

ML
Supervisado / No supervisado
Train/Test

Cross Validation


Machine Learning (ML) es un campo científico que mezcla diversas disciplicas que incluyen, entre otras, la ciencia de computación, estadística, ciencia cognitiva, ingenieria, teoría de optimización.

Hay numerosas aplicaciones de ML, pero destaca su uso en minería de datos. 

ML se divide en dos amplias categorias que son el 
 - **aprendizaje no supervisado** que consiste en técnicas que no hacen uso de la variable a predecir, o de la que se quiere obtener información.
 
 
 - **aprendizaje supervisado** que consiste en técnicas que hacen uso de la variable/s a predecir a la hora de *entrenar* los algoritmos. Unos datos se útilizan para entrenar, y otros para validar. Con esto se intenta evitar el ampliamente conocido efecto de *overfiting*.



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import collections
from time import time


# SKLEARN
import sklearn
# > Preprocesado
from sklearn.preprocessing import StandardScaler
# > Reducción de dimensionalidad por componentes principales
from sklearn.decomposition import PCA
# > Clustering
from sklearn.cluster import KMeans

# > Clasificadores
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# > Selección de modelo
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold,\
                                     cross_val_score, GridSearchCV, \
                                     learning_curve, validation_curve
# > Definir metricas para evaluar el desempeño de clasificadores
from sklearn.metrics import make_scorer
# >> Tipicos scorers
from sklearn.metrics import accuracy_score, recall_score, precision_score 
```

## Scikit-learn

### Comprobar la instalación

Comprobamos la versión de sklearn, debe ser 0.23.2 o superior.


```python
print(sklearn.__version__)
```

    0.23.2
    


```python
#pip install scikit-learn --upgrade
# Y reinicia el Kernel
```

### Aprendizaje no supervisado


```python
nombres_columnas = ['Id', 'RI', 'Na', 'Mg','Al','Si','K','Ca','Ba','Fe','Type']
                
glass_type_dict = {1: 'building windows, float processed',
                   2: 'building windows, non-float processed',
                   3: 'vehicle windows',
                   5: 'containers',
                   6: 'tableware',
                   7: 'headlamps'
                  }
glass_code_dict = {1:'BW-FP', 2:'BW-NFP', 3:'VW', 5:'C', 6:'TW', 7:'HL'}
glass = pd.read_csv('Data/Glass/GlassCSV.csv', names = nombres_columnas)

```


```python
glass = glass.set_index('Id')
glass['Type_str'] = list(map(lambda x: glass_code_dict[x], glass['Type']))
```


```python
columns_sorted = sorted(glass.columns)
input_columns = glass.drop(['Type', 'Type_str'], axis=1).columns.to_list()
```


```python
glass.head()
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
      <th>RI</th>
      <th>Na</th>
      <th>Mg</th>
      <th>Al</th>
      <th>Si</th>
      <th>K</th>
      <th>Ca</th>
      <th>Ba</th>
      <th>Fe</th>
      <th>Type</th>
      <th>Type_str</th>
    </tr>
    <tr>
      <th>Id</th>
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
      <td>1</td>
      <td>1.52101</td>
      <td>13.64</td>
      <td>4.49</td>
      <td>1.10</td>
      <td>71.78</td>
      <td>0.06</td>
      <td>8.75</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>BW-FP</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.51761</td>
      <td>13.89</td>
      <td>3.60</td>
      <td>1.36</td>
      <td>72.73</td>
      <td>0.48</td>
      <td>7.83</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>BW-FP</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.51618</td>
      <td>13.53</td>
      <td>3.55</td>
      <td>1.54</td>
      <td>72.99</td>
      <td>0.39</td>
      <td>7.78</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>BW-FP</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.51766</td>
      <td>13.21</td>
      <td>3.69</td>
      <td>1.29</td>
      <td>72.61</td>
      <td>0.57</td>
      <td>8.22</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>BW-FP</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.51742</td>
      <td>13.27</td>
      <td>3.62</td>
      <td>1.24</td>
      <td>73.08</td>
      <td>0.55</td>
      <td>8.07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>BW-FP</td>
    </tr>
  </tbody>
</table>
</div>



# Análisis descriptivo


```python
np.random.seed(10)
glass.groupby(by='Type_str').count().iloc[:,0].sort_values(ascending=False).plot.bar(color=np.random.random_sample((6,3)))
plt.xticks(rotation=0)
plt.xlabel('Tipo vidrio')
plt.ylabel('Número de muestras')
plt.title('Número de muestras por tipo de vidrio')
plt.show()
```


    
![png](05_Ejemplo_files/05_Ejemplo_16_0.png)
    



```python
glass.drop('Type', axis=1).describe().transpose()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RI</td>
      <td>214.0</td>
      <td>1.518365</td>
      <td>0.003037</td>
      <td>1.51115</td>
      <td>1.516523</td>
      <td>1.51768</td>
      <td>1.519157</td>
      <td>1.53393</td>
    </tr>
    <tr>
      <td>Na</td>
      <td>214.0</td>
      <td>13.407850</td>
      <td>0.816604</td>
      <td>10.73000</td>
      <td>12.907500</td>
      <td>13.30000</td>
      <td>13.825000</td>
      <td>17.38000</td>
    </tr>
    <tr>
      <td>Mg</td>
      <td>214.0</td>
      <td>2.684533</td>
      <td>1.442408</td>
      <td>0.00000</td>
      <td>2.115000</td>
      <td>3.48000</td>
      <td>3.600000</td>
      <td>4.49000</td>
    </tr>
    <tr>
      <td>Al</td>
      <td>214.0</td>
      <td>1.444907</td>
      <td>0.499270</td>
      <td>0.29000</td>
      <td>1.190000</td>
      <td>1.36000</td>
      <td>1.630000</td>
      <td>3.50000</td>
    </tr>
    <tr>
      <td>Si</td>
      <td>214.0</td>
      <td>72.650935</td>
      <td>0.774546</td>
      <td>69.81000</td>
      <td>72.280000</td>
      <td>72.79000</td>
      <td>73.087500</td>
      <td>75.41000</td>
    </tr>
    <tr>
      <td>K</td>
      <td>214.0</td>
      <td>0.497056</td>
      <td>0.652192</td>
      <td>0.00000</td>
      <td>0.122500</td>
      <td>0.55500</td>
      <td>0.610000</td>
      <td>6.21000</td>
    </tr>
    <tr>
      <td>Ca</td>
      <td>214.0</td>
      <td>8.956963</td>
      <td>1.423153</td>
      <td>5.43000</td>
      <td>8.240000</td>
      <td>8.60000</td>
      <td>9.172500</td>
      <td>16.19000</td>
    </tr>
    <tr>
      <td>Ba</td>
      <td>214.0</td>
      <td>0.175047</td>
      <td>0.497219</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>3.15000</td>
    </tr>
    <tr>
      <td>Fe</td>
      <td>214.0</td>
      <td>0.057009</td>
      <td>0.097439</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.100000</td>
      <td>0.51000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,7))
glass[input_columns].boxplot()
plt.title('Gráfico de cajas para cada atributo')
plt.ylabel('Peso (% en oxido)')
plt.show()
```


    
![png](05_Ejemplo_files/05_Ejemplo_18_0.png)
    



```python
fig, ax = plt.subplots(3,3,figsize=(15,8), gridspec_kw = {'hspace':0.7})
glass[input_columns+['Type_str']].boxplot(by='Type_str', ax=ax)
[ax[ii,jj].set_ylabel('Peso (% en oxido)') for ii,jj in itertools.product(range(3), range(3)) if columns_sorted[3*ii+jj]!='RI']
[ax[ii,jj].set_xlabel('Tipo cristal') for ii,jj in itertools.product(range(3), range(3))]

plt.suptitle('Gráfico de cajas por tipo para cada columna')
plt.show()

```

    C:\Users\win10\Anaconda3\lib\site-packages\pandas\plotting\_matplotlib\boxplot.py:355: UserWarning: When passing multiple axes, sharex and sharey are ignored. These settings must be specified when creating axes
      **kwds
    


    
![png](05_Ejemplo_files/05_Ejemplo_19_1.png)
    


## Reescalado de las variables


```python
X = glass.drop(['Type', 'Type_str'], axis=1).to_numpy()
X
```




    array([[ 1.52101, 13.64   ,  4.49   , ...,  8.75   ,  0.     ,  0.     ],
           [ 1.51761, 13.89   ,  3.6    , ...,  7.83   ,  0.     ,  0.     ],
           [ 1.51618, 13.53   ,  3.55   , ...,  7.78   ,  0.     ,  0.     ],
           ...,
           [ 1.52065, 14.36   ,  0.     , ...,  8.44   ,  1.64   ,  0.     ],
           [ 1.51651, 14.38   ,  0.     , ...,  8.48   ,  1.57   ,  0.     ],
           [ 1.51711, 14.23   ,  0.     , ...,  8.62   ,  1.67   ,  0.     ]])




```python
# Usamos el scalado estandar (Z-score: z_i=(x_i-media)/desviacion_estandar)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```


```python
glass_norm = pd.DataFrame(X_scaled, columns=input_columns)
glass_norm.loc[:, 'Type_str'] = glass['Type_str']

fig, ax = plt.subplots(3,3,figsize=(15,8), gridspec_kw = {'hspace':0.7})
glass_norm.boxplot(by='Type_str', ax=ax)
[ax[ii,jj].set_ylabel('Peso (% en oxido)') for ii,jj in itertools.product(range(3), range(3)) if columns_sorted[3*ii+jj]!='RI']
[ax[ii,jj].set_xlabel('Tipo cristal') for ii,jj in itertools.product(range(3), range(3))]

plt.suptitle('Gráfico de cajas por tipo para cada columna')
plt.show()

```

    C:\Users\win10\Anaconda3\lib\site-packages\pandas\plotting\_matplotlib\boxplot.py:355: UserWarning: When passing multiple axes, sharex and sharey are ignored. These settings must be specified when creating axes
      **kwds
    


    
![png](05_Ejemplo_files/05_Ejemplo_23_1.png)
    


## Reducción de la dimensionalidad: Principal Component Analysis (PCA)


```python
###########################
# PCA
###########################


fig, ax = plt.subplots(2,1, figsize=(10,8), gridspec_kw={'height_ratios':[0.7,0.3], 'hspace':0.6})

pca = PCA().fit(X_scaled)
ax[0].plot(np.arange(1,len(pca.explained_variance_ratio_)+1),   
    np.cumsum(pca.explained_variance_ratio_), marker=".")
ax[0].set_xlabel('Number of components')
ax[0].set_ylabel('Cumulative explained variance')
ax[0].set_title("Explained variance vs. Number of components")

# PC explained variance 
pca3 = PCA(n_components=3)
pca3.fit(X_scaled)
# Component meaning
ax[1].matshow(pca3.components_, cmap="bwr",vmin=-1,vmax=1)
ax[1].set_yticks(range(3))
ax[1].set_yticklabels(['1st Comp','2nd Comp', '3rd Comp'], fontsize=10)
#plt.colorbar()
ax[1].set_xticks(range(len(input_columns)), )
ax[1].set_xticklabels(input_columns, rotation=0,ha='center')
ax[1].set_ylim(2.5,-0.5)
[ax[1].text(jj,ii, f'{pca3.components_[ii,jj]:1.2f}', va='center', ha='center', fontsize=8) for ii, jj in itertools.product(range(3), range(len(input_columns)))]
ax[1].set_title('Correlations', pad=30)
plt.show()
```


    
![png](05_Ejemplo_files/05_Ejemplo_25_0.png)
    



```python

# 2PC plot
# > Performe PCA
pca2 = PCA(n_components=2)
pca2.fit(X_scaled)
X2 = pca2.transform(X_scaled)
# > Plot dimensionality reduction to 2PCs
# >> Scatter all points, with the corresponding color. We set 
#      transparency to 20%
plt.figure(figsize=(10,6))
plt.scatter(X2[:, 0], X2[:, 1], alpha=0.4, marker="o", s=28,c=glass['Type'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title("2PC Scatterplot")
plt.show()
```


    
![png](05_Ejemplo_files/05_Ejemplo_26_0.png)
    


## Clustering


### KMeans



```python
kmeans = KMeans(n_clusters=6, random_state=0).fit(X_scaled)
cluster_index = kmeans.labels_

collections.Counter(kmeans.labels_)


glass['cluster_KMeans'] = cluster_index

```


```python

def print_cluster_purity(data_pd, cluster_col):

    for cluster_idx, cluster_pd in glass.groupby(by=cluster_col):

        print(f'Cluster {cluster_idx}')
        print('\t',str(collections.Counter(cluster_pd['Type'])).replace('Counter',''))

print_cluster_purity(glass, 'cluster_KMeans')   
```

    Cluster 0
    	 ({2: 20, 1: 15, 3: 3, 5: 2})
    Cluster 1
    	 ({5: 2})
    Cluster 2
    	 ({7: 24, 5: 1, 6: 1})
    Cluster 3
    	 ({2: 7})
    Cluster 4
    	 ({1: 17, 2: 7, 6: 7, 5: 5, 7: 3, 3: 2})
    Cluster 5
    	 ({2: 42, 1: 38, 3: 12, 5: 3, 7: 2, 6: 1})
    

## Agglomerative Clustering


```python
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering().fit(X_scaled)
clustering
```




    AgglomerativeClustering()




```python
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram (source: scikit documentation)
    
    
    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# setting distance_threshold=0 ensures we compute the full tree.
linkage_list = ['ward', 'average', 'complete', 'single']
d_th = 15
model_agg = AgglomerativeClustering(distance_threshold=d_th, n_clusters=None, linkage=linkage_list[0])

model_agg = model_agg.fit(X_scaled)

fig, ax = plt.subplots(figsize=(15,10))
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model_agg, truncate_mode='level', p=10, ax=ax)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.axhline(d_th, linestyle='--', color='gray', zorder=-1)
plt.show()

glass['cluster_agg'] = model_agg.labels_
print_cluster_purity(glass, 'cluster_agg')
```


    
![png](05_Ejemplo_files/05_Ejemplo_33_0.png)
    


    Cluster 0
    	 ({2: 38, 1: 32, 3: 8, 6: 8, 5: 6, 7: 2})
    Cluster 1
    	 ({7: 25, 5: 1, 6: 1})
    Cluster 2
    	 ({2: 8, 5: 2})
    Cluster 3
    	 ({5: 2})
    Cluster 4
    	 ({2: 26, 1: 18, 3: 5, 5: 2})
    Cluster 5
    	 ({1: 20, 2: 4, 3: 4, 7: 2})
    

# Aprendizaje supervisado 

# Regresión

# Clasificación

Usaremos algunos algoritmos de clasificación de Sklearn:

 - [SVM, Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svc#sklearn.svm.SVC)
 
 - [KNN, K nearest neighbours](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier),  [Info](https://scikit-learn.org/stable/modules/neighbors.html#nearest-centroid-classifier)
 
 - [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
 
 - [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforest#sklearn.ensemble.RandomForestClassifier)
 
 ---
 

Comunmente se pretende entrenar un modelo y despues evaluar como de bueno es.

Para esto podemos dividir los datos. Esto podemos hacerlo mediante [*train_test_split*](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) de Sklearn.


```python

# Fracción de los datos reservados para la validación.
test_size = 0.2

# La variable a clasificar.
tarjet_values = glass['Type']

# Datos separados por entrenamiento / validación
X_train, X_test, y_train, y_test = train_test_split(X_scaled, tarjet_values, test_size=test_size, random_state = 7)
# La variable random_state se introduce para reproducibilidad. 
```


```python
print(f'Numero de instancias \n\t Entrenamiento\t {X_train.shape[0]:3d}\n\t Validación\t {X_test.shape[0]:3d}')
print()
print(f'Porcentaje Validación {(X_test.shape[0]/X_scaled.shape[0]):1.3f}')
```

    Numero de instancias 
    	 Entrenamiento	 171
    	 Validación	  43
    
    Porcentaje Validación 0.201
    

Cargaremos diversos modelos, los entrenaremos mediante [Cross Validation](https://towardsdatascience.com/cross-validation-430d9a5fee22) y veremos como han desempeñado en los datos de entrenamiento.

    La CV o validación cruzada consiste en dividir los datos en $k$ diferentes particiones de misma dimension (número de muestras). Los modelos se entrenan con una parte de los datos y se evaluan en la otra. Se obtiene una media y desviación de la métrica deseada (comunmente la exactitud o *accuracy*). Queremos una alta exactitud media y una baja dispersión.
    
Usaremos *StratifiedKfolds* y *cross_val_score* para 


```python

seed = 10
n_estimators = 200

model_list = [ ('SVC', SVC(random_state=seed)), 
               ('KNN',  KNeighborsClassifier()),
               ('DT', DecisionTreeClassifier()),
               ('RF', RandomForestClassifier(random_state=seed, n_estimators=n_estimators))]

results, names, times  = [], [] , []
n_folds = 4
scoring_list = ['accuracy', 'precision', 'recall']
scoring = 'accuracy'

# Inicializamos dataframe para 
results_pd = pd.DataFrame([], index = [model[0] for model in model_list], columns=['mean_accuracy (%)', 'std_accuracy (%)', 'cpu_time (s)'])

for clf_name, clf_model in model_list:

    start = time() # Empezamos el crono

    # Creamos objeto de K particiones
    kfold = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    # Resultados para el entrenamiento con *n_folds* particiones
    cv_results = cross_val_score(clf_model, X_train, y_train, cv=kfold, scoring = scoring, n_jobs=-1) 


    t_elapsed = time() - start # Calcular tiempo
    
    # Almacenar resultados
    results.append(cv_results) # Score
    names.append(clf_name)     # Nombre de clasificador
    times.append(t_elapsed)    # Tiempo de computación

    
    results_pd.loc[clf_name,:]=[100*cv_results.mean(), 100*cv_results.std(), t_elapsed ]
    
    msg = f'{clf_name}-{scoring}: \t{100*cv_results.mean():3.2f} +- {100*cv_results.std():3.2f} % \t [{t_elapsed} s]'
    print(msg)

```

    SVC-accuracy: 	67.24 +- 3.62 % 	 [0.031243324279785156 s]
    KNN-accuracy: 	64.53 +- 5.29 % 	 [0.01562047004699707 s]
    DT-accuracy: 	63.71 +- 12.96 % 	 [0.015620708465576172 s]
    RF-accuracy: 	73.86 +- 2.33 % 	 [0.3884427547454834 s]
    


```python
results_pd
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
      <th>mean_accuracy (%)</th>
      <th>std_accuracy (%)</th>
      <th>cpu_time (s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SVC</td>
      <td>67.2365</td>
      <td>3.62059</td>
      <td>0.0312433</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>64.5299</td>
      <td>5.28603</td>
      <td>0.0156205</td>
    </tr>
    <tr>
      <td>DT</td>
      <td>63.7108</td>
      <td>12.9565</td>
      <td>0.0156207</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>73.8604</td>
      <td>2.32656</td>
      <td>0.388443</td>
    </tr>
  </tbody>
</table>
</div>



### Ajuste de hiperparámetros: GridSearchCV

Vamos a probar diferentes valores de parámetros de un RF.


```python
# Creamos el objeto clasificador
RF_cls = RandomForestClassifier()


# Introducimos varias metricas a modo de ejemplo, tan solo usaremos 'accuracy'
#  Las otras dos las dejo para futuro uso
scoring_dict = {"accuracy":  make_scorer(accuracy_score), 
                "recall":    make_scorer(recall_score,    average="weighted"),
                "precision": make_scorer(precision_score, average="weighted")}
# Las métricas recall y precisión requieren especificar el parámetro average para problemas
#  multiclase

# Introduciomos los parámetros sobre los que buscaremos
tuned_parameters = {
    'n_estimators': [100, 200, 300, 400], # number of estimators
    #'criterion': ['gini', 'entropy'],    # splitting criterion
    'max_features':[0.05 , 0.1],          # maximum features used at each split
    'max_depth': [None, 5],               # max depth of the trees
    'min_samples_split': [0.005, 0.01],   # mininal samples in leafs
    }

# Especificamos el clasificador, los valores de parámetros, la métrica a usar, las K particiones para el CV y
#  usamos todos los procesadores del ordenador (njobs=-1, numero de trabajos en paralelo)
k_folds = 4
tuned_RF =  GridSearchCV(RF_cls, param_grid = tuned_parameters, scoring = scoring_dict['accuracy'], cv=k_folds, n_jobs=-1)

# Entrenamos para *k_folds* particiones y *tuned_parameters* malla de parámetros
tuned_RF.fit(X_train, y_train)
```




    GridSearchCV(cv=4, estimator=RandomForestClassifier(), n_jobs=-1,
                 param_grid={'max_depth': [None, 5], 'max_features': [0.05, 0.1],
                             'min_samples_split': [0.005, 0.01],
                             'n_estimators': [100, 200, 300, 400]},
                 scoring=make_scorer(accuracy_score))



Esto nos devuelve como veis un GridSearchCV, que tiene atributos


```python
[attr for attr in dir(tuned_RF) if '__' not in attr]
```




    ['_abc_impl',
     '_check_is_fitted',
     '_check_n_features',
     '_estimator_type',
     '_format_results',
     '_get_param_names',
     '_get_tags',
     '_more_tags',
     '_pairwise',
     '_repr_html_',
     '_repr_html_inner',
     '_repr_mimebundle_',
     '_required_parameters',
     '_run_search',
     '_validate_data',
     'best_estimator_',
     'best_index_',
     'best_params_',
     'best_score_',
     'classes_',
     'cv',
     'cv_results_',
     'decision_function',
     'error_score',
     'estimator',
     'fit',
     'get_params',
     'iid',
     'inverse_transform',
     'multimetric_',
     'n_features_in_',
     'n_jobs',
     'n_splits_',
     'param_grid',
     'pre_dispatch',
     'predict',
     'predict_log_proba',
     'predict_proba',
     'refit',
     'refit_time_',
     'return_train_score',
     'score',
     'scorer_',
     'scoring',
     'set_params',
     'transform',
     'verbose']




```python
tuned_RF.best_estimator_
```




    RandomForestClassifier(max_features=0.1, min_samples_split=0.01,
                           n_estimators=300)




```python
tuned_RF.cv_results_
```




    {'mean_fit_time': array([0.14908546, 0.27890319, 0.47095716, 0.70415729, 0.16582334,
            0.30010939, 0.47633469, 0.6424318 , 0.16171855, 0.29711807,
            0.61474818, 0.64790469, 0.14803153, 0.37406564, 0.55442089,
            0.68279034, 0.17039615, 0.44262999, 0.57217008, 0.72507274,
            0.1736477 , 0.30471683, 0.43290406, 0.5734849 , 0.15148193,
            0.40305036, 0.51105464, 0.7256456 , 0.19287503, 0.377877  ,
            0.44644654, 0.58700657]),
     'std_fit_time': array([0.00914073, 0.00832114, 0.00835548, 0.01445859, 0.00786493,
            0.00800933, 0.02750493, 0.01658586, 0.01186647, 0.01505596,
            0.04201937, 0.03580855, 0.00828019, 0.01611874, 0.01159653,
            0.01025783, 0.01371943, 0.01473541, 0.02533847, 0.02403158,
            0.00784573, 0.01096724, 0.00712165, 0.01062993, 0.00716527,
            0.01538994, 0.02117806, 0.02096453, 0.00809815, 0.00995392,
            0.00415028, 0.01340512]),
     'mean_score_time': array([0.01408887, 0.02774698, 0.03773135, 0.054941  , 0.01811779,
            0.02515697, 0.04730844, 0.05436116, 0.01862001, 0.02520311,
            0.04329383, 0.06046093, 0.0110724 , 0.02367002, 0.04229873,
            0.05503219, 0.01511711, 0.03081679, 0.06105644, 0.0528329 ,
            0.01812196, 0.02767795, 0.03322512, 0.04985476, 0.01458776,
            0.02823275, 0.03828776, 0.0629465 , 0.03274137, 0.03122735,
            0.04026943, 0.04062736]),
     'std_score_time': array([4.02093164e-03, 3.34922105e-03, 3.29107480e-03, 5.56277801e-03,
            3.47521124e-03, 5.02408636e-03, 9.53066561e-03, 4.01744196e-03,
            8.66893183e-04, 4.13792995e-03, 4.14510854e-03, 1.00610358e-02,
            1.01210295e-03, 3.86341235e-03, 3.48853273e-03, 5.56962602e-03,
            3.01125762e-03, 6.71331963e-03, 1.23599572e-02, 3.28889303e-03,
            3.46759491e-03, 4.36557287e-03, 4.16413691e-03, 9.03690754e-04,
            4.58503461e-03, 4.73722138e-03, 3.50141719e-03, 1.31029239e-02,
            2.25340009e-02, 7.17917089e-03, 1.95854550e-05, 9.80406753e-03]),
     'param_max_depth': masked_array(data=[None, None, None, None, None, None, None, None, None,
                        None, None, None, None, None, None, None, 5, 5, 5, 5,
                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                  mask=[False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False],
            fill_value='?',
                 dtype=object),
     'param_max_features': masked_array(data=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05,
                        0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1],
                  mask=[False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False],
            fill_value='?',
                 dtype=object),
     'param_min_samples_split': masked_array(data=[0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01,
                        0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01,
                        0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01,
                        0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01],
                  mask=[False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False],
            fill_value='?',
                 dtype=object),
     'param_n_estimators': masked_array(data=[100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300,
                        400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200,
                        300, 400, 100, 200, 300, 400, 100, 200, 300, 400],
                  mask=[False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False],
            fill_value='?',
                 dtype=object),
     'params': [{'max_depth': None,
       'max_features': 0.05,
       'min_samples_split': 0.005,
       'n_estimators': 100},
      {'max_depth': None,
       'max_features': 0.05,
       'min_samples_split': 0.005,
       'n_estimators': 200},
      {'max_depth': None,
       'max_features': 0.05,
       'min_samples_split': 0.005,
       'n_estimators': 300},
      {'max_depth': None,
       'max_features': 0.05,
       'min_samples_split': 0.005,
       'n_estimators': 400},
      {'max_depth': None,
       'max_features': 0.05,
       'min_samples_split': 0.01,
       'n_estimators': 100},
      {'max_depth': None,
       'max_features': 0.05,
       'min_samples_split': 0.01,
       'n_estimators': 200},
      {'max_depth': None,
       'max_features': 0.05,
       'min_samples_split': 0.01,
       'n_estimators': 300},
      {'max_depth': None,
       'max_features': 0.05,
       'min_samples_split': 0.01,
       'n_estimators': 400},
      {'max_depth': None,
       'max_features': 0.1,
       'min_samples_split': 0.005,
       'n_estimators': 100},
      {'max_depth': None,
       'max_features': 0.1,
       'min_samples_split': 0.005,
       'n_estimators': 200},
      {'max_depth': None,
       'max_features': 0.1,
       'min_samples_split': 0.005,
       'n_estimators': 300},
      {'max_depth': None,
       'max_features': 0.1,
       'min_samples_split': 0.005,
       'n_estimators': 400},
      {'max_depth': None,
       'max_features': 0.1,
       'min_samples_split': 0.01,
       'n_estimators': 100},
      {'max_depth': None,
       'max_features': 0.1,
       'min_samples_split': 0.01,
       'n_estimators': 200},
      {'max_depth': None,
       'max_features': 0.1,
       'min_samples_split': 0.01,
       'n_estimators': 300},
      {'max_depth': None,
       'max_features': 0.1,
       'min_samples_split': 0.01,
       'n_estimators': 400},
      {'max_depth': 5,
       'max_features': 0.05,
       'min_samples_split': 0.005,
       'n_estimators': 100},
      {'max_depth': 5,
       'max_features': 0.05,
       'min_samples_split': 0.005,
       'n_estimators': 200},
      {'max_depth': 5,
       'max_features': 0.05,
       'min_samples_split': 0.005,
       'n_estimators': 300},
      {'max_depth': 5,
       'max_features': 0.05,
       'min_samples_split': 0.005,
       'n_estimators': 400},
      {'max_depth': 5,
       'max_features': 0.05,
       'min_samples_split': 0.01,
       'n_estimators': 100},
      {'max_depth': 5,
       'max_features': 0.05,
       'min_samples_split': 0.01,
       'n_estimators': 200},
      {'max_depth': 5,
       'max_features': 0.05,
       'min_samples_split': 0.01,
       'n_estimators': 300},
      {'max_depth': 5,
       'max_features': 0.05,
       'min_samples_split': 0.01,
       'n_estimators': 400},
      {'max_depth': 5,
       'max_features': 0.1,
       'min_samples_split': 0.005,
       'n_estimators': 100},
      {'max_depth': 5,
       'max_features': 0.1,
       'min_samples_split': 0.005,
       'n_estimators': 200},
      {'max_depth': 5,
       'max_features': 0.1,
       'min_samples_split': 0.005,
       'n_estimators': 300},
      {'max_depth': 5,
       'max_features': 0.1,
       'min_samples_split': 0.005,
       'n_estimators': 400},
      {'max_depth': 5,
       'max_features': 0.1,
       'min_samples_split': 0.01,
       'n_estimators': 100},
      {'max_depth': 5,
       'max_features': 0.1,
       'min_samples_split': 0.01,
       'n_estimators': 200},
      {'max_depth': 5,
       'max_features': 0.1,
       'min_samples_split': 0.01,
       'n_estimators': 300},
      {'max_depth': 5,
       'max_features': 0.1,
       'min_samples_split': 0.01,
       'n_estimators': 400}],
     'split0_test_score': array([0.7037037 , 0.7037037 , 0.74074074, 0.66666667, 0.74074074,
            0.74074074, 0.66666667, 0.7037037 , 0.66666667, 0.66666667,
            0.74074074, 0.66666667, 0.62962963, 0.7037037 , 0.7037037 ,
            0.66666667, 0.66666667, 0.74074074, 0.85185185, 0.74074074,
            0.7037037 , 0.77777778, 0.7037037 , 0.77777778, 0.7037037 ,
            0.7037037 , 0.7037037 , 0.77777778, 0.66666667, 0.85185185,
            0.74074074, 0.7037037 ]),
     'split1_test_score': array([0.74074074, 0.74074074, 0.66666667, 0.77777778, 0.74074074,
            0.74074074, 0.74074074, 0.77777778, 0.74074074, 0.74074074,
            0.7037037 , 0.74074074, 0.74074074, 0.74074074, 0.74074074,
            0.77777778, 0.62962963, 0.66666667, 0.66666667, 0.66666667,
            0.62962963, 0.66666667, 0.66666667, 0.66666667, 0.66666667,
            0.66666667, 0.66666667, 0.66666667, 0.7037037 , 0.66666667,
            0.7037037 , 0.66666667]),
     'split2_test_score': array([0.77777778, 0.81481481, 0.77777778, 0.81481481, 0.88888889,
            0.88888889, 0.81481481, 0.81481481, 0.81481481, 0.81481481,
            0.85185185, 0.85185185, 0.81481481, 0.81481481, 0.88888889,
            0.81481481, 0.81481481, 0.85185185, 0.81481481, 0.81481481,
            0.77777778, 0.88888889, 0.85185185, 0.85185185, 0.85185185,
            0.81481481, 0.81481481, 0.77777778, 0.81481481, 0.81481481,
            0.85185185, 0.85185185]),
     'split3_test_score': array([0.65384615, 0.61538462, 0.65384615, 0.69230769, 0.65384615,
            0.65384615, 0.73076923, 0.65384615, 0.65384615, 0.69230769,
            0.69230769, 0.73076923, 0.69230769, 0.73076923, 0.69230769,
            0.65384615, 0.61538462, 0.69230769, 0.65384615, 0.65384615,
            0.69230769, 0.69230769, 0.69230769, 0.65384615, 0.61538462,
            0.65384615, 0.73076923, 0.69230769, 0.61538462, 0.61538462,
            0.65384615, 0.61538462]),
     'mean_test_score': array([0.71901709, 0.71866097, 0.70975783, 0.73789174, 0.75605413,
            0.75605413, 0.73824786, 0.73753561, 0.71901709, 0.72863248,
            0.747151  , 0.74750712, 0.71937322, 0.74750712, 0.75641026,
            0.72827635, 0.68162393, 0.73789174, 0.74679487, 0.71901709,
            0.7008547 , 0.75641026, 0.72863248, 0.73753561, 0.70940171,
            0.70975783, 0.7289886 , 0.72863248, 0.70014245, 0.73717949,
            0.73753561, 0.70940171]),
     'std_test_score': array([0.04584345, 0.07180321, 0.05140432, 0.06053712, 0.08449932,
            0.08449932, 0.05255595, 0.06272964, 0.06449213, 0.05642087,
            0.06304525, 0.06661258, 0.06773501, 0.04115521, 0.07855451,
            0.06941704, 0.0791432 , 0.0709682 , 0.08764085, 0.06449213,
            0.05261021, 0.08684787, 0.07239425, 0.08172204, 0.0880199 ,
            0.06335729, 0.05452673, 0.04997443, 0.07325802, 0.09872062,
            0.07284744, 0.0880199 ]),
     'rank_test_score': array([22, 25, 26, 11,  3,  3,  9, 12, 22, 17,  7,  5, 21,  6,  1, 20, 32,
            10,  8, 22, 30,  1, 17, 12, 28, 26, 16, 17, 31, 15, 12, 28])}



**Extraigamos y ordenemos los resultados**


```python

def GS_results2pandas(tuned_GS):
    # La media de la métrica escogida (del CV) para cada combinación de parámetros
    means = tuned_GS.cv_results_['mean_test_score']
    # La desviación estandar de la métrica escogida (del CV) para cada combinación
    stds = tuned_GS.cv_results_['std_test_score']


    # Inicializamos un dataframe
    #   > Instancias igual al número de combinaciones de parámetros
    #   > Las columnas serán la "media de la métrica para el CV", "la desviación de la métrica"
    #      y cada uno de los hiperparámetros

    gridCV_results_pd = pd.DataFrame([], index=np.arange(len(means)), 
                                     columns=['mean_score', 'std_score'] + list(tuned_GS.cv_results_['params'][0].keys()) )

    # 'tuned_RF.cv_results_['params'][0].keys()' es equivalente a *tuned_parameters.keys()*


    # Vayamos por cada valor de interes para volcarlo todo al pandas de resultados
    current_row = 0
    for mean, std, params in zip(means, stds, tuned_GS.cv_results_['params']):

        # Añadimos el resultado de la métrica (Accuracy, ...)
        gridCV_results_pd.loc[current_row, 'mean_score'] = mean
        gridCV_results_pd.loc[current_row, 'std_score']  = std

        # Añadimos los hiperparámetros 
        for col in params.keys():
            gridCV_results_pd.loc[current_row, col] = params[col] 

        #print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        #print("")
        # Siguiente fila (combinación de parámetros)
        current_row += 1
    return(gridCV_results_pd.sort_values(by=['mean_score', 'std_score'],ascending=[False, True]))
```


```python
GS_results2pandas(tuned_RF)
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
      <th>mean_score</th>
      <th>std_score</th>
      <th>max_depth</th>
      <th>max_features</th>
      <th>min_samples_split</th>
      <th>n_estimators</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>14</td>
      <td>0.813123</td>
      <td>0.0456962</td>
      <td>None</td>
      <td>0.1</td>
      <td>0.01</td>
      <td>300</td>
    </tr>
    <tr>
      <td>0</td>
      <td>0.807171</td>
      <td>0.054977</td>
      <td>None</td>
      <td>0.05</td>
      <td>0.005</td>
      <td>100</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.795681</td>
      <td>0.0434759</td>
      <td>None</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>100</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.795543</td>
      <td>0.0405937</td>
      <td>None</td>
      <td>0.05</td>
      <td>0.005</td>
      <td>200</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.789729</td>
      <td>0.0355635</td>
      <td>None</td>
      <td>0.1</td>
      <td>0.005</td>
      <td>300</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.789729</td>
      <td>0.0455632</td>
      <td>None</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>200</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.789729</td>
      <td>0.0585499</td>
      <td>None</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>300</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.789729</td>
      <td>0.0585499</td>
      <td>None</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>400</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.78959</td>
      <td>0.0361144</td>
      <td>None</td>
      <td>0.05</td>
      <td>0.005</td>
      <td>400</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.78959</td>
      <td>0.0361144</td>
      <td>None</td>
      <td>0.1</td>
      <td>0.005</td>
      <td>200</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.78959</td>
      <td>0.0361144</td>
      <td>None</td>
      <td>0.1</td>
      <td>0.01</td>
      <td>400</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.78959</td>
      <td>0.0429545</td>
      <td>None</td>
      <td>0.1</td>
      <td>0.01</td>
      <td>100</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.783915</td>
      <td>0.0520287</td>
      <td>None</td>
      <td>0.05</td>
      <td>0.005</td>
      <td>300</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.783776</td>
      <td>0.0291882</td>
      <td>None</td>
      <td>0.1</td>
      <td>0.01</td>
      <td>200</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.783776</td>
      <td>0.0407823</td>
      <td>None</td>
      <td>0.1</td>
      <td>0.005</td>
      <td>100</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.777962</td>
      <td>0.0471695</td>
      <td>None</td>
      <td>0.1</td>
      <td>0.005</td>
      <td>400</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.760659</td>
      <td>0.0489029</td>
      <td>5</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>100</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0.754845</td>
      <td>0.0518116</td>
      <td>5</td>
      <td>0.1</td>
      <td>0.01</td>
      <td>400</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.754707</td>
      <td>0.0404056</td>
      <td>5</td>
      <td>0.05</td>
      <td>0.005</td>
      <td>400</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.748893</td>
      <td>0.0430854</td>
      <td>5</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>400</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.748893</td>
      <td>0.0430854</td>
      <td>5</td>
      <td>0.1</td>
      <td>0.005</td>
      <td>300</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.748893</td>
      <td>0.0430854</td>
      <td>5</td>
      <td>0.1</td>
      <td>0.005</td>
      <td>400</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.748754</td>
      <td>0.0329594</td>
      <td>5</td>
      <td>0.05</td>
      <td>0.005</td>
      <td>300</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.74294</td>
      <td>0.0311517</td>
      <td>5</td>
      <td>0.1</td>
      <td>0.005</td>
      <td>200</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.74294</td>
      <td>0.0352256</td>
      <td>5</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>200</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.74294</td>
      <td>0.0352256</td>
      <td>5</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>300</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.737126</td>
      <td>0.0280523</td>
      <td>5</td>
      <td>0.1</td>
      <td>0.01</td>
      <td>300</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.737126</td>
      <td>0.0325169</td>
      <td>5</td>
      <td>0.05</td>
      <td>0.005</td>
      <td>100</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.731451</td>
      <td>0.0489117</td>
      <td>5</td>
      <td>0.1</td>
      <td>0.005</td>
      <td>100</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.725498</td>
      <td>0.0360388</td>
      <td>5</td>
      <td>0.05</td>
      <td>0.005</td>
      <td>200</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.713732</td>
      <td>0.0323103</td>
      <td>5</td>
      <td>0.1</td>
      <td>0.01</td>
      <td>100</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.713594</td>
      <td>0.0170367</td>
      <td>5</td>
      <td>0.1</td>
      <td>0.01</td>
      <td>200</td>
    </tr>
  </tbody>
</table>
</div>




```python
RF_final = tuned_RF.best_estimator_
RF_final
```




    RandomForestClassifier(max_features=0.1, min_samples_split=0.01,
                           n_estimators=300)



¿Cuánto mejorara otro clasificador mediante esta técnica?

¡Pruebalo!

En la documentación aparecen los parámetros. Repite el proceso con ellos.

### Validación


```python

predicted = RF_final.predict(X_test)

disp = metrics.plot_confusion_matrix(RF_final, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")

plt.show()
```


    
![png](05_Ejemplo_files/05_Ejemplo_57_0.png)
    



```python
print("Classification report for classifier %s:\n%s\n"
      % (clf_svm, metrics.classification_report(y_test, predicted)))
```

    Classification report for classifier SVC(gamma=0.1):
                  precision    recall  f1-score   support
    
               1       0.53      0.90      0.67        10
               2       0.94      0.73      0.82        22
               3       0.00      0.00      0.00         3
               5       1.00      1.00      1.00         2
               6       0.50      1.00      0.67         1
               7       1.00      1.00      1.00         5
    
        accuracy                           0.77        43
       macro avg       0.66      0.77      0.69        43
    weighted avg       0.78      0.77      0.75        43
    
    
    

    C:\Users\win10\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1221: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python

```
