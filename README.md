# Mesh denoising via L0 Minimization
# Abstract
This work is an implementation  of the paper Mesh denoising via L0 minimization proposed by Lei He (Texas A&M University) and 
Scott Schaefer (Texas A&M University), this is an algorithm to remove the noise in triangulated models, this method maximize the flat regions and preserve sharp features.
This implementation is written in python as Blender addon to add this operator in the sculpt process or mesh denoising.

# Recursos
- [Slides](https://github.com/AlonzoQuio/MeshDenoisingViaL0Minimization/blob/master/presentation/slides_MeshDenoisingViaL0Minimization.pdf)

# Video de presentación
<iframe width="640" height="480" src="https://www.youtube.com/embed/clS97D_JxYQ" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

# Instalación y ejemplo de uso
A continuación se muestra como se debe instalar el addon y un ejemplo de su utilización.

<iframe width="640" height="480" src="https://www.youtube.com/embed/LBJyFWKMSrs" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

# Reporte
## Detalles de implementación
### Blender panel
En la siguiente imagen se puede apreciar el panel que permitira controlar los parámetros para la ejecución del proceso de eliminación de ruido.
Este panel se encuentra en la sección de Propiedades > Object y cuenta con dos botones:
- **Denoise:** Comienza el proceso de eliminación de ruido utilizando los parámetros seleccionados.
- **Reload parámeters:** Reinicia los parámetros del addon.
<p><img src="https://github.com/AlonzoQuio/MeshDenoisingViaL0Minimization/blob/master/page/images/blender_addon.png?raw=true" alt=""></p>

### Pre-calcular edge handle y edge handle vertex
Luego de culminar la implementación del paper y obteniendo tiempos muy largos, se identificaron dos secciones que pueden ser optimizadas utilizando precalculo ya que en estas matrices se almacenaba la relación entre edges y vertices de dos caras opuestas y la relación entre vértices y la estructura anterior, debido a que estas relaciones no cambian aunque la posición de los vértices sea modificada se realizó un precálculo de estos valores utilizando la gpu para aumentar la velocidad obteniendo una reducción de 39 segundos aprox a 0.35 segundos para el caso puntual del objeto Fandisk el cual tiene 6475 vértices.

### Calcular los valores del área basado en el operador de edges
Se actualizo este método para utilizar las matrices precalculadas.
En este paso se realiza el cálculo de los valores del operador tomando en cuenta las 2 caras que comparten un edge.

### Resolver el valor de delta
Se utilizan los valores encontrados en la sección anterior para obtener el valor mínimo de beta por cada edge.

### Resolver el valor de los vertices
El último paso es resolver el sistema lineal para obtener el nuevo valor de los vértices, en este apartado fue donde se obtuvo una gran mejora en el tiempo ya que aquí se realizaba el cálculo de la matriz edge handle la cual ahora es precalculada.
<p><img src="https://github.com/AlonzoQuio/MeshDenoisingViaL0Minimization/blob/master/page/images/opencl.png?raw=true" alt=""></p>

## Limitaciones
- No fue posible probar con todas las mallas presentadas en el articulo ya que no se tenia acceso a los objetos ni los parametros especificos que fueron utilizados.
- Al no estar disponible el codigo del articulo no se lograron hacer comparaciones respecto al tiempo indicado y el tiempo obtenido en la re-implementación.
- El paper indica que utiliza TAUCS: A library of sparse linear solvers (presentado por Sivan Toledo y Rotkin) para resolver el sistema esparso de ecuaciones, en esta implementación se utilizara la libreria scipy de python.

## Comparaciones con los resultados mostrados en el artículo
En la siguiente figura podemos ver a la izquierda el resultado mostrado en el paper para el objeto Fandisk y a la izquierda el resultado obtenido por la reimplementación en Blender, donde podemos apreciar que se lograron replicar los resultados presentados en el paper.
<center>
<img src="https://github.com/AlonzoQuio/MeshDenoisingViaL0Minimization/blob/master/page/images/result_compare.png?raw=true" alt="">
</center>

El paper indica tiempos de ejecucion de 2 segundos para un objeto con 3800 vertices y 3 minutos para un objeto con 134345 vertices sin embargo estos tiempos no pudieron ser verificados debido a que no se cuenta con el codigo del paper. El paper "Guided mesh normal filtering" proporciona una implementación cuyos tiempos son notoriamente diferentes, a continuación se presenta una tabla comparativa de los tiempos obtenidos en la implementacion de "Guided mesh normal filtering" vs el addon implementado para blender.

| Modelo  | Vertices | T ejecución Reimplementación     |T ejecución Addon |
| ------- | -------- |--------------------------------- | ---------------- |
| Fandisk |    6475  |                       2:05       |             0:41 |
| Iron    |   85574  |                      43:21       |            10:41 |

# Resultados
Para realizar las comparaciones se utilizaron los parámetros mostrados en la siguiente imagen, estos parametros fueron obtenidos del paper "Guided mesh normal filtering."
<p><img src="https://github.com/AlonzoQuio/MeshDenoisingViaL0Minimization/blob/master/page/images/parameters.png?raw=true" alt=""></p>
A continuacion se muestran algunos ejemplos de ejecución del algoritmo.
<iframe src="https://drive.google.com/file/d/16m7ARnW7VhpZ--DkFbnOdeNhWW53tpms/preview" width="320" height="240"></iframe>
<iframe src="https://drive.google.com/file/d/1sIfbXYrvst5y2nK1ZBIrSZPBASXwS9bx/preview" width="320" height="240"></iframe>
<iframe src="https://drive.google.com/file/d/1V1rQfjsr8x2SaRxpaQWTiVjPQ2sDSfsF/preview" width="320" height="240"></iframe>
<iframe src="https://drive.google.com/file/d/1iY7boI6uyBglfkV-VeqaXWdIXwdXzpJn/preview" width="320" height="240"></iframe>

# Referencias
- Lei He and Scott Schaefer. Mesh denoising via L0 minimization - ACM Transactions on Graphics - SIGGRAPH, July 2013.
- Wangyu Zhang, Bailin Deng, Juyong Zhang, Sofien Bouaziz, Ligang Liu. Guided mesh normal filtering, 2015.
- Sivan Toledo, D. C., and Rotkin, V. 2001. Taucs: A library of sparse linear solvers.

# Author

* **Alonzo Quio** - *Master student at UCSP*
