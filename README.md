# Mesh denoising via L0 Minimization
# Abstract
This work is an implementation  of the paper Mesh denoising via L0 minimization proposed by Lei He (Texas A&M University) and 
Scott Schaefer (Texas A&M University), this is an algorithm to remove the noise in triangulated models, this method maximize the flat regions and preserve sharp features.
This implementation is written in python as Blender addon to add this operator in the sculpt process or mesh denoising.

# Recursos
- [Slides](https://github.com/AlonzoQuio/MeshDenoisingViaL0Minimization/edit/master/README.md)

# Video de presentación
<iframe src="https://drive.google.com/file/d/1Nhz3uZ1Nvk8wwky-_0gewWPUt6_NKLyn/preview" width="640" height="480"></iframe>

# Instalacion y ejemplo de uso
<iframe src="https://drive.google.com/file/d/1shnBk35uXhpGyJPSvvZ8GId3KPZQg2TY/preview" width="640" height="480"></iframe>

# Reporte
## Detalles de implementación
### Blender panel
La propuesta fue reimplementar el paper Mesh Denoising Via $L_{0}$ minimization\cite{1} en forma de una addon para blender, en la figura \ref{fig:panel_blender} se puede apreciar el panel que permitira controlar los parametros para la ejecución del proceso de eliminación de ruido. 
<img src="https://github.com/AlonzoQuio/MeshDenoisingViaL0Minimization/blob/master/page/images/blender_addon.png?raw=true" alt="">

MeshDenoisingViaL0Minimization/page/images/blender_addon.png
### Pre-calcular edge handle y edge handle vertex
Luego de culminar la implementación del paper y obteniendo tiempos muy largos, se identificaron dos secciones que pueden ser optimizadas utilizando precalculo ya que en estas matrices se almacenaba la relación entre edges y vertices de dos caras opuestas y la relación entre vértices y la estructura anterior, debido a que estas relaciones no cambian aunque la posición de los vértices sea modificada se realizó un precálculo de estos valores utilizando la gpu para aumentar la velocidad obteniendo una reducción de 39 segundos aprox a 0.35 segundos para el caso puntual del objeto Fandisk el cual tiene 6475 vértices.

### Calcular los valores del área basado en el operador de edges
Se actualizo este método para utilizar las matrices precalculadas.
En este paso se realiza el cálculo de los valores del operador tomando en cuenta las 2 caras que comparten un edge.

### Resolver el valor de delta
Se utilizan los valores encontrados en la sección anterior para obtener el valor mínimo de beta por cada edge.

### Resolver el valor de los vertices
El último paso es resolver el sistema lineal para obtener el nuevo valor de los vértices, en este apartado fue donde se obtuvo una gran mejora en el tiempo ya que aquí se realizaba el cálculo de la matriz edge handle la cual ahora es precalculada.

## Limitaciones
- No fue posible probar con todas las mallas presentadas en el articulo ya que no se tenia acceso a los objetos ni los parametros especificos que fueron utilizados.
- Al no estar disponible el codigo del articulo no se lograron hacer comparaciones respecto al tiempo indicado y el tiempo obtenido en la re-implementación.
- El paper indica que utiliza TAUCS: A library of sparse linear solvers (presentado por Sivan Toledo y Rotkin) para resolver el sistema esparso de ecuaciones.

## Comparaciones con los resultados mostrados en el artículo
Se realizaron algunas pruebas de la ejecución del algoritmo, obteniendo los resultados mostrados en la tabla \ref{tab:test} donde se puede observar el tiempo empleado para realizar la eliminación de ruido de las diferentes mallas.

Finalmente en la figura \ref{fig:result_fandisk} podemos ver a la izquierda el resultado mostrado en el paper para la malla presentada en la Fig.\ref{fig:input_fandisk} y a la izquierda el resultado obtenido por la reimplementación en Blender, donde podemos apreciar que se logró replicar los resultados presentados en el paper.

2 seconds for Figures 1 and 4, which have about 3800 vertices each, to about 3 minutes for the statue in
Figure 11, which has 134345 vertices

## Discusion

y todo lo que veas necesario para explicar y valorar tu trabajo.

# Resultados
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
