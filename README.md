# Mesh denoising via L0 Minimization - Blender addon
# Abstract

# Recursos
- [Slides](https://github.com/AlonzoQuio/MeshDenoisingViaL0Minimization/edit/master/README.md)

# Video de presentación
<iframe src="https://drive.google.com/file/d/1Nhz3uZ1Nvk8wwky-_0gewWPUt6_NKLyn/preview" width="640" height="480"></iframe>

# Instalación y ejemplo de ejecución
<iframe src="https://drive.google.com/file/d/1shnBk35uXhpGyJPSvvZ8GId3KPZQg2TY/preview" width="640" height="480"></iframe>

# Reporte 
## Detalles de implementación
### Panel en blender
La propuesta fue reimplementar el paper Mesh Denoising Via $L_{0}$ minimization\cite{1} en forma de una addon para blender, en la figura \ref{fig:panel_blender} se puede apreciar el panel que permitira controlar los parametros para la ejecución del proceso de eliminación de ruido. 
\begin{figure}[!hbt]
	\begin{center}
	\includegraphics[width=4cm]{panel_blender}
	\caption{Panel del addon en blender}
	\label{fig:panel_blender}
	\end{center}
\end{figure}

### Pre-calcular edge handle y edge handle vertex
Luego de culminar la implementación del paper y obteniendo tiempos muy largos, se identificaron dos secciones que pueden ser optimizadas utilizando precalculo ya que en estas matrices se almacenaba la relación entre edges y vertices de dos caras opuestas y la relación entre vértices y la estructura anterior, debido a que estas relaciones no cambian aunque la posición de los vértices sea modificada se realizó un precálculo de estos valores utilizando la gpu para aumentar la velocidad obteniendo una reducción de 39 segundos aprox a 0.35 segundos para el caso puntual del objeto Fandisk Fig.\ref{fig:input_fandisk} el cual tiene 6475 vértices.

### Calcular los valores del área basado en el operador de edges
Se actualizo este método para utilizar las matrices precalculadas.
En este paso se realiza el cálculo de los valores del operador tomando en cuenta las 2 caras que comparten un edge.

### Resolver el valor de delta
Se utilizan los valores encontrados en la sección anterior para obtener el valor mínimo de beta por cada edge.

### Resolver el valor de los vertices
El último paso es resolver el sistema lineal para obtener el nuevo valor de los vértices, en este apartado fue donde se obtuvo una gran mejora en el tiempo ya que aquí se realizaba el cálculo de la matriz edge handle la cual ahora es precalculada.

## Limitaciones
## Comparaciones con los resultados mostrados en el artículo
Se realizaron algunas pruebas de la ejecución del algoritmo, obteniendo los resultados mostrados en la tabla \ref{tab:test} donde se puede observar el tiempo empleado para realizar la eliminación de ruido de las diferentes mallas.

Finalmente en la figura \ref{fig:result_fandisk} podemos ver a la izquierda el resultado mostrado en el paper para la malla presentada en la Fig.\ref{fig:input_fandisk} y a la izquierda el resultado obtenido por la reimplementación en Blender, donde podemos apreciar que se logró replicar los resultados presentados en el paper.


## Discusion

y todo lo que veas necesario para explicar y valorar tu trabajo.

# Resultados
A continuacion se muestran algunos ejemplos de ejecución del algoritmo.
<iframe src="https://drive.google.com/file/d/16m7ARnW7VhpZ--DkFbnOdeNhWW53tpms/preview" width="320" height="240"></iframe>
<iframe src="https://drive.google.com/file/d/1sIfbXYrvst5y2nK1ZBIrSZPBASXwS9bx/preview" width="320" height="240"></iframe>
<iframe src="https://drive.google.com/file/d/1V1rQfjsr8x2SaRxpaQWTiVjPQ2sDSfsF/preview" width="320" height="240"></iframe>

# Referencias
- Lei He and Scott Schaefer. Mesh denoising via L0 minimization - ACM Transactions on Graphics - SIGGRAPH, July 2013.

- Wangyu Zhang, Bailin Deng, Juyong Zhang, Sofien Bouaziz, Ligang Liu. Guided mesh normal filtering, 2015.


Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

# Header 1
You can use the [editor on GitHub](https://github.com/AlonzoQuio/MeshDenoisingViaL0Minimization/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.
## Header 2
### Header 3

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/AlonzoQuio/MeshDenoisingViaL0Minimization/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
