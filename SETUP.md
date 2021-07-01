Tutorial usado: https://www.youtube.com/watch?v=XpBGwZNyUh0

 La idea principal es poner las librerias de GLFW y GLAD en cada proyecto de VS 2019, entonces se pueden tener distintas 
versiones diferentes de GLAD (OpenGL) en esta carpeta.


-------- PASOS PARA DESCARGAR E INSTALAR UNA VERSION DE GLFW y GLAD ---------------------

- Descargar GLFW Source package : https://www.glfw.org/download.html
- Descargar version de OpenGL/GLAD (version 3.3 Core para learnopengl): https://glad.dav1d.de/
- Crear un proyecto en VS 2019 vacio
- Crear directorio "Libraries" en la carpeta del proyecto y dentro los directorios "include" y "lib"
- Descomprimir el zip descargado en glfw
- Abrir Cmake y poner la siguiente configuracion:
	- Where is the source code: directorio descomprimido de glfw
	- Where to build the binares: En el mismo directorio anterior, crear la carpeta build e indicarla
	- Ir a Configure y debe haber la siguiente configuracion:
		- Specify the generator for this project: Visual Studio 16 2019
		- Optional platform for geenerator(...): Blank
		- Optional toolsetto use (argument to -T): Blank
		- [x] Use default native compilers
		- [ ] Specify native compilers
		- [ ] Specify toolchain file for cross compiling
		- [ ] Specify options for cross compiling
	- Debe aparecer en el cuadro de al medio en rojo lo sgt:
		- [ ] BUILD_SHARED_LIBS                 
		- CMAKE_AR                          C:/Program Files/Microsoft Visual Studio/2019/......
		- CMAKE_CONFIGURATION_TYPES         Debug;Release;MinSize ...
		- CMAKE_INSTALL_PREFIX              C:/Program Files (x86)/GLFW
		- [x] GLFW_BUILD_DOCS                   
		- [x] GLFW_BUILD_EXAMPLES               
		- [x] GLFW_BUILD_TESTS                  
		- [x] GLFW_INSTALL                      
		- [ ] GLFW_USE_HYBRID_HPG               
		- [ ] GLFW_VULKAN_STATIC                
		- [x] GLFW_MSVC_RUNTIME_LIBRARY_DLL     
	- Apretar Configure, se vuelve blanco y apretar Generate
- Abrir carpeta donde se hizo el build y abrir GLFW.sln
- Seleccionar la solucion -> Build Solution
- Ir a ../build/src/Debug y copiar el archivo "glfw3.lib" a la carpeta del proyecto /project/Libraries/lib
- Ir a ../include y copiar la carpeta "GLFW" a la carpeta del proyecto /project/Libraries/include
- Abrir glad.zip -> ir a /include y copiar carpetas "glad" y "KHR" a la carpeta del proyecto /project/Libraries/include
- Del mismo zip -> ir a /src y copiar el archivo "glad.c" en la carpeta del proyecto /project

-- Ahora hay que configurar el proyecto en VS
- Seleccionar la plataforma x64 en el editor
- Ir a configuraciones del proyecto y seleccionar en Platform: All Platforms
- Ir a VC++ Directories -> Include Directories -> Edit -> new -> ... -> seleccionar la carpeta project/Libraries/include -> ok
- Ir a VC++ Directories -> Library Directories -> Edit -> new -> ... -> seleccionar la carpeta project/Libraries/lib -> ok
- Ir a Linker -> Input -> Additional Dependencies -> Edit -> poner en el campo de texto:
```
glfw3.lib
opengl32.lib
```
  -> ok
- Arrastar el archivo glad.c de la carpeta del proyecto a "Source FIles" en VS 2019
- Crear un main.cpp

----PASOS para instalar GLM--------
Libreria necesaria para calculos vectoriales
- Descargar GLM -> Descomprimir
- Copiar directorio que sea raiz de glm.h y pegarla en projects/Libraries/include

----PASOS para instalar stb_image-------
Libreria necesaria para setear texturas
- Descargar header stb_image.h
- Crear archivo stb.cpp que contenga lo siguiente:
```
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
```
- Copiar stb_image.h en project/Libraries/include
- Copiar stb.cpp en algun directorio, ya uqe es necesario su compilación.

** Se debe agregar stb.cpp en cada proyecto que requiera texturas

----PASOS para instalar imgui 1.8-------
- Descargar los archivos y descomprimir
- incorporar los archivos del directorio base y de la version a usar (open gl 3.* y glfw, headers y cpp que se encuentran en la carpeta backends) al proyecto como una libreria o directamente
- Si se trabaja con una carpeta de libreria, se tiene que setear como tal y en la configuracion del proyecto ejecutable agregar imgui como Add-> Reference
- Mirar el ejemplo para saber agregarlo

----PASOS para instalar assimp -------
- Usar el tutorial para comipilar y la sgt guia para compilar la libreria: https://static.packt-cdn.com/downloads/ModelLoading.pdf
- Agregar config.h desde build/ en include/
- Agregar archivo assimp-vc140-mt.dll en la raiz del proyecto



-------- PASOS PARA crear un Proyecto ---------------------

- Crear un proyecto en VS 2019 vacio
- Crear directorio "Libraries" en la carpeta del proyecto y dentro los directorios "include" y "lib"
- Ir a ../build/src/Debug y copiar el archivo "glfw3.lib" a la carpeta del proyecto /project/Libraries/lib
- Ir a ../include y copiar la carpeta "GLFW" a la carpeta del proyecto /project/Libraries/include
- Abrir glad.zip -> ir a /include y copiar carpetas "glad" y "KHR" a la carpeta del proyecto /project/Libraries/include
- Del mismo zip -> ir a /src y copiar el archivo "glad.c" en la carpeta del proyecto /project
-- Ahora hay que configurar el proyecto en VS
- Seleccionar la plataforma x64 en el editor
- Ir a configuraciones del proyecto y seleccionar en Platform: All Platforms
- Ir a VC++ Directories -> Include Directories -> Edit -> new -> ... -> seleccionar la carpeta project/Libraries/include -> ok
- Ir a VC++ Directories -> Library Directories -> Edit -> new -> ... -> seleccionar la carpeta project/Libraries/lib -> ok
- Ir a Linker -> Input -> Additional Dependencies -> Edit -> poner en el campo de texto:
```
glfw3.lib
opengl32.lib
```
  -> ok
- Arrastar el archivo glad.c de la carpeta del proyecto a "Source FIles" en VS 2019
- Crear un main.cpp