# Detección de Objetos: HSV vs YOLOv8

Prototipo de tesis desarrollado en Python y PyQt6 para comparar la detección de objetos mediante visión computacional tradicional (filtrado de color HSV) y aprendizaje profundo (YOLOv8).

## Características Principales
* **Detección por Color (HSV):** Selección interactiva de rangos de color en tiempo real mediante clics en la interfaz.
* **Detección con Inteligencia Artificial (YOLOv8):** Identificación de objetos preentrenados con porcentajes de confianza.
* **Múltiples Fuentes de Entrada:** Soporte integrado para cámara web, imágenes estáticas y archivos de video pregrabados. ( **Nota:** Para la sección de videos, se requiere usar estrictamente archivos en formato **.mp4**).
* **Captura de Evidencias:** Toma rápida de fotogramas procesados para documentar resultados.

## Requisitos Previos e Instalaciones Necesarias
Asegúrate de tener instalado **Python 3.8 o superior**. 

Antes de ejecutar el código por primera vez, necesitas instalar las librerías de las que depende el sistema. Abre tu terminal o consola de comandos y ejecuta la siguiente línea:

```bash
pip install ultralytics opencv-python numpy PyQt6
