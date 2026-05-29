<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<div style="position: sticky; top: 0; background-color: white; padding: 10px 0; border-bottom: 1px solid #ddd; z-index: 999; text-align: center; width: 100%;">
  <a href="index.html" style="text-decoration: none;">🏠 <b>Inicio</b></a> | 
  <a href="prediccion.html" style="text-decoration: none;">📈 <b>Módulo 1: Demanda</b></a> |
  <a href="clasificacion.html" style="text-decoration: none;">📸 <b>Módulo 2: Conducción</b></a> | 
  <a href="recomendacion.html" style="text-decoration: none;">🗺️ <b>Módulo 3: Recomendación</b></a> | 
  <a href="herramienta-web.html" style="text-decoration: none;">🌐 <b>Sitio Web</b></a>
</div>

<br>

# Módulo 2: Clasificación de Conducción Distractiva (CNN + Transfer Learning)

## Descripción del Problema

La conducción distractiva es uno de los principales problemas de seguridad vial en la actualidad. Los conductores frecuentemente utilizan su celular para llamadas o mensajes de texto, experimentan somnolencia o realizan otras actividades que reducen su concentración, incrementando el índice de accidentes y siniestros viales alrededor del mundo.

### Objetivo

Diseñar, entrenar y evaluar un modelo de inteligencia artificial basado en redes neuronales convolucionales (CNN) que identifique el comportamiento de los conductores a partir de imágenes, determinando cuándo un conductor maneja de forma segura y cuándo presenta alguna distracción. A partir de su imagen, el conductor se clasifica en una de las siguientes **5 categorías**:

| Clase | Descripción |
| : | : |
| `safe_driving` | Conducción segura |
| `talking_phone` | Hablando por teléfono |
| `texting_phone` | Escribiendo en el teléfono |
| `turning` | Girando el volante |
| `other_activities` | Realizando otras actividades distractoras |

El dataset utilizado es el [*Multi-Class Driver Behavior Image Dataset* de Kaggle](https://www.kaggle.com/datasets/arafatsahinafridi/multi-class-driver-behavior-image-dataset/data), alojado en Google Drive y procesado en el entorno de Google Colab.


## 1. Análisis Exploratorio de los Datos

Antes de diseñar el modelo se realizó un análisis exploratorio para identificar y corregir inconsistencias que pudieran afectar el desempeño del clasificador.

### 1.1 Distribución de Clases

Se analizó la distribución de las muestras para verificar la presencia de desbalances severos que pudieran sesgar el aprendizaje de la red neuronal.

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo2/01_distribucion_clases.png?raw=true" width="750" alt="Distribución de imágenes por clase" />
    <p><em>Figura 1. Distribución de imágenes por clase en el dataset</em></p>
</div>

El dataset cuenta con un total de **7,276 imágenes**. Como se observa en la **Figura 1**, las clases se encuentran distribuidas de manera casi equitativa: `safe_driving` lidera con $1,679$ imágenes, mientras que `other_activities` registra la menor cantidad con $1,184$ imágenes. El ratio de desbalance es prácticamente unitario ($1.4$), lo que nos permite prescindir de técnicas complejas de remuestreo (como SMOTE o pesos de pérdida balanceados) y utilizar la exactitud (*accuracy*) como métrica de evaluación válida.

### 1.2 Muestras del Dataset por Clase

Se visualizaron conjuntos de imágenes originales para entender el ángulo de captura, la variabilidad de sujetos y la iluminación de las escenas.

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo2/02_ejemplos_clases.png?raw=true" width="900" alt="Ejemplos de imágenes por clase" />
    <p><em>Figura 2. Muestras representativas de las cinco categorías de conducción en el dataset</em></p>
</div>

El mosaico de la **Figura 2** revela que las tomas provienen principalmente de cámaras de salpicadero instaladas desde el ángulo del copiloto. Se observa una amplia heterogeneidad en cuanto a género, rango de edad, vestimenta, e iluminación de fondo. Sin embargo, los patrones físicos son consistentes: en `talking_phone` hay un brazo levantado pegado a la oreja; en `texting_phone` la mirada del conductor se desvía visiblemente hacia abajo, y en `turning` el torso y los hombros muestran rotaciones características.

### 1.3 Distribución de Canales RGB por Clase

Se calcularon los promedios de los canales Rojo (R), Verde (G) y Azul (B) sobre los píxeles de todas las imágenes pertenecientes a cada categoría para descartar sesgos cromáticos.

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo2/03_rgb_por_clase.png?raw=true" width="800" alt="Promedio de canales RGB por clase" />
    <p><em>Figura 3. Valores promedio de intensidad de píxeles en canales R, G y B</em></p>
</div>

Como se observa en la **Figura 3**, las distribuciones de color son homogéneas entre clases, con valores promedio que oscilan entre $80$ y $103$ en la escala estándar de $[0-255]$. El canal **Verde (G)** y el **Azul (B)** son ligeramente dominantes en todas las categorías. Este fenómeno es físicamente coherente, ya que gran parte de las ventanas de los automóviles y parabrisas capturan el entorno exterior (follaje, cielo, asfalto), introduciendo un tono frío/verdoso de fondo constante que no interfiere de forma selectiva con ninguna de las clases de interés.

### 1.4 Imagen Promedio por Clase

Se construyó una "imagen fantasma" promedio para cada categoría con el fin de identificar espacialmente las zonas de mayor variabilidad de movimiento.

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo2/04_imagen_promedio_clase.png?raw=true" width="100%" alt="Imagen promedio por clase" />
    <p><em>Figura 4. Visualización del promedio de intensidad de píxeles por categoría</em></p>
</div>

Las representaciones difusas de la **Figura 4** confirman que la estructura estática del habitáculo del vehículo (asiento del conductor, marco de la ventana y tablero) se mantiene nítida. En contraste, la silueta del conductor aparece sumamente borrosa debido a los cambios de postura. Es notable cómo en `talking_phone` y `texting_phone` el área correspondiente a la consola central y el lado derecho del rostro presenta manchas de mayor resplandor, evidenciando la presencia recurrente de las manos y el dispositivo móvil en dichas coordenadas espaciales.



## 2. Preprocesamiento y Preparación de los Datos

### 2.1 Corrección de Orientación EXIF
Durante la exploración, se identificaron **81 imágenes con orientación incorrecta (rotadas 90°)** debido a metadatos de cámaras de dispositivos móviles. Se aplicó una corrección automática mediante la función `ImageOps.exif_transpose()` de Pillow para asegurar la consistencia geométrica y evitar que la red interpretara erróneamente la verticalidad del conductor.

### 2.2 Data Augmentation (Aumento de Datos)
Para inducir robustez y regularizar el modelo, se aplicaron transformaciones geométricas y cromáticas aleatorias sobre el set de entrenamiento:

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo2/05_imagenes_preprocesadas.png?raw=true" width="100%" alt="Muestras preprocesadas con data augmentation" />
    <p><em>Figura 5. Muestras resultantes tras la aplicación de técnicas de aumento de datos</em></p>
</div>

Como se observa en la **Figura 5**, las transformaciones aplicadas (rotación leve de $\pm10^\circ$, desplazamientos horizontales y verticales del $10\%$, variaciones de brillo entre $[0.7, 1.3]$ y recortes de zoom) simulan perturbaciones reales del entorno de cabina, tales como cambios repentinos de luz solar o vibraciones físicas de la cámara, sin alterar la semántica de la acción clasificada.



## 3. Diseño y Entrenamiento de la Red Convolucional

### 3.1 Arquitectura del Modelo Base (Entrenamiento desde Cero)

Se implementó un modelo inicial propio compuesto por tres bloques convolucionales apilados de manera secuencial (filtros de $32$, $64$ y $128$ con kernels de $3\times3$ y activación ReLU), seguidos de Max-Pooling y capas de Batch Normalization.

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo2/06_curvas_entrenamiento.png?raw=true" width="900" alt="Curvas de entrenamiento del modelo base" />
    <p><em>Figura 6. Desempeño y curvas de aprendizaje del modelo CNN entrenado desde cero</em></p>
</div>

El modelo base demostró un rendimiento limitado, alcanzando un **accuracy del 44%** en validación. Como se observa en la **Figura 6**, las curvas revelan un aprendizaje lento y ruidoso, con oscilaciones severas en la pérdida de validación. Esto se debe a que aprender características abstractas de alta complejidad espacial (como la sujeción de objetos pequeños o desvíos sutiles de la mirada) requiere de una arquitectura mucho más profunda, lo cual resulta inviable de entrenar con un set de datos de tamaño moderado.



### 3.2 Transfer Learning con MobileNetV2

Para superar las limitaciones del modelo base, se empleó la técnica de **Transfer Learning** utilizando el backbone de **MobileNetV2** preentrenado en ImageNet. El entrenamiento se estructuró en dos fases metodológicas:

1.  **Fase 1 (Feature Extraction):** Se congelaron las 154 capas del backbone de MobileNetV2 y se entrenó únicamente la cabeza clasificadora (Global Average Pooling seguida de una capa Dense de 256 neuronas, Dropout de 0.4 y una capa de salida Softmax para 5 clases).
2.  **Fase 2 (Fine-Tuning):** Se descongelaron las **últimas 30 capas** de MobileNetV2 (manteniendo congeladas las capas de Batch Normalization para no alterar las estadísticas de media y varianza móvil aprendidas) y se reentrenó el modelo completo con una tasa de aprendizaje extremadamente baja ($\eta = 5 \times 10^{-5}$).

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo2/08_mobilenet.png?raw=true" width="900" alt="Curvas de entrenamiento de MobileNetV2" />
    <p><em>Figura 7. Curvas de accuracy y pérdida durante la extracción de características (Fase 1) y ajuste fino (Fase 2)</em></p>
</div>

Como se visualiza en la **Figura 7**:
*   La **Fase 1** (épocas 1 a 9) estabilizó rápidamente el accuracy de validación en torno al **$62.7\%$**.
*   Al activar la **Fase 2** (marcada por la línea discontinua vertical en la época 9), se observa un salto cualitativo en la precisión, disminuyendo la pérdida y llevando el accuracy de validación hasta un **$68.4\%$**. Esto comprueba la efectividad de especializar las capas superiores de extracción de características en los detalles finos de la cabina del conductor.



## 4. Resultados y Métricas de Evaluación

### 4.1 Matriz de Confusión y Test-Time Augmentation (TTA)

Para robustecer la clasificación, se aplicó *Test-Time Augmentation* (TTA) durante la inferencia, realizando 10 predicciones aumentadas por cada imagen y promediando los resultados. El modelo final consolidó una **exactitud global del 69.1%**.

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo2/07_matriz_confusion.png?raw=true" width="650" alt="Matriz de confusión MobileNetV2" />
    <p><em>Figura 8. Matriz de confusión resultante de la evaluación con TTA</em></p>
</div>

El análisis detallado de la **Figura 8** revela el comportamiento del clasificador:
*   Las clases **`talking_phone`** ($221$ aciertos) y **`texting_phone`** ($257$ aciertos) presentan la menor tasa de confusión cruzada entre sí, demostrando que la red diferencia con éxito el gesto de sostener el teléfono en la oreja frente al de sostenerlo abajo frente al torso.
*   **La mayor fuente de error se concentra en la diagonal de `other_activities` y `safe_driving`.** El modelo confundió $88$ imágenes de `other_activities` clasificándolas como `safe_driving`. Esto ocurre porque la clase de "otras actividades" es una categoría comodín que carece de una firma visual única, asemejándose a menudo a la postura de conducción neutra y segura.
*   La clase **`turning`** se confunde frecuentemente con `other_activities` ($90$ casos) y `safe_driving` ($53$ casos). Esto se debe a que la acción de girar el volante es transitoria y su captura estática a menudo se solapa visualmente con tener las manos colocadas sobre el volante de manera segura.



## 5. Clasificaciones del Modelo

### 5.1 Ejemplos de Clasificaciones Correctas

El modelo muestra un excelente desempeño en imágenes que cumplen con las condiciones de distribución del set de entrenamiento estándar de vehículos particulares.

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo2/09_clasificaciones_correctas.png?raw=true" width="100%" alt="Ejemplos de clasificaciones correctas" />
    <p><em>Figura 9. Inferencia exitosa en diferentes escenarios de iluminación y fisionomías utilizando MobileNetV2</em></p>
</div>

### 5.2 Ejemplos de Clasificaciones Incorrectas (Análisis de Falla)

El análisis de falsos positivos y falsos negativos es fundamental para comprender las limitaciones del modelo bajo condiciones de transferencia de dominio.

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo2/10_clasificaciones_incorrectas.png?raw=true" width="100%" alt="Ejemplos de clasificaciones incorrectas" />
    <p><em>Figura 10. Errores de clasificación e inferencias fallidas en entornos no controlados</em></p>
</div>

#### Descubrimiento Crítico: El Fenómeno del "Domain Shift"
Al inspeccionar las fallas del modelo en la **Figura 10**, se revela un patrón sumamente interesante: **casi la totalidad de las clasificaciones erróneas ocurren en imágenes capturadas dentro de cabinas de autobuses de transporte público o tractocamiones.**

*   **¿Por qué falla el modelo en estos casos?** Las cabinas de los vehículos de gran tamaño (buses) presentan una geometría espacial radicalmente distinta a la de los automóviles particulares: el volante es mucho más grande y plano, el conductor se posiciona más alto, el parabrisas es de proporciones masivas y el ángulo relativo de la cámara varía sustancialmente.
*   Debido a esto, acciones reales como `turning` (girar el volante de un autobús) o `other_activities` son catalogadas sistemáticamente como `safe_driving`. Para el modelo, la disposición espacial de los brazos en un entorno de autobús no coincide con los patrones aprendidos en vehículos particulares, lo que evidencia una vulnerabilidad de generalización ante cambios de dominio (*Out-of-Distribution - OOD*).



## 6. Aspectos Éticos y Creatividad

### Privacidad y Protección de Datos en Cabina
La implementación de sistemas de monitoreo por cámara dentro de cabinas de conducción plantea serios dilemas de privacidad. Estas imágenes contienen datos biométricos implícitos y capturan el comportamiento cotidiano de los operadores de transporte en su espacio de trabajo. 

*   **Solución propuesta (Edge Computing):** Para garantizar un despliegue ético, el modelo desarrollado (`modelo_web.keras`) está diseñado para operar localmente en el dispositivo de procesamiento integrado del vehículo (*on-the-edge*). Las imágenes capturadas por la cámara interna se procesan en la memoria RAM temporal del hardware local y se descartan inmediatamente después de la inferencia, sin almacenar registros de video ni transmitir datos de imagen a servidores en la nube. Únicamente se registran y transmiten señales de alerta de telemetría anonimizadas (por ejemplo: *"Alerta de distracción detectada en Ruta 12 - 14:35h"*).

### Sesgos de Datos y Equidad Algorítmica
Como se demostró en el análisis de fallas (Figura 10), el modelo presenta un **sesgo geométrico institucional**. Si este sistema se utilizara directamente para evaluar y sancionar a conductores de autobuses o camiones de carga de la empresa, se cometerían injusticias laborales severas debido a la alta tasa de falsos positivos en estas cabinas. Un despliegue ético exige declarar los límites del modelo y restringir su uso exclusivamente a vehículos particulares de pasajeros, o bien, enriquecer de forma balanceada el dataset con imágenes de cabinas de vehículos pesados antes de su implementación corporativa.

### Enfoque Creativo del Módulo
La innovación metodológica de este módulo radica en el uso integrado de metadatos EXIF para saneamiento geométrico automático y la aplicación sistemática de Test-Time Augmentation (TTA), logrando exprimir el rendimiento de una arquitectura ligera y de bajo consumo energético (MobileNetV2) hasta rozar el **70% de exactitud**, ideal para su ejecución en tiempo real en dispositivos de hardware limitado integrados al tablero del vehículo.



## 7. Bibliografía

*   **Afridi, A. S. (2024).** *Multi-Class Driver Behavior Image Dataset*. Recuperado de Kaggle: https://www.kaggle.com/datasets/arafatsahinafridi/multi-class-driver-behavior-image-dataset/data
*   **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).** MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4510-4520.
*   **Shorten, C., & Khoshgoftaar, T. M. (2019).** A survey on Image Data Augmentation for Deep Learning. *Journal of Big Data*, 6(1), 1-48.
*   **Wang, G., Li, W., Aertsen, M., Deprest, J., Ourselin, S., & Vercauteren, T. (2019).** Aleatoric uncertainty estimation with test-time data augmentation for medical image segmentation. *International Conference on Information Processing in Medical Imaging*, 551-563.