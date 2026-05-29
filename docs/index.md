<p align="center">
  <img src="https://cdiac.manizales.unal.edu.co/imagenes/LogosMini/un.png" width="250">
  <br>
  <strong>Universidad Nacional de Colombia</strong><br>
  Facultad de Minas - Sede Medellín
</p>

**Curso:** Introducción a Redes Neuronales y Algoritmos Bioinspirados  
**Semestre:** 2026-01  

**Profesor:** Juan David Ospina  
**Monitor:** Andrés Mauricio Zapata  

**Integrantes:**
*   Tomás Acevedo Roldán
*   Santiago Cardona Franco
*   Jimena Hernández Castillo

# **Sistema Inteligente Integrado para Predicción, Clasificación y Recomendación en Transporte**

## 1. Resumen Ejecutivo

El sector de transporte de pasajeros se enfrenta a tres desafíos operativos críticos: la ineficiencia en la asignación de recursos debido a la variabilidad de la demanda, los riesgos de seguridad derivados de comportamientos inadecuados durante la conducción, y la necesidad de fidelizar clientes mediante estrategias personalizadas de viaje.

Este proyecto propone una solución tecnológica integral dividida en tres módulos funcionales sustentados en redes neuronales y algoritmos de aprendizaje automático:

1. **Predicción de Demanda:** Implementación de modelos de aprendizaje supervisado sobre series de tiempo para proyectar el volumen diario de viajes en las rutas más frecuentes con un horizonte de 30 días, utilizando XGBoost como modelo campeón tras comparar con Ridge Regression y Random Forest.
2. **Detección de Conducción Distractiva:** Clasificación automática de imágenes de conductores al volante mediante redes neuronales convolucionales (CNN) y Transfer Learning con MobileNetV2, identificando cinco comportamientos: conducción segura, uso del teléfono para llamadas, uso del teléfono para mensajes, giro del volante y otras actividades distractoras.
3. **Recomendación de Destinos:** Motor de recomendación híbrido basado en una Red Neuronal de Filtrado Colaborativo (Neural CF) que personaliza la oferta de viajes cruzando el historial real de experiencias y las reseñas públicas de los usuarios con su perfil demográfico y preferencias declaradas.

Los resultados obtenidos muestran que la combinación de estas tecnologías permite estructurar un esquema metodológico viable para mitigar la incertidumbre en la planeación, promover entornos de transporte más seguros y orientar la oferta comercial hacia el cliente.

## 2. Introducción

### Contexto del Problema e Importancia

La gestión de flotas y servicios de transporte de pasajeros a gran escala requiere decisiones logísticas rápidas y fundamentadas. Históricamente, la planificación de frecuencias de viaje se ha realizado con base en heurísticas manuales o promedios simples, lo que genera sobrecostos operativos por subutilización de unidades o pérdida de ingresos por sobredemanda.

Paralelamente, la seguridad vial representa la prioridad ética y operativa de cualquier empresa del sector. Los accidentes causados por distracciones del conductor —uso de dispositivos móviles, somnolencia o desatención— generan impactos humanos y económicos considerables. Por último, la competitividad comercial del negocio exige transformar la interacción con el cliente de un modelo transaccional a uno personalizado, sugiriendo destinos adecuados según el comportamiento real de cada usuario para optimizar la ocupación de las rutas disponibles.

### Objetivos del Proyecto

**General:** Desarrollar e integrar tres soluciones analíticas basadas en aprendizaje automático y redes neuronales para optimizar la operación, elevar los estándares de seguridad vial y personalizar la oferta comercial.

**Específicos:**
- Proyectar el volumen diario de viajes para los próximos 30 días en las rutas de mayor demanda, minimizando el error de planeación logística mediante modelos de gradient boosting con features temporales y rezagos.
- Identificar de forma automática los cinco comportamientos de conducción distractora a partir de imágenes de conductores al volante, aprovechando Transfer Learning sobre un backbone preentrenado en ImageNet.
- Generar recomendaciones personalizadas de destinos turísticos basadas en la fusión del historial operativo de viajes y las reseñas públicas del usuario, complementadas con su perfil demográfico y preferencias declaradas.
- Integrar las tres soluciones en un prototipo de interfaz web funcional.

### Alcances y Limitaciones

**Alcance:** El sistema opera sobre tres conjuntos de datos públicos: el *NYC TLC Trip Record Data* (2023–2025) para el módulo de predicción, el *Multi-Class Driver Behavior Image Dataset* de Kaggle para el módulo de clasificación, y el *Travel Recommendation Dataset* para el motor de recomendación. La interfaz web sirve como demostrador interactivo de las capacidades técnicas del sistema integrado.

**Limitaciones:** Los modelos están sujetos a la calidad, representatividad y balance de los datos de entrenamiento. El módulo de predicción se limita a las 20 rutas origen–destino de mayor volumen histórico en Nueva York y su desempeño puede degradarse ante eventos disruptivos no capturados en el historial (clima extremo, cambios de infraestructura vial, eventos masivos). El módulo de clasificación fue entrenado con imágenes de un dataset público con variabilidad de condiciones, por lo que su rendimiento puede variar en entornos con calidad de imagen muy distinta a la del conjunto de entrenamiento. El motor de recomendación opera sobre un catálogo reducido de 5 destinos del subcontinente indio, lo que limita su capacidad de generalización a perfiles de usuarios de otras regiones.

## 3. Metodología General

### Conceptos Clave

- **Aprendizaje Supervisado sobre Series de Tiempo:** Transformación de datos secuenciales en un problema de regresión mediante la construcción de features de rezago temporal (*lags*) y estadísticos de ventana deslizante (*rolling statistics*), permitiendo que modelos de gradient boosting capturen dependencias temporales y patrones estacionales sin asumir linealidad.
- **Redes Neuronales Convolucionales (CNN) y Transfer Learning:** Modelos de aprendizaje profundo especializados en procesamiento de imágenes mediante capas de extracción de características jerárquicas. El Transfer Learning reutiliza el conocimiento visual de un modelo preentrenado en un problema de gran escala (ImageNet) como punto de partida para resolver un problema más específico con datos limitados, reduciendo dramáticamente el tiempo de entrenamiento y mejorando la generalización.
- **Sistemas de Recomendación Híbridos (Neural CF):** Arquitecturas que combinan Embeddings de usuarios e ítems —representaciones vectoriales densas que capturan afinidades abstractas— con características contextuales del perfil del usuario, procesadas por un Perceptrón Multicapa para aprender interacciones no lineales entre el comportamiento histórico y los atributos demográficos.

El desarrollo del proyecto se rige por un enfoque metodológico estructurado en fases consecutivas que van desde el entendimiento de los datos hasta la implementación de un prototipo interactivo:

### Fases del Proyecto

1. **Fase I — Exploración y Tratamiento de Datos:** Limpieza de valores nulos, validación de integridad referencial entre tablas, ingeniería de características temporales (lags, rolling stats, variables de calendario), corrección de orientación EXIF en imágenes, data augmentation para el módulo de clasificación y fusión de fuentes de interacción para el módulo de recomendación.

2. **Fase II — Entrenamiento y Optimización de Modelos:** Experimentación comparativa con múltiples arquitecturas por módulo: Ridge, Random Forest y XGBoost para predicción de demanda; CNN base y MobileNetV2 con entrenamiento bifásico (Feature Extraction + Fine-tuning) para clasificación de conducción; y tres variantes de Neural CF (base, profundo y regularizado) para el motor de recomendación.

3. **Fase III — Desarrollo de la Interfaz Web:** Construcción de un entorno accesible donde se integran los tres modelos exportados para permitir pruebas en tiempo real: pronóstico de demanda a 30 días, clasificación de imágenes de conducción y generación de recomendaciones de destinos personalizadas.

4. **Fase IV — Validación Técnica y Ética:** Evaluación de métricas de rendimiento (R², MAE, RMSE para predicción; Accuracy, F1-Score y matriz de confusión para clasificación; MAE, Precisión y Recall binario para recomendación) e identificación de riesgos éticos como sesgos geográficos en la selección de rutas prioritarias, privacidad de los datos de conductores y equidad en la cobertura de rutas de menor demanda.

## **Navegación del Proyecto**

Utilice los siguientes enlaces para explorar detalladamente el desarrollo técnico, los resultados y las conclusiones de cada componente de la investigación:

📊 [**Módulo 1: Predicción de Demanda**](prediccion.md)  
* *Análisis de tendencias temporales, preprocesamiento de secuencias, entrenamiento del modelo y predicciones a 30 días.*

📸 [**Módulo 2: Clasificación de Conducción Distractiva**](clasificacion.md)  
* *Arquitecturas CNN utilizadas, análisis de métricas de clasificación en cabina y evaluación de casos de error.*

🗺️ [**Módulo 3: Recomendación de Destinos**](recomendacion.md)  
* *Desarrollo de algoritmos de recomendación híbridos y análisis de efectividad de las sugerencias de viaje.*

🌐 [**Herramienta e Interfaz Web**](herramienta-web.md)  
* *Diseño, funcionalidades de carga de archivos e interacción con los tres módulos predictivos en el aplicativo web.*

<br>
<div align="center">
  <a href="https://github.com/jihernandezc/rnaab_sistema_integrado" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Ver_Repositorio_Oficial-181717?style=for-the-badge&logo=github" alt="Ver en GitHub">
  </a>
</div>