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

---

# **Sistema Inteligente Integrado para Predicción, Clasificación y Recomendación en la Empresa de Transporte**

---

## 1. Resumen Ejecutivo

El sector de transporte terrestre de pasajeros se enfrenta a tres desafíos operativos críticos: la ineficiencia en la asignación de recursos debido a la variabilidad de la demanda, los riesgos de seguridad derivados de comportamientos inadecuados durante la conducción, y la necesidad de fidelizar clientes mediante estrategias personalizadas de viaje. 

Este proyecto propone una solución tecnológica integral dividida en tres módulos funcionales sustentados en el uso de redes neuronales y algoritmos de aprendizaje de máquinas:
1. **Predicción de Demanda:** Implementación de modelos de series de tiempo para proyectar el flujo de pasajeros en rutas críticas con un horizonte de 30 días.
2. **Detección de Conducción Distractiva:** Clasificación automática de imágenes en cabina mediante redes neuronales convolucionales (CNN) para identificar conductas de riesgo en los conductores.
3. **Recomendación de Destinos:** Un motor de filtrado colaborativo e híbrido que personaliza la oferta de viajes según el historial y preferencias de los usuarios.

Los resultados obtenidos muestran que la combinación de estas tecnologías permite estructurar un esquema metodológico viable para mitigar la incertidumbre en la planeación y promover entornos de transporte más seguros y orientados al cliente.

---

## 2. Introducción

### Contexto del Problema e Importancia
La gestión de flotas y servicios de transporte de pasajeros a gran escala requiere decisiones logísticas rápidas y fundamentadas. Históricamente, la planificación de frecuencias de viaje se ha realizado con base en heurísticas manuales o promedios simples, lo que genera sobrecostos operativos por subutilización de unidades o pérdida de ingresos por sobredemanda. 

Paralelamente, la seguridad vial representa la prioridad ética y operativa de cualquier empresa del sector. Los accidentes causados por distracciones del conductor (uso de dispositivos móviles, fatiga o desatención) generan impactos humanos y económicos considerables. Por último, la competitividad comercial del negocio exige transformar la interacción con el cliente de un modelo transaccional a uno personalizado, sugiriendo destinos adecuados para optimizar la ocupación de las rutas existentes.

### Objetivos del Proyecto
* **General:** Desarrollar e integrar tres soluciones analíticas basadas en redes neuronales y algoritmos de aprendizaje para optimizar la operación, elevar los estándares de seguridad vial y personalizar la oferta comercial en una empresa de transporte.
* **Específicos:**
  * Proyectar la demanda de transporte para los próximos 30 días en rutas clave, minimizando el error de planeación logística.
  * Identificar de forma automática comportamientos de conducción distractora a partir de registros visuales de la cabina de control.
  * Generar recomendaciones personalizadas de destinos turísticos o comerciales basadas en el comportamiento histórico de consumo de los usuarios.
  * Integrar las tres soluciones en un prototipo de interfaz web funcional.

### Alcances y Limitaciones
* **Alcance:** El sistema opera bajo datos históricos estructurados de la empresa de transporte, un corpus de imágenes etiquetadas para el módulo de clasificación y registros de interacción usuario-destino. La interfaz web sirve como demostrador interactivo de las capacidades técnicas del sistema.
* **Limitaciones:** Los modelos están sujetos a la calidad, representatividad y balance de los datos de entrenamiento. El módulo de clasificación de imágenes asume condiciones de iluminación controladas en cabina, y la predicción de demanda se limita a la estabilidad de las dinámicas de estacionalidad históricas presentes en el conjunto de datos.

---

## 3. Metodología General

### Conceptos Clave
* **Series de Tiempo (Análisis de Demanda):** Tratamiento de datos secuenciales con dependencias temporales para la identificación de tendencias, ciclos y estacionalidades.
* **Redes Neuronales Convolucionales (CNN):** Modelos de aprendizaje profundo especializados en el procesamiento de datos con topología de cuadrícula, como imágenes, mediante capas de extracción de características jerárquicas.
* **Sistemas de Recomendación (Filtrado Colaborativo e Híbrido):** Algoritmos diseñados para sugerir ítems (destinos) analizando relaciones entre diferentes usuarios y sus patrones de consumo históricos.

El desarrollo del proyecto se rige por un enfoque metodológico estructurado en fases consecutivas que van desde el entendimiento de los datos hasta la implementación de un prototipo interactivo:

### Fases del Proyecto
1. **Fase I: Exploración y Tratamiento de Datos:** Limpieza de valores nulos, normalización de variables continuas, ingeniería de características temporales y balanceo o aumento de datos visuales.
2. **Fase II: Entrenamiento y Optimización de Modelos:** Experimentación iterativa con hiperparámetros de redes y arquitecturas candidatas para cada uno de los tres módulos.
3. **Fase III: Desarrollo de la Interfaz Web:** Construcción de un entorno accesible donde se integran las tres API de los modelos para permitir pruebas en tiempo real con datos de usuario.
4. **Fase IV: Validación Técnica y Ética:** Evaluación exhaustiva de métricas de rendimiento e impacto sistémico, acompañadas de un análisis de los riesgos éticos de privacidad y sesgos de decisión.

---

## **Navegación del Proyecto**

Utilice los siguientes enlaces para explorar detalladamente el desarrollo técnico, los resultados y las conclusiones de cada componente de la investigación:

📊 [**Módulo 1: Predicción de Demanda (Series de Tiempo)**](modulo1-predicion.md)  
* *Análisis de tendencias temporales, preprocesamiento de secuencias, entrenamiento del modelo y predicciones a 30 días.*

📸 [**Módulo 2: Clasificación de Conducción Distractiva (Imágenes)**](modulo2-clasificacion.md)  
* *Arquitecturas CNN utilizadas, análisis de métricas de clasificación en cabina y evaluación de casos de error.*

🗺️ [**Módulo 3: Sistema de Recomendación de Destinos**](modulo3-recomendacion.md)  
* *Desarrollo de algoritmos de recomendación híbridos y análisis de efectividad de las sugerencias de viaje.*

🌐 [**Herramienta e Interfaz Web**](herramienta-web.md)  
* *Diseño, funcionalidades de carga de archivos e interacción con los tres módulos predictivos en el aplicativo web.*


<br>
<div align="center">
  <a href="https://github.com/jihernandezc/rnaab_sistema_integrado" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Ver_Repositorio_Oficial-181717?style=for-the-badge&logo=github" alt="Ver en GitHub">
  </a>
</div>