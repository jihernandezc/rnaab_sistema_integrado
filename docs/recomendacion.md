<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<div style="position: sticky; top: 0; background-color: white; padding: 10px 0; border-bottom: 1px solid #ddd; z-index: 999; text-align: center; width: 100%;">
  <a href="index.html" style="text-decoration: none;">🏠 <b>Inicio</b></a> | 
  <a href="prediccion.html" style="text-decoration: none;">📈 <b>Predicción</b></a> |
  <a href="clasificacion.html" style="text-decoration: none;">📸 <b>Clasificación</b></a> | 
  <a href="recomendacion.html" style="text-decoration: none;">🗺️ <b>Recomendación</b></a> | 
  <a href="herramienta-web.html" style="text-decoration: none;">🌐 <b>Sitio Web</b></a>
</div>

<br>

# Módulo 3: Recomendación de Destinos de Viaje

## Descripción del Problema

El sector turístico moderno se enfrenta al reto de la sobrecarga de información: los usuarios disponen de miles de opciones de viaje, pero carecen del tiempo para filtrarlas de manera manual. Los sistemas de recomendación tradicionales, basados únicamente en filtrado colaborativo, son incapaces de recomendar a usuarios nuevos sin historial de viajes e ignoran por completo los metadatos de contexto del cliente, como el tamaño del grupo familiar o sus preferencias explícitas de registro. Por otro lado, los sistemas basados únicamente en contenido no logran asimilar las interacciones cruzadas complejas entre diferentes perfiles de usuarios.

### Objetivo

Diseñar, entrenar y evaluar un motor de recomendación híbrido basado en **Redes Neuronales de Filtrado Colaborativo (NCF)** que combine el comportamiento histórico de calificaciones de los usuarios del [*Travel Recommendation Dataset* de Kaggle](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/data) con sus metadatos contextuales (género, tamaño del grupo de viaje y preferencias temáticas explícitas). El sistema debe predecir la calificación estimada (de 1 a 5 estrellas) que un usuario le asignaría a un destino aún no visitado, permitiendo generar un **Top-3 de recomendaciones personalizado** para optimizar la toma de decisiones comerciales y mejorar la fidelización del cliente.

## 1. Análisis Exploratorio de Datos

### 1.1 Catálogo de Destinos

El primer paso fue entender la estructura del catálogo disponible para el recomendador. El dataset contiene exactamente **5 destinos turísticos reales**, cada uno representado con 200 registros en la tabla de destinos.

<div style="text-align: center;">
  <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo3/01_frecuencia_destinos.png?raw=true" width="75%" alt="Análisis de Frecuencia de Destinos"/>
  <p><em>Figura 1. Frecuencia de aparición de cada destino en el catálogo — distribución perfectamente uniforme entre los 5 atractivos</em></p>
</div>

Los destinos disponibles son: **Taj Mahal** (histórico), **Goa Beaches** (playa), **Jaipur City** (ciudad), **Kerala Backwaters** (naturaleza) y **Leh Ladakh** (aventura). La uniformidad de 200 registros por destino confirma que la oferta del catálogo está completamente nivelada, lo cual significa que cualquier sesgo en las recomendaciones finales provendrá exclusivamente de las preferencias reales de los usuarios y no de un desequilibrio en los datos de entrada.

### 1.2 Distribución Interna del Catálogo

Con el catálogo identificado, se analizaron las distribuciones de sus variables internas para entender si alguna característica del destino podría introducir sesgos en el modelo.

<div style="text-align: center;">
  <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo3/02_analisis_destinos.png?raw=true" width="100%" alt="Análisis de Distribuciones del Catálogo"/>
  <p><em>Figura 2. Distribución del índice de popularidad, temporadas óptimas de visita y tipos de destino en el catálogo</em></p>
</div>

Tres hallazgos clave emergen de la **Figura 2**:

- **Popularidad:** Los destinos presentan una distribución bimodal con concentraciones alrededor de scores 8.0–8.4 y 9.0–9.3. Esto indica que hay destinos de alta popularidad consolidada (como el Taj Mahal) y destinos de nicho con puntuaciones más moderadas, lo que enriquece la capacidad diferenciadora del modelo.
- **Mejor época para visitar:** La distribución es uniforme entre las 5 ventanas de meses disponibles (200 registros cada una). Esto garantiza que el modelo no favorezca ninguna época estacional de forma artificial, manteniendo recomendaciones válidas durante todo el año.
- **Tipos de destino:** La distribución por categoría (Historical, Beach, City, Nature, Adventure) también es perfectamente uniforme. El sesgo que el motor deberá aprender a capturar viene entonces de las **preferencias declaradas de los usuarios**, no del catálogo.

### 1.3 Contraste entre Feedback Explícito e Implícito

Este es el hallazgo más valioso del EDA y la principal justificación para unificar dos fuentes de datos. Se analizaron por separado las calificaciones públicas (reviews) y el historial operativo de viajes realizados.

<div style="text-align: center;">
  <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo3/03_distribucion_reviews.png?raw=true" width="65%" alt="Distribución de Calificaciones en Reviews Públicas"/>
  <p><em>Figura 3. Distribución de calificaciones en reseñas públicas — distribución relativamente uniforme, promedio ≈ 3.02 / 5.0</em></p>
</div>

Las reviews públicas muestran una distribución casi uniforme entre 1 y 5 estrellas, con un promedio de **3.02 / 5.0**. Este comportamiento es típico de sistemas de opinión pública donde los usuarios tienen motivaciones variadas para calificar (desde entusiasmo hasta queja puntual), lo que produce una distribución plana sin preferencia clara.

<div style="text-align: center;">
  <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo3/04_historial_usuario.png?raw=true" width="65%" alt="Distribución de Calificaciones en el Historial de Viajes"/>
  <p><em>Figura 4. Distribución de calificaciones en el historial de viajes realizados — masa concentrada en 1–2 estrellas, promedio ≈ 2.90 / 5.0</em></p>
</div>

El historial operativo cuenta una historia diferente: la distribución es **decreciente**, con la mayor concentración en 1 y 2 estrellas y una caída progresiva hacia los valores altos. El promedio cae a **2.90 / 5.0**, revelando que la experiencia real de viaje genera mayor insatisfacción de lo que la opinión pública refleja.

Este contraste demuestra por qué el modelo no puede basarse únicamente en las reviews. **Un recomendador entrenado solo con datos públicos aprendería a sugerir destinos "populares" pero no necesariamente satisfactorios para el perfil específico de cada usuario.** Fusionar ambas fuentes en una matriz maestra de interacciones permite que la IA aprenda de la fricción real del cliente y ofrezca recomendaciones genuinamente personalizadas.

### 1.4 Perfil de los Usuarios: Preferencias y Demografía

El último bloque del EDA estudia quiénes son los usuarios del sistema y qué buscan, información que el modelo híbrido incorpora como variables de contexto.

<div style="text-align: center;">
  <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo3/05_analisis_usuarios.png?raw=true" width="100%" alt="Preferencias Explícitas y Distribución de Género"/>
  <p><em>Figura 5. Preferencias explícitas declaradas por los usuarios (izquierda) y distribución de género (derecha)</em></p>
</div>

Dos patrones estructurales definen el perfil de los usuarios:

- **Preferencias declaradas:** `Historical` domina de manera absoluta con más de **650 menciones**, prácticamente el doble que las demás categorías (Beaches, Nature, Adventure y City rondan las **330 menciones** cada una). Esto genera una asimetría de demanda que el motor debe aprender a gestionar: si el modelo ignorara las preferencias individuales y simplemente recomendara por popularidad, casi todos los usuarios recibirían el Taj Mahal como primera opción. El enfoque híbrido evita este sesgo al cruzar la preferencia declarada con el historial real de satisfacción del usuario.
- **Distribución de género:** El dataset está perfectamente balanceado entre usuarios femeninos (~495) y masculinos (~505), lo cual garantiza que la variable de género —incorporada al modelo como feature de contexto binario— no introduzca un sesgo de representación durante el entrenamiento.

Adicionalmente, con un promedio de **1.5 adultos y casi 1 niño por usuario**, se identifica un perfil de viaje predominantemente familiar, lo que refuerza la decisión de incluir el tamaño del grupo (`Total_Travelers`) como variable de contexto del modelo.



## 2. Justificación de la Arquitectura

Para abordar de manera óptima este escenario de datos, se optó por una arquitectura de **Filtrado Colaborativo Neuronal (Neural CF) Híbrido** implementada en PyTorch. Los algoritmos clásicos de factorización de matrices son incapaces de asimilar variables no lineales o incorporar metadatos de contexto demográfico de los usuarios, limitándose únicamente a la matriz de interacciones vacía. Al utilizar una red neuronal híbrida, podemos proyectar las identidades a espacios densos de baja dimensión (*embeddings*) y cruzar esa información con el perfil demográfico real del cliente en una sola etapa de aprendizaje.

La formulación matemática que rige la predicción de la calificación estimada para el par de usuario $$u$$ y destino $$d$$ se describe a continuación:

<div align="center" markdown="1">

$$\hat{r}_{u,d} = \sigma\!\left(\text{MLP}\bigl([\mathbf{e}_u \;\|\; \mathbf{e}_d \;\|\; \mathbf{c}_u]\bigr)\right) \times 5.0$$

*Ecuación 1. Predicción de calificación: concatenación de embeddings de usuario* ($$\mathbf{e}_u$$)*, destino* ($$\mathbf{e}_d$$) *y contexto* ($$\mathbf{c}_u$$)*, escalada al rango [0, 5]*
</div>

- **Embeddings:** Traducen los IDs de usuarios y destinos en vectores densos que capturan gustos y afinidades abstractas no observables directamente en los datos crudos.
- **Contexto:** Variables numéricas del perfil del usuario: género binario, total de viajeros y preferencias en Multi-Hot Encoding. Estas features anclan la predicción al perfil demográfico real del cliente.
- **MLP:** Perceptrón Multicapa con activaciones ReLU y Dropout que aprende interacciones no lineales entre los tres componentes anteriores.
- **Salida:** Sigmoide × 5.0 para garantizar predicciones estrictamente dentro del rango [0, 5] estrellas.

**Estrategia de los 3 modelos evaluados:** Para encontrar el balance óptimo entre capacidad y generalización, se evaluaron tres arquitecturas competitivas:

<div align="center" markdown="1">

*Tabla 1. Comparativa de arquitecturas evaluadas*

| Arquitectura | Embedding | Capas MLP | Dropout | Weight Decay | LR |
| **Modelo Base** | 16 | 64 → 32 | 0.2 | $$10^{-5}$$ | 0.005 |
| **Modelo Profundo** | 32 | 128 → 64 → 32 | 0.2 | $$10^{-5}$$ | 0.003 |
| **Modelo Regularizado** | 16 | 32 → 16 | **0.4** | $$\mathbf{10^{-4}}$$ | 0.005 |

</div>

El **Modelo Base** establece la línea base con la arquitectura más simple. El **Modelo Profundo** amplía la capacidad representacional para buscar patrones más abstractos. El **Modelo Regularizado** reduce el tamaño de las capas e incrementa agresivamente el Dropout y la penalización por peso, obligando a la red a generalizar en lugar de memorizar.



## 3. Preparación de los Datos

**Consolidación de interacciones:** Reviews e Historial se fusionaron en una **matriz maestra única**, promediando calificaciones cuando un mismo usuario había interactuado con el mismo destino en ambas fuentes. Esta decisión está respaldada por el hallazgo del EDA: unificar ambas señales neutraliza el sesgo de positividad de las reviews públicas con la fricción real del historial.

**Ingeniería de características contextuales:**
- `Gender_Bin`: género codificado a binario (Male = 1, Female = 0)
- `Total_Travelers`: suma de `NumberOfAdults` + `NumberOfChildren`
- Preferencias de viaje: Multi-Hot Encoding de las categorías declaradas

**División:** 80% entrenamiento / 20% prueba, con DataLoaders de `batch_size = 64`. La función de pérdida es **MSELoss** y el optimizador es **Adam** en los tres experimentos.


## 4. Evaluación de los Modelos

Se evaluó cada arquitectura con métricas en dos dimensiones complementarias:

**Métricas continuas** — miden la precisión de la predicción en escala de estrellas:

<div align="center" markdown="1">

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|r_i - \hat{r}_i| \qquad RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(r_i - \hat{r}_i)^2}$$

*Ecuaciones 2 y 3. Error absoluto medio y raíz del error cuadrático medio*
</div>

**Métricas binarias** — miden la capacidad de discriminar destinos relevantes (umbral ≥ 3.0 estrellas):

<div align="center" markdown="1">

$$\text{Precisión} = \frac{TP}{TP+FP} \qquad \text{Recall} = \frac{TP}{TP+FN} \qquad F1 = \frac{2 \cdot \text{Precisión} \cdot \text{Recall}}{\text{Precisión}+\text{Recall}}$$

*Ecuaciones 4, 5 y 6. Métricas binarias para clasificación de recomendaciones relevantes*
</div>

<div align="center" markdown="1">

*Tabla 2. Comparativa de métricas entre arquitecturas (set de prueba)*

| Arquitectura | MAE | RMSE | Precisión | Recall | F1-Score |
| **Modelo Base** | ~1.53 | ~1.85| ~0.56 | ~0.42 | ~0.48 |
| **Modelo Profundo** | ~1.4 | ~1.74 | ~0.56 | ~0.47 | ~0.51 |
| **Modelo Regularizado** | ~1.46 | ~1.78 | ~0.58 | ~0.47 | ~0.52 |

</div>

*   **Error Continuo:** El Modelo Profundo y el Modelo Regularizado empatan estrechamente en el error absoluto de predicción, alcanzando un MAE de **1.43 estrellas**, superando con claridad al Modelo Base (~1.53).
*   **Métricas de Recomendación (Binarias):** El **Modelo Regularizado se consolida como el modelo campeón**. Logra una precisión binaria superior al **58%**, lo cual significa que de cada 10 destinos que el motor sugiere activamente, aproximadamente **6 corresponden a elecciones realmente satisfactorias** para el perfil del usuario. Asimismo, alcanza el F1-Score más alto (~0.518), logrando el mejor balance entre sugerencias relevantes encontradas y precisión del catálogo recomendado.
*   **Justificación del Fenómeno:** Al contar con un dataset de tamaño acotado (1,000 interacciones únicas), las arquitecturas con alta capacidad representacional (como el Modelo Profundo con embeddings de 32 y tres capas densas) sufren severamente de sobreajuste, memorizando el ruido de la muestra. El Modelo Regularizado, al simplificar el número de neuronas e incorporar una tasa de Dropout agresiva del **40%** junto con una penalización L2 diez veces mayor ($$10^{-4}$$), actúa como un regularizador eficaz que fuerza a la red a generalizar las fronteras de decisión, resultando en proyecciones mucho más estables y útiles en el conjunto de prueba.


## 5. Ejemplos de Recomendaciones Generadas

El motor opera en tres pasos para cada usuario:
1. Identifica los destinos que el usuario **aún no ha visitado**.
2. Calcula la calificación predicha para cada candidato con el Modelo Regularizado.
3. Ordena de mayor a menor y entrega el **Top-3 personalizado**.

Para ilustrar el comportamiento práctico del filtrado híbrido contextual, se presentan los reportes de recomendación generados para dos perfiles con intereses y estructuras familiares distintas.

### 👤 Reporte para el Cliente: 976
*   **Preferencias de registro:** *Beaches, Historical*
*   **Acompañantes en el viaje:** 2 personas (Perfil familiar o grupal intermedio)

<div style="display: flex; gap: 15px; flex-wrap: wrap; margin: 20px 0; font-family: -apple-system, sans-serif;">

  <!-- Card 1 -->
  <div style="flex: 1; min-width: 280px; background: white; border-radius: 12px; border: 1px solid #e1e8ed; box-shadow: 0 4px 10px rgba(0,0,0,0.05); padding: 20px; position: relative;">
    <div style="position: absolute; top: 15px; right: 15px; background: #e8f5e9; color: #2e7d32; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: bold;">Top 1 - Nature</div>
    <h4 style="margin: 0 0 5px 0; color: #1f425c; font-size: 18px;">Kerala Backwaters</h4>
    <p style="margin: 0 0 15px 0; color: #7f8c8d; font-size: 13px;">📍 Ubicación: Kerala</p>
    <div style="background: #f8f9fa; border-radius: 8px; padding: 12px; font-size: 13px; margin-bottom: 15px;">
      <div style="margin-bottom: 5px;">📅 <b>Época ideal:</b> Sep-Mar</div>
      <div>📈 <b>Popularidad:</b> 7.98 / 10.0</div>
    </div>
    <div style="display: flex; align-items: center; justify-content: space-between;">
      <span style="color: #f39c12; font-size: 16px;">★★★★☆ <span style="color: #7f8c8d; font-size: 12px;">(4/5)</span></span>
      <span style="font-size: 11px; color: #95a5a6; font-style: italic;">Score IA: 4.02</span>
    </div>
  </div>

  <!-- Card 2 -->
  <div style="flex: 1; min-width: 280px; background: white; border-radius: 12px; border: 1px solid #e1e8ed; box-shadow: 0 4px 10px rgba(0,0,0,0.05); padding: 20px; position: relative;">
    <div style="position: absolute; top: 15px; right: 15px; background: #e3f2fd; color: #1565c0; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: bold;">Top 2 - Historical</div>
    <h4 style="margin: 0 0 5px 0; color: #1f425c; font-size: 18px;">Taj Mahal</h4>
    <p style="margin: 0 0 15px 0; color: #7f8c8d; font-size: 13px;">📍 Ubicación: Uttar Pradesh</p>
    <div style="background: #f8f9fa; border-radius: 8px; padding: 12px; font-size: 13px; margin-bottom: 15px;">
      <div style="margin-bottom: 5px;">📅 <b>Época ideal:</b> Nov-Feb</div>
      <div>📈 <b>Popularidad:</b> 8.69 / 10.0</div>
    </div>
    <div style="display: flex; align-items: center; justify-content: space-between;">
      <span style="color: #f39c12; font-size: 16px;">★★★★☆ <span style="color: #7f8c8d; font-size: 12px;">(4/5)</span></span>
      <span style="font-size: 11px; color: #95a5a6; font-style: italic;">Score IA: 3.81</span>
    </div>
  </div>

  <!-- Card 3 -->
  <div style="flex: 1; min-width: 280px; background: white; border-radius: 12px; border: 1px solid #e1e8ed; box-shadow: 0 4px 10px rgba(0,0,0,0.05); padding: 20px; position: relative;">
    <div style="position: absolute; top: 15px; right: 15px; background: #f3e5f5; color: #6a1b9a; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: bold;">Top 3 - City</div>
    <h4 style="margin: 0 0 5px 0; color: #1f425c; font-size: 18px;">Jaipur City</h4>
    <p style="margin: 0 0 15px 0; color: #7f8c8d; font-size: 13px;">📍 Ubicación: Rajasthan</p>
    <div style="background: #f8f9fa; border-radius: 8px; padding: 12px; font-size: 13px; margin-bottom: 15px;">
      <div style="margin-bottom: 5px;">📅 <b>Época ideal:</b> Oct-Mar</div>
      <div>📈 <b>Popularidad:</b> 9.23 / 10.0</div>
    </div>
    <div style="display: flex; align-items: center; justify-content: space-between;">
      <span style="color: #f39c12; font-size: 16px;">★★★★☆ <span style="color: #7f8c8d; font-size: 12px;">(4/5)</span></span>
      <span style="font-size: 11px; color: #95a5a6; font-style: italic;">Score IA: 3.75</span>
    </div>
  </div>

</div>

### 👤 Reporte para el Cliente: 977
*   **Preferencias de registro:** *Nature, Adventure*
*   **Acompañantes en el viaje:** 2 personas (Perfil familiar o grupal intermedio)

<div style="display: flex; gap: 15px; flex-wrap: wrap; margin: 20px 0; font-family: -apple-system, sans-serif;">

  <!-- Card 1 -->
  <div style="flex: 1; min-width: 280px; background: white; border-radius: 12px; border: 1px solid #e1e8ed; box-shadow: 0 4px 10px rgba(0,0,0,0.05); padding: 20px; position: relative;">
    <div style="position: absolute; top: 15px; right: 15px; background: #fff3e0; color: #e65100; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: bold;">Top 1 - Adventure</div>
    <h4 style="margin: 0 0 5px 0; color: #1f425c; font-size: 18px;">Leh Ladakh</h4>
    <p style="margin: 0 0 15px 0; color: #7f8c8d; font-size: 13px;">📍 Ubicación: Jammu and Kashmir</p>
    <div style="background: #f8f9fa; border-radius: 8px; padding: 12px; font-size: 13px; margin-bottom: 15px;">
      <div style="margin-bottom: 5px;">📅 <b>Época ideal:</b> Apr-Jun</div>
      <div>📈 <b>Popularidad:</b> 8.40 / 10.0</div>
    </div>
    <div style="display: flex; align-items: center; justify-content: space-between;">
      <span style="color: #f39c12; font-size: 16px;">★★★★☆ <span style="color: #7f8c8d; font-size: 12px;">(4/5)</span></span>
      <span style="font-size: 11px; color: #95a5a6; font-style: italic;">Score IA: 4.38</span>
    </div>
  </div>

  <!-- Card 2 -->
  <div style="flex: 1; min-width: 280px; background: white; border-radius: 12px; border: 1px solid #e1e8ed; box-shadow: 0 4px 10px rgba(0,0,0,0.05); padding: 20px; position: relative;">
    <div style="position: absolute; top: 15px; right: 15px; background: #e8f5e9; color: #2e7d32; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: bold;">Top 2 - Nature</div>
    <h4 style="margin: 0 0 5px 0; color: #1f425c; font-size: 18px;">Kerala Backwaters</h4>
    <p style="margin: 0 0 15px 0; color: #7f8c8d; font-size: 13px;">📍 Ubicación: Kerala</p>
    <div style="background: #f8f9fa; border-radius: 8px; padding: 12px; font-size: 13px; margin-bottom: 15px;">
      <div style="margin-bottom: 5px;">📅 <b>Época ideal:</b> Sep-Mar</div>
      <div>📈 <b>Popularidad:</b> 7.98 / 10.0</div>
    </div>
    <div style="display: flex; align-items: center; justify-content: space-between;">
      <span style="color: #f39c12; font-size: 16px;">★★★★☆ <span style="color: #7f8c8d; font-size: 12px;">(4/5)</span></span>
      <span style="font-size: 11px; color: #95a5a6; font-style: italic;">Score IA: 3.76</span>
    </div>
  </div>

  <!-- Card 3 -->
  <div style="flex: 1; min-width: 280px; background: white; border-radius: 12px; border: 1px solid #e1e8ed; box-shadow: 0 4px 10px rgba(0,0,0,0.05); padding: 20px; position: relative;">
    <div style="position: absolute; top: 15px; right: 15px; background: #efebe9; color: #4e342e; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: bold;">Top 3 - Beach</div>
    <h4 style="margin: 0 0 5px 0; color: #1f425c; font-size: 18px;">Goa Beaches</h4>
    <p style="margin: 0 0 15px 0; color: #7f8c8d; font-size: 13px;">📍 Ubicación: Goa</p>
    <div style="background: #f8f9fa; border-radius: 8px; padding: 12px; font-size: 13px; margin-bottom: 15px;">
      <div style="margin-bottom: 5px;">📅 <b>Época ideal:</b> Nov-Mar</div>
      <div>📈 <b>Popularidad:</b> 8.61 / 10.0</div>
    </div>
    <div style="display: flex; align-items: center; justify-content: space-between;">
      <span style="color: #f39c12; font-size: 16px;">★★★☆☆ <span style="color: #7f8c8d; font-size: 12px;">(3/5)</span></span>
      <span style="font-size: 11px; color: #95a5a6; font-style: italic;">Score IA: 3.00</span>
    </div>
  </div>

</div>

### Análisis del Comportamiento Híbrido
La comparación de ambos perfiles proporciona una demostración del funcionamiento del modelo:
*   Para el **Cliente 976**, cuyas preferencias explícitas son *Beaches* e *Historical*, el sistema situó en segundo lugar al *Taj Mahal* (categoría *Historical*, con un score estimado de **3.81**). No obstante, priorizó a *Kerala Backwaters* (categoría *Nature*, con **4.02**) debido a las interacciones implícitas de usuarios similares y la alta coincidencia en el tamaño del grupo familiar, lo que demuestra que la red no se limita a un simple mapeo de palabras clave, sino que cruza el contexto real con la densidad de la matriz de filtrado.
*   Para el **Cliente 977**, interesado en *Nature* y *Adventure*, el sistema recomendó en primer lugar y con la mayor certidumbre de clasificación a *Leh Ladakh* (categoría *Adventure*, con un score estimado de **4.38**), seguido de *Kerala Backwaters* (categoría *Nature*, con **3.76**). De esta forma, el modelo es capaz de reconfigurar de manera dinámica las sugerencias según la firma contextual de entrada del usuario.

## 6. Aspectos Éticos y Responsabilidad Algorítmica

El desarrollo y despliegue de sistemas de recomendación en el ámbito turístico implica responsabilidades éticas fundamentales que van más allá del rendimiento puramente matemático del modelo:

### 6.1 Cámaras de Eco Algorítmicas (*Filter Bubbles*)
Un riesgo común en el filtrado colaborativo es la sobre-especialización: si un usuario muestra un interés inicial en destinos históricos, el algoritmo puede tender a recomendar únicamente templos o monumentos, limitando su horizonte cultural. Nuestro enfoque híbrido mitiga parcialmente este efecto al combinar embeddings latentes con variables de contexto, asegurando una "serendipia controlada" que permite sugerir destinos de naturaleza o playas (como se observó en el caso del usuario 976) si perfiles sociodemográficos similares tuvieron experiencias satisfactorias allí.

### 6.2 Sobreturismo y Sostenibilidad Territorial
Recomendar de manera indiscriminada los destinos de mayor popularidad global (como el *Taj Mahal*) puede agudizar problemas de sobreturismo (*overtourism*), impactando negativamente la infraestructura local, el medio ambiente y la calidad de vida de las comunidades residentes. El uso de un recomendador inteligente permite a la agencia promover destinos secundarios con menor densidad histórica pero con altas proyecciones de satisfacción (como *Kerala Backwaters*), distribuyendo el flujo de turistas de forma equilibrada y fomentando un modelo de negocio de turismo sostenible.

### 6.3 Privacidad de los Datos del Perfil
El modelo requiere datos personales como género, preferencias de ocio e información de acompañantes (incluyendo menores de edad en el conteo de niños). Para asegurar el cumplimiento de estándares éticos de privacidad, el sistema debe implementar técnicas de anonimización en la base de datos de producción, absteniéndose de asociar de forma directa nombres o identificaciones reales con las entradas del modelo, y limitando el procesamiento de características contextuales estrictamente a los índices mapeados durante la inferencia local.

## 7. Conclusiones

Los gustos de los de viajeros responden a múltiples factores simultáneos: intereses personales (historia, playa, aventura, naturaleza, ciudad), tamaño del grupo y perfil demográfico. Este proyecto demostró que un sistema de recomendación útil debe ir más allá de la popularidad agregada y aprender del comportamiento real de cada usuario.

El hallazgo más importante del EDA fue el contraste entre las reviews públicas (señal optimista, promedio 3.02) y el historial de experiencias reales (señal crítica, promedio 2.90). Fusionar ambas fuentes fue la decisión metodológica más relevante del módulo: sin el historial, el modelo aprendería únicamente del ruido de la opinión pública; sin las reviews, perdería señal de usuarios con pocas interacciones operativas.

Como limitación principal, el catálogo se restringe a **5 destinos del subcontinente indio**, lo que acota la diversidad geográfica del sistema. Una base de datos más amplia y culturalmente variada mejoraría sustancialmente la capacidad de generalización del recomendador y su aplicabilidad a perfiles de usuarios de distintas regiones del mundo.


## 8. Propuesta de Evaluación de Efectividad Comercial

Como el dataset actual es estático y no existe un entorno en producción, se propone la siguiente estrategia para medir el éxito del recomendador en un escenario real:

**Medición de satisfacción:**
- **Pruebas A/B:** Grupo A recibe recomendaciones por popularidad pura; Grupo B recibe el Top-3 del modelo de IA. El **Click-Through Rate (CTR)** mediaría el incremento real en el interés del cliente.
- **Encuestas post-viaje:** Al regresar, el usuario califica el destino en escala 1–5. Cruzando esta nota real con la calificación sugerida por la IA se calcularía el error de satisfacción en producción.

**Medición de impacto logístico:**
- **Tasa de conversión por ruta:** Monitorear si el volumen de reservas hacia destinos específicos aumenta tras activar el menú personalizado.
- **Análisis de ocupación:** Al conocer con semanas de anticipación las preferencias de categoría de cada usuario, la agencia puede prever la demanda y negociar reservas de hoteles y transporte en bloque, reduciendo costos operativos.


## 8. Bibliografía

- **He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017).** Neural Collaborative Filtering. *Proceedings of the 26th International Conference on World Wide Web (WWW '17)*, 173–182.
- **Ricci, F., Rokach, L., & Shapira, B. (2015).** *Recommender Systems Handbook* (2nd ed.). Springer.
- **Koren, Y., Bell, R., & Volinsky, C. (2009).** Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30–37.
- **Paszke, A., et al. (2019).** PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems*, 32.