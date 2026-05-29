<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<div style="position: sticky; top: 0; background-color: white; padding: 10px 0; border-bottom: 1px solid #ddd; z-index: 999; text-align: center; width: 100%;">
  <a href="index.html" style="text-decoration: none;">🏠 <b>Inicio</b></a> | 
  <a href="modulo1-prediccion.html" style="text-decoration: none;">📈 <b>Módulo 1: Demanda</b></a> |
  <a href="modulo2-clasificacion.html" style="text-decoration: none;">📸 <b>Módulo 2: Conducción</b></a> | 
  <a href="modulo3-recomendacion.html" style="text-decoration: none;">🗺️ <b>Módulo 3: Recomendación</b></a> | 
  <a href="herramienta-web.html" style="text-decoration: none;">🌐 <b>Sitio Web</b></a>
</div>

<br>

# Módulo 3: Sistema Inteligente de Recomendación de Destinos Turísticos (Neural CF Híbrido)

Este módulo presenta el diseño, entrenamiento y evaluación de un motor de recomendación personalizado basado en una **Red Neuronal Híbrida de Filtrado Colaborativo (NCF)**. El objetivo es sugerir destinos de viaje a los clientes combinando su comportamiento histórico con su perfil demográfico y sus preferencias personales.

---

## Descripción del Módulo

El motor de recomendación se construyó en cuatro etapas:

1. **Análisis Exploratorio de Datos (EDA):** Entendimiento profundo del catálogo de destinos, el comportamiento de calificación de los usuarios y su perfil demográfico.
2. **Red Neuronal Híbrida (Neural CF):** Arquitectura basada en capas de *Embeddings* para usuarios y destinos, conectadas a un Perceptrón Multicapa (MLP) que incorpora variables de contexto demográfico.
3. **Evaluación comparativa** de tres arquitecturas usando métricas continuas (MAE, RMSE) y binarias (Precisión, Recall, F1-Score).
4. **Productivización** del modelo seleccionado para integración en la herramienta web del proyecto.

---

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
  <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo3/05_preferencias.png?raw=true" width="100%" alt="Preferencias Explícitas y Distribución de Género"/>
  <p><em>Figura 5. Preferencias explícitas declaradas por los usuarios (izquierda) y distribución de género (derecha)</em></p>
</div>

Dos patrones estructurales definen el perfil de los usuarios:

- **Preferencias declaradas:** `Historical` domina de manera absoluta con más de **650 menciones**, prácticamente el doble que las demás categorías (Beaches, Nature, Adventure y City rondan las **330 menciones** cada una). Esto genera una asimetría de demanda que el motor debe aprender a gestionar: si el modelo ignorara las preferencias individuales y simplemente recomendara por popularidad, casi todos los usuarios recibirían el Taj Mahal como primera opción. El enfoque híbrido evita este sesgo al cruzar la preferencia declarada con el historial real de satisfacción del usuario.
- **Distribución de género:** El dataset está perfectamente balanceado entre usuarios femeninos (~495) y masculinos (~505), lo cual garantiza que la variable de género —incorporada al modelo como feature de contexto binario— no introduzca un sesgo de representación durante el entrenamiento.

Adicionalmente, con un promedio de **1.5 adultos y casi 1 niño por usuario**, se identifica un perfil de viaje predominantemente familiar, lo que refuerza la decisión de incluir el tamaño del grupo (`Total_Travelers`) como variable de contexto del modelo.

---

## 2. Justificación de la Arquitectura

**¿Por qué un enfoque híbrido?** Los datos combinan interacciones usuario–destino (calificaciones de reviews e historial) con metadatos contextuales (género, tamaño del grupo, preferencias). Un algoritmo de filtrado colaborativo tradicional ignoraría el contexto demográfico; el enfoque híbrido cruza el comportamiento histórico con el perfil real del cliente para una personalización más precisa.

**Arquitectura general (Neural CF Híbrida):**

<div align="center" markdown="1">

$$\hat{r}_{u,d} = \sigma\!\left(\text{MLP}\bigl([\mathbf{e}_u \;\|\; \mathbf{e}_d \;\|\; \mathbf{c}_u]\bigr)\right) \times 5.0$$

*Ecuación 1. Predicción de calificación: concatenación de embeddings de usuario* ($\mathbf{e}_u$)*, destino* ($\mathbf{e}_d$) *y contexto* ($\mathbf{c}_u$)*, escalada al rango [0, 5]*
</div>

- **Embeddings:** Traducen los IDs de usuarios y destinos en vectores densos que capturan gustos y afinidades abstractas no observables directamente en los datos crudos.
- **Contexto:** Variables numéricas del perfil del usuario: género binario, total de viajeros y preferencias en Multi-Hot Encoding. Estas features anclan la predicción al perfil demográfico real del cliente.
- **MLP:** Perceptrón Multicapa con activaciones ReLU y Dropout que aprende interacciones no lineales entre los tres componentes anteriores.
- **Salida:** Sigmoide × 5.0 para garantizar predicciones estrictamente dentro del rango [0, 5] estrellas.

**Estrategia de los 3 modelos evaluados:** Para encontrar el balance óptimo entre capacidad y generalización, se evaluaron tres arquitecturas competitivas:

<div align="center" markdown="1">

*Tabla 1. Comparativa de arquitecturas evaluadas*

| Arquitectura | Embedding | Capas MLP | Dropout | Weight Decay | LR |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Modelo Base** | 16 | 64 → 32 | 0.2 | $10^{-5}$ | 0.005 |
| **Modelo Profundo** | 32 | 128 → 64 → 32 | 0.2 | $10^{-5}$ | 0.003 |
| **Modelo Regularizado** | 16 | 32 → 16 | **0.4** | $\mathbf{10^{-4}}$ | 0.005 |

</div>

El **Modelo Base** establece la línea base con la arquitectura más simple. El **Modelo Profundo** amplía la capacidad representacional para buscar patrones más abstractos. El **Modelo Regularizado** reduce el tamaño de las capas e incrementa agresivamente el Dropout y la penalización por peso, obligando a la red a generalizar en lugar de memorizar.

---

## 3. Preparación de los Datos

**Consolidación de interacciones:** Reviews e Historial se fusionaron en una **matriz maestra única**, promediando calificaciones cuando un mismo usuario había interactuado con el mismo destino en ambas fuentes. Esta decisión está respaldada por el hallazgo del EDA: unificar ambas señales neutraliza el sesgo de positividad de las reviews públicas con la fricción real del historial.

**Ingeniería de características contextuales:**
- `Gender_Bin`: género codificado a binario (Male = 1, Female = 0)
- `Total_Travelers`: suma de `NumberOfAdults` + `NumberOfChildren`
- Preferencias de viaje: Multi-Hot Encoding de las categorías declaradas

**División:** 80% entrenamiento / 20% prueba, con DataLoaders de `batch_size = 64`. La función de pérdida es **MSELoss** y el optimizador es **Adam** en los tres experimentos.

---

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
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Modelo Base** | ~1.53 | — | — | — | — |
| **Modelo Profundo** | ~1.43 | — | — | — | — |
| **Modelo Regularizado** | ~1.43 | — | **58–61%** | **~51%** | **~0.518** |

</div>

**Modelo seleccionado: Modelo Regularizado.** Dado que el dataset es reducido (1,000 filas), las arquitecturas más complejas tendieron a memorizar el ruido muestral. El Modelo Regularizado —gracias al Dropout del 40% y la penalización por peso más alta ($10^{-4}$)— controló el sobreajuste y aprendió patrones generales con mayor capacidad de generalización. En términos comerciales: de cada 10 destinos recomendados, aproximadamente **6 son un acierto real** para el usuario.

---

## 5. Ejemplos de Recomendaciones Generadas

El motor opera en tres pasos para cada usuario:

1. Identifica los destinos que el usuario **aún no ha visitado**.
2. Calcula la calificación predicha para cada candidato con el Modelo Regularizado.
3. Ordena de mayor a menor y entrega el **Top-3 personalizado**, incluyendo nombre del atractivo, ubicación, calificación sugerida, mejor época para visitar e índice de popularidad.

---

## 6. Conclusiones

Los gustos de los viajeros responden a múltiples factores simultáneos: intereses personales (historia, playa, aventura, naturaleza, ciudad), tamaño del grupo y perfil demográfico. Este proyecto demostró que un sistema de recomendación útil debe ir más allá de la popularidad agregada y aprender del comportamiento real de cada usuario.

El hallazgo más importante del EDA fue el contraste entre las reviews públicas (señal optimista, promedio 3.02) y el historial de experiencias reales (señal crítica, promedio 2.90). Fusionar ambas fuentes fue la decisión metodológica más relevante del módulo: sin el historial, el modelo aprendería únicamente del ruido de la opinión pública; sin las reviews, perdería señal de usuarios con pocas interacciones operativas.

Como limitación principal, el catálogo se restringe a **5 destinos del subcontinente indio**, lo que acota la diversidad geográfica del sistema. Una base de datos más amplia y culturalmente variada mejoraría sustancialmente la capacidad de generalización del recomendador y su aplicabilidad a perfiles de usuarios de distintas regiones del mundo.

---

## 7. Propuesta de Evaluación de Efectividad Comercial

Como el dataset actual es estático y no existe un entorno en producción, se propone la siguiente estrategia para medir el éxito del recomendador en un escenario real:

**Medición de satisfacción:**
- **Pruebas A/B:** Grupo A recibe recomendaciones por popularidad pura; Grupo B recibe el Top-3 del modelo de IA. El **Click-Through Rate (CTR)** mediaría el incremento real en el interés del cliente.
- **Encuestas post-viaje:** Al regresar, el usuario califica el destino en escala 1–5. Cruzando esta nota real con la calificación sugerida por la IA se calcularía el error de satisfacción en producción.

**Medición de impacto logístico:**
- **Tasa de conversión por ruta:** Monitorear si el volumen de reservas hacia destinos específicos aumenta tras activar el menú personalizado.
- **Análisis de ocupación:** Al conocer con semanas de anticipación las preferencias de categoría de cada usuario, la agencia puede prever la demanda y negociar reservas de hoteles y transporte en bloque, reduciendo costos operativos.

---

## 8. Bibliografía

- **He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017).** Neural Collaborative Filtering. *Proceedings of the 26th International Conference on World Wide Web (WWW '17)*, 173–182.
- **Ricci, F., Rokach, L., & Shapira, B. (2015).** *Recommender Systems Handbook* (2nd ed.). Springer.
- **Koren, Y., Bell, R., & Volinsky, C. (2009).** Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30–37.
- **Paszke, A., et al. (2019).** PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems*, 32.