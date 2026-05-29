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

## Descripción del Problema

El sector turístico moderno se enfrenta al reto de la sobrecarga de información: los usuarios disponen de miles de opciones de viaje, pero carecen del tiempo para filtrarlas de manera manual. Los sistemas de recomendación tradicionales, basados únicamente en filtrado colaborativo, sufren del problema de "arranque en frío" (incapacidad de recomendar a usuarios nuevos sin historial de viajes) e ignoran por completo los metadatos de contexto del cliente, como el tamaño del grupo familiar o sus preferencias explícitas de registro. Por otro lado, los sistemas basados únicamente en contenido no logran asimilar las interacciones cruzadas complejas entre diferentes perfiles de usuarios.

### Objetivo

Diseñar, entrenar y evaluar un motor de recomendación híbrido basado en **Redes Neuronales de Filtrado Colaborativo (NCF)** que combine el comportamiento histórico de calificaciones de los usuarios con sus metadatos contextuales (género, tamaño del grupo de viaje y preferencias temáticas explícitas). El sistema debe predecir la calificación estimada (de 1 a 5 estrellas) que un usuario le asignaría a un destino aún no visitado, permitiendo generar un **Top-3 de recomendaciones personalizado** para optimizar la toma de decisiones comerciales y mejorar la fidelización del cliente.

El dataset utilizado es el [*Travel Recommendation Dataset* de Kaggle](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/data), procesado mediante un pipeline de unificación matricial en PyTorch.

---

## 1. Análisis Exploratorio de Datos (EDA)

Antes de estructurar la red neuronal, se realizó una fase de exploración para diagnosticar el balance de la oferta de atractivos turísticos, caracterizar el perfil de los usuarios e investigar la consistencia de las métricas de retroalimentación (*feedback*) disponibles en el sistema.

### 1.1 Frecuencia de Aparición de Destinos

El primer paso consistió en analizar el volumen de registros asignados a cada atractivo turístico dentro de la base de datos para verificar si existía una distribución equitativa de la oferta.

<div style="text-align: center;">
  <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo3/01_frecuencia_destinos.png?raw=true" width="80%" alt="Análisis de Frecuencia de Destinos"/>
  <p><em>Figura 1. Frecuencia de aparición de cada destino en el catálogo</em></p>
</div>

El catálogo de la **Figura 1** revela una distribución perfectamente uniforme: cada uno de los **5 destinos turísticos reales** (*Taj Mahal*, *Goa Beaches*, *Jaipur City*, *Kerala Backwaters* y *Leh Ladakh*) cuenta con exactamente **200 registros** en la base de datos de destinos. Esta nivelación absoluta es de gran importancia práctica, ya que garantiza que la oferta inicial del catálogo carece de desbalances inherentes; en consecuencia, cualquier sesgo o preferencia detectada por la red neuronal provendrá estrictamente del comportamiento real de los usuarios y no de una asimetría artificial en los datos de entrada.

### 1.2 Distribución de Variables del Catálogo

Con la oferta de destinos identificada, se evaluaron los atributos internos de popularidad y temporalidad del catálogo para verificar que no existieran sesgos implícitos que pudieran distorsionar las sugerencias del modelo.

<div style="text-align: center;">
  <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo3/02_analisis_destinos.png?raw=true" width="100%" alt="Análisis de Distribuciones del Catálogo"/>
  <p><em>Figura 2. Distribución del índice de popularidad, temporadas óptimas y tipos de destino</em></p>
</div>

Al analizar los tres histogramas de la **Figura 2**, se extraen las siguientes observaciones sobre la estructura de los datos:
*   **Popularidad:** Presenta una distribución bimodal bien definida con picos en los rangos de $8.0\text{--}8.4$ y $9.0\text{--}9.3$. Esto indica la coexistencia de destinos consagrados de alta afluencia (como monumentos históricos nacionales) con destinos alternativos o de nicho, lo cual dota al recomendador de una variabilidad útil para diversificar sus propuestas.
*   **Temporada Óptima (`BestTimeToVisit`):** Registra exactamente 200 observaciones para cada una de las 5 ventanas estacionales de viaje. Esto es fundamental para asegurar que el modelo no recomiende de manera desproporcionada periodos específicos del año, manteniendo una oferta equilibrada y aplicable en cualquier estación.
*   **Tipos de Destino:** Al igual que las temporadas, las 5 tipologías de viaje analizadas (*Historical*, *Beach*, *City*, *Nature* y *Adventure*) están perfectamente balanceadas con 200 registros cada una, confirmando que la especialización temática del recomendador dependerá exclusivamente del perfil del viajero.

### 1.3 Contraste entre Feedback Explícito e Implícito

Este análisis constituye el pilar estratégico del preprocesamiento. Evaluamos por separado las calificaciones públicas directas (*reviews*) de los usuarios frente al historial operativo de los viajes reales completados por los clientes para detectar discrepancias en el comportamiento del feedback.

<div style="text-align: center; display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 300px; text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo3/03_distribucion_reviews.png?raw=true" width="90%" alt="Distribución de Calificaciones en Reviews Públicas"/>
    <p><em>Figura 3. Distribución de calificaciones en reseñas públicas (Promedio ≈ 3.02)</em></p>
  </div>
  <div style="flex: 1; min-width: 300px; text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo3/04_calificaciones_historial.png?raw=true" width="90%" alt="Distribución de Calificaciones en el Historial de Viajes"/>
    <p><em>Figura 4. Distribución de calificaciones en el historial operativo (Promedio ≈ 2.90)</em></p>
  </div>
</div>

El contraste visual entre la **Figura 3** y la **Figura 4** devela una asimetría de comportamiento crítica para el negocio:
*   Las reseñas públicas muestran una distribución prácticamente uniforme en toda la escala de 1 a 5 estrellas (con un promedio de **3.02**). Este comportamiento plano es común en foros públicos, donde los usuarios que se toman el tiempo de escribir suelen polarizarse entre el entusiasmo o la queja, cancelando los sesgos centrales.
*   El historial de viajes operativos muestra una distribución marcadamente decreciente, con una alta acumulación de experiencias calificadas con 1 y 2 estrellas, cayendo el promedio general a **2.90**. Esto revela que la experiencia real vivida por el cliente suele registrar mayor insatisfacción de lo que reflejan las reseñas públicas.

Este hallazgo justifica plenamente la decisión de **fusionar ambas fuentes de datos en una sola matriz maestra única de interacciones**. Si el modelo se entrenara únicamente con reseñas públicas, desarrollaría un sesgo optimista que ignoraría la fricción real del servicio histórico. La unificación matemática promedia el ruido de ambas señales y permite al recomendador aprender de la insatisfacción real del cliente para evitar sugerir destinos propensos a malas experiencias.

### 1.4 Perfil de los Usuarios: Preferencias y Demografía

Por último, se analizó la composición demográfica y las preferencias declaradas por los usuarios durante su registro, variables que actúan como el contexto del modelo híbrido.

<div style="text-align: center;">
  <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo3/05_preferencias_genero.png?raw=true" width="100%" alt="Preferencias Explícitas y Distribución de Género"/>
  <p><em>Figura 5. Preferencias declaradas por los usuarios y distribución de género</em></p>
</div>

El estudio de la **Figura 5** arroja las siguientes condiciones demográficas:
*   **Asimetría de Intereses:** La preferencia por destinos de tipo **`Historical`** es abrumadora, acumulando más de **650 menciones** explícitas (prácticamente el doble que el resto de categorías como *Beaches*, *Nature* o *Adventure*, que promedian $330$ menciones). Esta concentración representa un reto técnico: si recomendáramos basándonos en la popularidad cruda de los datos, el Taj Mahal saturaría las sugerencias de casi todos los usuarios. El enfoque híbrido de Neural CF es indispensable aquí para cruzar esta preferencia de entrada con la satisfacción real del historial de viaje de cada individuo.
*   **Balance de Género:** La muestra se encuentra perfectamente dividida con aproximadamente $495$ mujeres y $505$ hombres. Esto garantiza que la variable de género (codificada como una característica binaria de contexto en la red) no incorpore sesgos de representación o subrepresentación durante el proceso de optimización del gradiente.
*   **Estructura Familiar:** Con un registro promedio de **1.5 adultos y casi 1 niño por usuario**, se evidencia que el núcleo de clientes está compuesto por grupos familiares medianos. Esto fundamenta técnicamente la decisión de consolidar la variable `Total_Travelers` como una característica contextual clave para guiar el recomendador hacia destinos aptos para el tamaño del grupo.

---

## 2. Justificación de la Arquitectura

Para abordar de manera óptima este escenario de datos, se optó por una arquitectura de **Filtrado Colaborativo Neuronal (Neural CF) Híbrido** implementada en PyTorch. Los algoritmos clásicos de factorización de matrices son incapaces de asimilar variables no lineales o incorporar metadatos de contexto demográfico de los usuarios, limitándose únicamente a la matriz de interacciones vacía. Al utilizar una red neuronal híbrida, podemos proyectar las identidades a espacios densos de baja dimensión (*embeddings*) y cruzar esa información con el perfil demográfico real del cliente en una sola etapa de aprendizaje.

La formulación matemática que rige la predicción de la calificación estimada para el par de usuario $u$ y destino $d$ se describe a continuación:

<div align="center" markdown="1">

$$\hat{r}_{u,d} = \sigma\bigl(\text{MLP}([\mathbf{e}_u \;\|\; \mathbf{e}_d \;\|\; \mathbf{c}_u])\bigr) \times 5.0$$

*Ecuación 1. Predicción híbrida de calificación mediante concatenación de embeddings y contexto, escalada al rango [0, 5]*
</div>

Donde:
*   **$\mathbf{e}_u$ y $\mathbf{e}_d$:** Representan los vectores de *embeddings* entrenables de usuarios y destinos, respectivamente, diseñados para capturar de forma latente gustos y afinidades no observables directamente.
*   **$\mathbf{c}_u$:** Es el vector de características de contexto del usuario (género binario, tamaño consolidado del grupo familiar y preferencias codificadas mediante Multi-Hot Encoding).
*   **$\text{MLP}$:** Es el Perceptrón Multicapa encargado de mapear las interacciones no lineales de la concatenación ($\|$) de los vectores.
*   **$\sigma \times 5.0$:** Es la función de activación sigmoide escalada linealmente, una decisión de diseño crítica que restringe físicamente la salida del modelo al rango real de calificación de $[0, 5]$ estrellas, previniendo que el optimizador genere valores fuera del dominio durante las fases de alta inestabilidad de gradiente.

Para dar con la estructura con mejor capacidad de generalización sobre estos datos, se propusieron tres experimentos competitivos:

<div align="center" markdown="1">

*Tabla 3. Arquitecturas propuestas para el motor de recomendación híbrido*

| Parámetro / Componente | Modelo 1 (Base) | Modelo 2 (Profundo) | Modelo 3 (Regularizado) |
| :--- | :---: | :---: | :---: |
| **Dimensión de Embeddings** | 16 | 32 | 16 |
| **Arquitectura MLP** | 64 $\rightarrow$ 32 | 128 $\rightarrow$ 64 $\rightarrow$ 32 | 32 $\rightarrow$ 16 |
| **Tasa de Dropout** | 20% | 20% | **40%** |
| **Penalización L2 (Weight Decay)** | $10^{-5}$ | $10^{-5}$ | **$10^{-4}$** |
| **Tasa de Aprendizaje ($\eta$)** | 0.005 | 0.003 | 0.005 |

</div>

El **Modelo Base** establece nuestra referencia de baja complejidad. El **Modelo Profundo** busca incrementar la capacidad de la red mediante embeddings de mayor dimensión y una capa oculta adicional para extraer abstracciones de mayor nivel. Por último, el **Modelo Regularizado** simplifica la estructura interna de la red e incrementa drásticamente la tasa de Dropout y la penalización de pesos con el objetivo explícito de combatir el sobreentrenamiento (*overfitting*).

---

## 3. Preparación de los Datos

**1. Consolidación Matricial:** Se fusionaron las tablas de reseñas e historial, promediando las puntuaciones de las interacciones duplicadas para obtener registros únicos por pareja de usuario y destino, logrando balancear la señal de opinión pública con la experiencia operativa real descrita en el EDA.

**2. Mapeo de Índices en PyTorch:** Se construyeron diccionarios bidireccionales para mapear los identificadores únicos a índices enteros contiguos (`user_idx`, `dest_idx`) compatibles con la capa de `nn.Embedding` de PyTorch, garantizando la traducción inversa en la etapa de producción.

**3. Codificación de Contexto:** 
*   `Gender_Bin`: Mapeo de género a formato binario (Male = 1, Female = 0).
*   `Total_Travelers`: Suma de adultos y niños acompañantes para representar la carga familiar del viaje.
*   Preferencias: Codificación de variables de texto mediante Multi-Hot Encoding usando `str.get_dummies()` para alimentar de forma binaria el vector de contexto con las tipologías preferidas de cada usuario.

**4. DataLoaders y Partición:** División cronológica 80% entrenamiento / 20% prueba con un tamaño de lote de 64. La función de pérdida utilizada fue **MSELoss** (Error Cuadrático Medio) y el optimizador seleccionado fue **Adam** para asegurar un ajuste dinámico de los momentos del gradiente.

---

## 4. Evaluación de los Modelos y Selección

La evaluación de las tres arquitecturas candidatas se realizó cruzando el rendimiento del error continuo en la escala de estrellas con la capacidad de clasificación del recomendador en el dominio binario (considerando un destino como "Relevante / Acierto" si la calificación real es $\geq 3.0$ estrellas).

<div align="center" markdown="1">

*Tabla 4. Comparativa de desempeño de las tres arquitecturas en el set de prueba*

| Métrica de Evaluación | Modelo 1 (Base) | Modelo 2 (Profundo) | Modelo 3 (Regularizado - Campeón) |
| :--- | :---: | :---: | :---: |
| **MAE (Error Absoluto Medio)** | ~1.53 | ~1.43 | **~1.43** |
| **Precisión Binaria (Aciertos)**| — | — | **58% -- 61%** |
| **Recall Binario (Sensibilidad)**| — | — | **~51%** |
| **F1-Score Binario** | — | — | **~0.518** |

</div>

### Análisis Técnico y Selección del Campeón

*   **Error Continuo:** El Modelo Profundo y el Modelo Regularizado empatan estrechamente en el error absoluto de predicción, alcanzando un MAE de **1.43 estrellas**, superando con claridad al Modelo Base (~1.53).
*   **Métricas de Recomendación (Binarias):** El **Modelo Regularizado se consolida como el modelo campeón**. Logra una precisión binaria superior al **58%**, lo cual significa que de cada 10 destinos que el motor sugiere activamente, aproximadamente **6 corresponden a elecciones realmente satisfactorias** para el perfil del usuario. Asimismo, alcanza el F1-Score más alto (~0.518), logrando el mejor balance entre sugerencias relevantes encontradas y precisión del catálogo recomendado.
*   **Justificación del Fenómeno:** Al contar con un dataset de tamaño acotado (1,000 interacciones únicas), las arquitecturas con alta capacidad representacional (como el Modelo Profundo con embeddings de 32 y tres capas densas) sufren severamente de sobreajuste, memorizando el ruido de la muestra. El Modelo Regularizado, al simplificar el número de neuronas e incorporar una tasa de Dropout agresiva del **40%** junto con una penalización L2 diez veces mayor ($10^{-4}$), actúa como un regularizador eficaz que fuerza a la red a generalizar las fronteras de decisión, resultando en proyecciones mucho más estables y útiles en el conjunto de prueba.

---

## 5. Ejemplos de Recomendaciones Generadas

El motor predictivo opera bajo el siguiente pipeline lógico para recomendar destinos en tiempo real:

1.  **Detección de Candidatos:** Extrae la lista de destinos del catálogo que el usuario específico **aún no ha visitado** en su historial.
2.  **Inferencia de Calificación:** Alimenta la red neuronal híbrida con el ID del usuario, su vector de contexto (familia, género, preferencias) y los atributos de los destinos candidatos para predecir la calificación estimada de 1 a 5 estrellas.
3.  **Filtrado y Ranking:** Ordena de mayor a menor los candidatos según la nota proyectada y genera el **Top-3 de destinos personalizados**. El output final exporta de manera estructurada el nombre del atractivo, la temporada ideal de viaje, su popularidad global y la estrella proyectada que sirve como justificación de la sugerencia.

---

## 6. Conclusiones

La personalización en el sector de viajes responde a dinámicas multifactoriales complejas que involucran intereses subjetivos de recreación, limitaciones de acompañamiento familiar e históricos de consumo. El diseño e implementación de este recomendador Neural CF Híbrido comprobó que es viable cruzar interacciones con variables demográficas para mitigar el problema del arranque en frío y ofrecer recomendaciones dirigidas y altamente personalizadas.

La decisión metodológica de fusionar reseñas públicas y el historial operativo fue el factor determinante del éxito del módulo. Como reveló el EDA, las opiniones públicas presentan un sesgo optimista artificial que contrasta con la fricción real registrada en el historial de viajes del cliente. La matriz unificada equilibra ambos mundos, permitiendo que el recomendador aprenda de la insatisfacción operativa y proteja al usuario de malas experiencias futuras.

La principal limitación identificada reside en el alcance geográfico del dataset, el cual está restringido a **5 destinos específicos del subcontinente indio**. Para un despliegue de escala global, es indispensable enriquecer la base de datos con rutas culturalmente diversas y una mayor variedad de atractivos internacionales, lo cual robustecería los embeddings latentes del modelo y dotaría al sistema de una capacidad de generalización idónea para usuarios de cualquier región del mundo.

---

## 7. Propuesta de Evaluación de Efectividad Comercial

Debido a que el desarrollo se realizó sobre un entorno estático y sin conexión en vivo con clientes, se plantea el siguiente marco metodológico para evaluar la efectividad comercial del recomendador una vez desplegado en producción:

### 1. Medición de Satisfacción del Cliente (Pruebas A/B)
*   **Métricas de Interacción (CTR):** Se propone dividir el tráfico de la web de manera aleatoria. El Grupo A (Control) recibirá recomendaciones basadas únicamente en la popularidad global de los destinos (Taj Mahal para la mayoría de los usuarios). El Grupo B (Tratamiento) recibirá el Top-3 de la red neuronal híbrida. Monitorear la tasa de clics sobre las propuestas (*Click-Through Rate*) permitirá cuantificar de forma objetiva el incremento del interés real provocado por la personalización.
*   **Encuestas de Retroalimentación Post-Viaje:** Tras completar un viaje recomendado por el sistema, el usuario calificará de forma obligatoria su experiencia de 1 a 5 estrellas. El error absoluto entre la calificación predicha por la IA y la nota real recolectada servirá como la métrica de control de deriva (*drift*) del recomendador en producción.

### 2. Medición del Impacto Operativo y Logístico
*   **Tasa de Conversión de Compras:** Evaluar si la tasa de reservas y compra de paquetes turísticos se eleva en el segmento de usuarios expuestos al motor predictivo.
*   **Planificación Anticipada de Capacidad:** Conocer con semanas de anticipación el perfil de preferencias predictivo de los usuarios activos permite a la agencia prever picos de demanda en categorías específicas (por ejemplo, turismo histórico o de playa). Esto habilita la negociación en bloque de reservas de hoteles y transporte terrestre con los proveedores locales, disminuyendo costos fijos y mejorando los márgenes de rentabilidad de la empresa de transporte.

---

## 8. Bibliografía

*   **He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017).** Neural Collaborative Filtering. *Proceedings of the 26th International Conference on World Wide Web (WWW '17)*, 173–182.
*   **Koren, Y., Bell, R., & Volinsky, C. (2009).** Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30–37.
*   **Paszke, A., et al. (2019).** PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems*, 32.
*   **Ricci, F., Rokach, L., & Shapira, B. (2015).** *Recommender Systems Handbook* (2nd ed.). Springer.