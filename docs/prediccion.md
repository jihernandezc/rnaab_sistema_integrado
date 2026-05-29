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

# Módulo 1: Predicción de Demanda de Transporte (Series de Tiempo)

## Descripción del Problema

La demanda de transporte es un factor crítico para la planificación y operación eficiente de los servicios de transporte público. Anticipar esta demanda permite a las empresas y autoridades optimizar la asignación de recursos, mejorar la experiencia del usuario y reducir costos operativos.

En este proyecto se desarrolla un modelo utilizando el conjunto de datos históricos [*NYC TLC Trip Record Data*](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) para predecir la demanda de transporte (cantidad diaria de viajes) en las 20 rutas origen–destino más frecuentes de la ciudad durante los próximos 30 días.


## 1. Entendimiento del Dataset

### Estructura del Dataset

El conjunto de datos utilizado comprende el registro histórico diario de viajes en taxi amarillo de Nueva York durante el período 2023–2025 (3 años). Las variables principales se detallan en la Tabla 1:

<div align="center" markdown="1">

*Tabla 1. Diccionario de variables del módulo de predicción de demanda*

| Nombre de Variable | Tipo de Dato | Descripción |
| `pickup_date` | Temporal | Fecha del registro diario (AAAA-MM-DD). |
| `PULocationID` | Numérico (Entero) | Identificador de la zona de origen del viaje. |
| `DOLocationID` | Numérico (Entero) | Identificador de la zona de destino del viaje. |
| `trips` | Numérico (Entero) | Variable objetivo: cantidad total de viajes realizados en esa ruta ese día. |
| `trip_distance` | Numérico (Flotante) | Distancia recorrida en millas (usada en filtros de calidad). |
| `fare_amount` | Numérico (Flotante) | Tarifa del viaje en USD (usada en filtros de calidad). |

</div>

El dataset consolidado abarca **9,407,598 registros** diarios de demanda agregada por ruta. Para el modelado se seleccionaron las **20 rutas origen–destino con mayor volumen histórico**, excluyendo zonas desconocidas (IDs 264 y 265).

## 2. Preprocesamiento y Feature Engineering

Para preparar la serie de tiempo antes de alimentar al modelo, se implementó un pipeline de transformación que garantiza series continuas, variables temporales enriquecidas y estadísticos rezagados sin fuga de información entre rutas.

**1. Reindexación Continua:**  
Cada ruta se reindexó sobre el rango de fechas completo (2023-01-01 a 2025-12-31) con el fin de asegurar una temporalidad homogénea. Esto es indispensable porque los modelos basados en rezagos asumen un paso del tiempo constante ($$\Delta t = 1$$ día); de no hacerlo, la red asumiría una continuidad falsa entre días no consecutivos. Los días sin registros se imputaron con cero viajes para representar de manera realista la ausencia de actividad operativa, evitando sesgar al alza los promedios históricos en periodos de inactividad.

**2. Variables de Calendario:**  
Se extrajeron variables cíclicas a partir de la fecha: día de la semana (`dow`), mes, semana del año, día del año, indicador de fin de semana (`is_weekend`) e indicador de festivo nacional de Nueva York (`is_holiday`), usando la librería `holidays`. La justificación detrás de este enriquecimiento radica en que la movilidad urbana responde directamente a rutinas humanas e institucionales; al proveer estas variables exógenas como señales directas de contexto, el modelo puede identificar caídas predecibles en días festivos y fines de semana o incrementos en meses de alta estacionalidad sin tener que deducirlos únicamente a partir de la secuencia numérica.

**3. Lags y Rolling Statistics:**  
Para habilitar el aprendizaje supervisado en algoritmos que no poseen una estructura de memoria interna secuencial inherente, se calcularon rezagos temporales y estadísticos de ventana deslizante. Estas transformaciones convierten la serie de tiempo en un formato tabular compatible con el modelo, proporcionándole memoria de corto plazo (el comportamiento de ayer) y de largo plazo (tendencias semanales y mensuales). El cálculo se encapsuló estrictamente por grupo de ruta para evitar que las dinámicas de movilidad de una zona geográfica se mezclaran con las de otra, previniendo distorsiones en el entrenamiento:

<div align="center" markdown="1">

$$x_{lag_k}(t) = y_{t-k}, \quad k \in \{1, 7, 14, 30\}$$

*Ecuación 1. Construcción de variables de rezago temporal*
</div>

<div align="center" markdown="1">

$$\bar{y}_{t,w} = \frac{1}{w}\sum_{i=1}^{w} y_{t-i}, \quad w \in \{7, 14, 30\}$$

*Ecuación 2. Media móvil centrada en el pasado inmediato*
</div>

**4. Normalización Z-Score por Ruta:**  
Para eliminar la escala diferencial entre rutas de alto volumen (que pueden registrar miles de viajes diarios) y de bajo volumen, se aplicó una normalización estándar calculada exclusivamente sobre el conjunto de entrenamiento:

<div align="center" markdown="1">

$$y_t^* = \frac{y_t - \mu_{train}}{\sigma_{train}}$$

*Ecuación 3. Normalización z-score libre de fuga de datos*
</div>

Este escalado es crítico para evitar que las rutas con mayor densidad dominen el cálculo del gradiente de la función de pérdida durante la optimización. Además, el uso estricto de la media ($$\mu$$) y desviación estándar ($$\sigma$$) del set de entrenamiento garantiza que no exista fuga de información de los datos futuros del set de prueba hacia la fase de aprendizaje.

**5. Codificación One-Hot de Rutas:**  
Las combinaciones origen–destino se codificaron mediante One-Hot Encoding (`route_id_str`). Este paso es fundamental para poder entrenar un **único modelo global consolidado** en lugar de 20 modelos individuales independientes. La codificación binaria le permite al algoritmo diferenciar geográficamente qué ruta específica está evaluando y ajustar el sesgo de la predicción según la zona, mientras se beneficia del aprendizaje compartido de los patrones temporales comunes de toda la ciudad de Nueva York.

## 3. Análisis de Estacionalidad y Tendencias

Para comprender los factores dinámicos que afectan la demanda diaria de taxis, se realizó un estudio estacional y de estacionariedad para cada una de las 20 rutas principales seleccionadas.

### 3.1 Patrón Estacional Interanual

Al analizar la evolución agregada de la demanda mediante una media móvil de 7 días entre los años 2023 y 2025, se observó un patrón cíclico anual muy definido, junto con un crecimiento interanual moderado de la cantidad total de viajes.

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo1/01_estacionalidad_comparativa.png?raw=true" width="100%" alt="Patrón Estacional Comparativo" />
    <p><em>Figura 1. Patrón estacional interanual comparativo (2023-2025) para la ruta Upper East Side South → Upper East Side North</em></p>
</div>

Como se visualiza en la **Figura 1**, la demanda presenta un comportamiento repetitivo caracterizado por:
*   Una caída pronunciada a inicios de año (invierno septentrional, días 1-50).
*   Una notable disminución estival durante los meses de julio y agosto (días 180-240), explicada por el periodo de vacaciones y la consecuente reducción en traslados de negocios u oficinas.
*   Una tendencia sostenida de cierre de año (días 250-330) con un pico masivo hacia mediados de diciembre (días 340-350), impulsado por la temporada de fiestas, eventos culturales y turismo de fin de año en la ciudad de Nueva York.

### 3.2 Análisis Granular de Estacionalidad (Semana y Mes)

El comportamiento cíclico se desglosó por día de la semana y mes del año, permitiendo identificar patrones operativos clave a nivel de micro y macroescala.

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo1/02_patrones_demanda_estilo.png?raw=true" width="100%" alt="Patrones de Demanda - Análisis Estacional" />
    <p><em>Figura 2. Distribución de viajes promedio por día de la semana, mes y mapa de calor de densidad temporal</em></p>
</div>

El análisis descriptivo de la **Figura 2** revela la estructura subyacente de la demanda de movilidad:
1.  **Estacionalidad Semanal (Día de la semana):** Los viajes diarios promedian su máximo volumen a mitad de semana (Miércoles con $$392$$ viajes y Jueves con $$389$$), descendiendo significativamente durante los fines de semana hasta alcanzar su punto mínimo los domingos ($$210$$ viajes de promedio). Esto sugiere un perfil de movilidad fuertemente corporativo y asociado a jornadas laborales de oficina.
2.  **Estacionalidad Mensual (Mes):** Mayo, Octubre, Noviembre y Diciembre representan los periodos más productivos del año (superando los $$365$$ viajes promedio diarios), mientras que Julio y Agosto reflejan valles de baja afluencia por debajo de los $$250$$ viajes.
3.  **Mapa de Calor Cruzado (Mes × Día):** El heatmap confirma la estabilidad temporal del comportamiento semanal. Sin importar el mes del año, los días laborables (especialmente martes a jueves) se mantienen en coloraciones oscuras de alta demanda, mientras que los fines de semana en verano (Julio y Agosto) constituyen los periodos de menor actividad operativa global de la red.

### 3.3 Prueba de Estacionariedad (ADF)

Para evaluar rigurosamente la estabilidad temporal de los datos antes de proceder con el modelamiento predictivo, se aplicó la prueba de Dickey-Fuller Aumentada (ADF). Este paso es de vital importancia en el análisis de series de tiempo, ya que la mayoría de los algoritmos de aprendizaje supervisado asumen que las propiedades estadísticas subyacentes de la serie (como su media, varianza y estructura de autocorrelación) se mantienen constantes a lo largo del tiempo. Si las series presentaran tendencias o variaciones descontroladas (no estacionariedad), el modelo aprendería patrones temporales efímeros que no se generalizarían correctamente al futuro, destruyendo la capacidad predictiva del sistema en producción.

El test ADF evalúa formalmente la presencia de una raíz unitaria mediante la siguiente formulación matemática:

<div align="center" markdown="1">

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$

*Ecuación 4. Formulación matemática de la prueba de Dickey-Fuller Aumentada*
</div>

El hallazgo más relevante de esta validación estadística fue que **la totalidad de las 20 rutas origen-destino seleccionadas resultaron ser estrictamente estacionarias** en su estado original. Los p-valores de la prueba se situaron consistentemente muy por debajo del umbral crítico de significancia de $$0.05$$, lo que permitió rechazar categóricamente la hipótesis nula de raíz unitaria ($$H_0$$) para cada una de las series individuales de la red. 

Este resultado es sumamente valioso para la viabilidad técnica del proyecto: al confirmar que todas las rutas son estacionarias de manera intrínseca (debido a que los datos representan flujos consolidados de movilidad urbana sin tendencias explosivas a largo plazo), se elimina la necesidad de aplicar transformaciones de diferenciación regular ($\Delta y_t = y_t - y_{t-1}$). Esto simplifica drásticamente el flujo de trabajo, previene la pérdida de varianza útil de la serie y facilita una des-normalización directa de las predicciones recursivas en el aplicativo web sin necesidad de integrar complejos procesos de acumulación matemática inversa.

## 4. Desarrollo de los Modelos

Para capturar las dependencias temporales no lineales en las series de demanda, se entrenaron tres modelos bajo un esquema global (un único modelo para todas las rutas simultáneamente). 

La decisión de adoptar un **esquema global** en lugar de entrenar 20 modelos individuales responde a la necesidad de optimizar el mantenimiento del sistema en producción. Al agrupar las rutas en un solo estimador, este puede asimilar dinámicas de comportamiento compartidas en toda la red vial de Nueva York, reduciendo drásticamente la carga de cómputo y previniendo el sobreajuste que sufrirían modelos individuales expuestos a series de datos más cortas.

**Configuración del conjunto de features:**  
El vector de entrada a cada modelo incluye: `dow`, `month`, `week`, `day_of_year`, `is_holiday`, `is_weekend`, `lag_1`, `lag_7`, `lag_14`, `lag_30`, `rolling_7_mean`, `rolling_7_std`, y las columnas de One-Hot Encoding de rutas.

**División temporal:** 80% para entrenamiento (2023–2024 aprox.) y 20% para prueba (2025 aprox.), respetando el orden cronológico estricto para simular de forma realista un escenario de producción donde el modelo solo puede predecir el futuro utilizando información del pasado.

**Modelos entrenados:**

*   **Regresión Lineal con Regularización (Ridge):** Se seleccionó este algoritmo para actuar como nuestro modelo de referencia (*baseline*). Su inclusión es indispensable para cuantificar con precisión el verdadero valor agregado por los algoritmos no lineales más complejos. Se configuró con penalización L2 ($$\alpha = 10$$) y estandarización mediante `StandardScaler` para mitigar el riesgo de multicolinealidad entre los rezagos temporales altamente correlacionados (como `lag_7` y `rolling_7_mean`), garantizando estimaciones de coeficientes estables.
*   **Random Forest:** Se eligió este ensamble de árboles basados en embolsado (*bagging*) por su robustez ante valores atípicos y su capacidad natural para capturar interacciones no lineales entre las variables del calendario y los rezagos. Al limitar la profundidad máxima a 10 y exigir un mínimo de 4 muestras por hoja, obligamos al modelo a aprender umbrales de decisión generalizables (por ejemplo, el impacto de un lunes festivo) en lugar de memorizar el ruido específico del set de entrenamiento.
*   **XGBoost (Gradient Boosting):** Se incorporó como el modelo candidato de alto rendimiento debido a su estado del arte en datos tabulares y su estrategia de optimización basada en el descenso de gradiente secuencial (donde cada nuevo árbol corrige de forma dirigida los errores cometidos por los anteriores). Para asegurar una convergencia suave y evitar el sobreajuste ante fluctuaciones estocásticas de la demanda, se configuró con una tasa de aprendizaje baja ($$\eta = 0.05$$), 150 estimadores y un submuestreo del 80% en filas y columnas que induce aleatoriedad y diversidad en las subdivisiones de los árboles.

Aunque las Redes Neuronales Recurrentes como las **LSTM (Long Short-Term Memory)** son arquitecturas potentes para modelar secuencias temporales crudas, su implementación en este escenario específico se descartó bajo el principio de parsimonia por las siguientes razones:

1.  **Naturaleza Tabular del Problema:** Al estructurar explícitamente la serie temporal mediante ingeniería de datos (creando variables de rezago, medias móviles, desviaciones estándar y variables de calendario exógenas), transformamos con éxito el problema en una tarea de aprendizaje supervisado sobre datos tabulares estructurados. En este dominio, los algoritmos basados en árboles de decisión (especialmente XGBoost) igualan o superan consistentemente el rendimiento de las redes profundas, pero con una fracción del esfuerzo de cómputo.
2.  **Costo de Mantenimiento y Despliegue:** Una red LSTM requiere mayor potencia de cómputo para entrenamiento, hiperparametrización y ejecución en producción. Además, son altamente sensibles al desvanecimiento o explosión del gradiente y propensas al sobreajuste en presencia de datos con ruido estocástico diario. Implementar una LSTM habría introducido una complejidad de infraestructura desproporcionada en la herramienta web final, sin garantizar una ganancia estadística significativa frente a la robustez y velocidad de inferencia de XGBoost.

## 5. Comparación de Modelos y Métricas

El rendimiento de las predicciones en el conjunto de prueba se evaluó con métricas estándar para series de tiempo: el **Error Absoluto Medio (MAE)**, la **Raíz del Error Cuadrático Medio (RMSE)** y el **coeficiente de determinación (R²)**. Todas las métricas se calcularon sobre los viajes reales tras des-normalizar los valores para garantizar la interpretabilidad de los resultados en el contexto logístico.

<div align="center" markdown="1">

$$MAE = \frac{1}{n}\sum_{t=1}^{n}|y_t - \hat{y}_t| \qquad RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)^2} \qquad R^2 = 1 - \frac{\sum(y_t - \hat{y}_t)^2}{\sum(y_t - \bar{y})^2}$$

*Ecuaciones 5, 6 y 7. Métricas de rendimiento estadístico de proyecciones*
</div>

Los resultados agregados ponderados sobre todas las rutas e identidades origen-destino se consolidan de manera comparativa a continuación:

<div align="center" markdown="1">

*Tabla 2. Comparativa de rendimiento promedio entre modelos (todas las rutas, set de prueba 2025)*

| Modelo | MAE (Viajes diarios) | RMSE (Viajes diarios) | R² |
| :--- | :---: | :---: | :---: |
| **Ridge Regression (Lineal)** | 56.2 | 71.7 | 0.564 |
| **Random Forest** | 43.9 | 56.6 | 0.703 |
| **XGBoost (Propuesto)** | **38.8** | **50.4** | **0.754** |

</div>

<br>

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo1/04_model_comparison.png?raw=true" width="100%" alt="Comparativa de Modelos Barras" />
    <p><em>Figura 3. Métricas promedio consolidadas de RMSE, MAE y R² para el conjunto de rutas de prueba</em></p>
</div>

Como se detalla cuantitativamente en la **Tabla 2** y en los gráficos comparativos de la **Figura 3**:
*   La regresión lineal con penalización L2 (Ridge) arrojó un desempeño deficiente frente a los ensambles de árboles de decisión ($$R^2 = 0.564$$). Esto evidencia que los modelos lineales sufren de subajuste ante patrones complejos de estacionalidad estocástica cruzada.
*   **XGBoost demostró ser el modelo campeón** para la serie, obteniendo el menor error absoluto medio (**$$38.8$$ viajes**) y un coeficiente de determinación sobresaliente de **$$0.754$$**. Esto significa que el modelo captura el **75.4% de la varianza** presente en la demanda agregada diaria de las 20 rutas de prueba del año 2025.

### 5.1 Desempeño Gráfico de Inferencia en Test

Para evaluar de forma detallada la capacidad de seguimiento de trayectoria, se contrastó la predicción de los tres modelos en el conjunto de test para una ruta de alto volumen.

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo1/03_pred_vs_real.png?raw=true" width="100%" alt="Predicción vs Demanda Real" />
    <p><em>Figura 4. Comparativa temporal detallada de la demanda real vs. proyecciones de los modelos para la ruta origen-destino 132-230</em></p>
</div>

El comportamiento dinámico observado en la **Figura 4** revela que el modelo de regresión lineal (Ridge) sobreestima sistemáticamente los valles de demanda de baja afluencia y presenta excesiva rigidez. Por el contrario, XGBoost (línea discontinua verde) se adapta con precisión a los picos transitorios y asimila de manera satisfactoria las caídas semanales del flujo de traslados, consolidándose como la opción óptima para el despliegue del sistema integrado.


## 6. Predicción Final: Próximos 30 Días

Con el modelo XGBoost seleccionado como campeón, se generó una proyección para **enero 2026** (30 días de horizonte) para las rutas prioritarias del sistema. El pronóstico se ejecutó bajo una estrategia **multi-paso recursiva**, donde el modelo utiliza su propia salida del día anterior ($$t-1$$) para recalcular de manera iterativa los componentes rezagados de los días subsiguientes.

<div style="text-align: center;">
    <img src="https://github.com/jihernandezc/rnaab_sistema_integrado/blob/main/output/modulo1/05_forecast.png?raw=true" width="100%" alt="Pronóstico de Demanda a 30 Días" />
    <p><em>Figura 5. Pronóstico de demanda diaria para los próximos 30 días (Enero 2026) en la ruta JFK Airport → Times Sq / Theatre District</em></p>
</div>

La proyección de la **Figura 5** demuestra la robustez del método recursivo. El modelo hereda la inercia del volumen histórico de cierre de diciembre de 2025 de la NYC TLC y proyecta de manera coherente el patrón de alta y baja estacionalidad semanal que experimentará la ruta durante el primer mes de 2026.


## 7. Resumen Operacional

A partir del pronóstico generado, se derivaron métricas operativas para apoyar la planificación de flota y personal. Asumiendo un promedio de **4 pasajeros por vehículo** y **25 viajes por conductor por día**, se estimaron los requerimientos de vehículos y conductores para el pico diario proyectado en cada ruta.

<div align="center" markdown="1">

*Tabla 3. Resumen operacional de demanda pronosticada (30 días, enero 2026)*

<div style="overflow-x: auto; margin: 20px 0;">
  <table style="border-collapse: collapse; width: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 14px; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
    <thead>
      <tr style="background-color: #1f425c; color: white; text-align: left;">
        <th style="padding: 12px 15px; border-bottom: 2px solid #ddd;">Ruta (Origen → Destino)</th>
        <th style="padding: 12px 15px; border-bottom: 2px solid #ddd; text-align: right;">Prom. Diario (Viajes)</th>
        <th style="padding: 12px 15px; border-bottom: 2px solid #ddd; text-align: right;">Pico Diario</th>
        <th style="padding: 12px 15px; border-bottom: 2px solid #ddd; text-align: right;">Mínimo Diario</th>
        <th style="padding: 12px 15px; border-bottom: 2px solid #ddd; text-align: right;">Total 30 Días</th>
        <th style="padding: 12px 15px; border-bottom: 2px solid #ddd; text-align: center;">Vehículos Peak</th>
        <th style="padding: 12px 15px; border-bottom: 2px solid #ddd; text-align: center;">Conductores</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">JFK Airport → Times Sq / Theatre District</td>
        <td style="padding: 10px 15px; text-align: right;">248.1</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">277.9</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">173.7</td>
        <td style="padding: 10px 15px; text-align: right;">7,444</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">12</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">12</td>
      </tr>
      <tr style="background-color: #f8f9fa; border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Lenox Hill East → Upper East Side (North)</td>
        <td style="padding: 10px 15px; text-align: right;">243.6</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">283.4</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">175.1</td>
        <td style="padding: 10px 15px; text-align: right;">7,309</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">12</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">12</td>
      </tr>
      <tr style="border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Lincoln Square → Upper West Side (North)</td>
        <td style="padding: 10px 15px; text-align: right;">292.4</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">337.7</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">234.5</td>
        <td style="padding: 10px 15px; text-align: right;">8,771</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">14</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">14</td>
      </tr>
      <tr style="background-color: #f8f9fa; border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Midtown Center → Upper East Side (South)</td>
        <td style="padding: 10px 15px; text-align: right;">313.1</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">413.7</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">179.8</td>
        <td style="padding: 10px 15px; text-align: right;">9,393</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">17</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">17</td>
      </tr>
      <tr style="border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Midtown Center → Upper East Side (North)</td>
        <td style="padding: 10px 15px; text-align: right;">370.3</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">484.2</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">201.0</td>
        <td style="padding: 10px 15px; text-align: right;">11,109</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">20</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">20</td>
      </tr>
      <tr style="background-color: #f8f9fa; border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Midtown East → Upper East Side (North)</td>
        <td style="padding: 10px 15px; text-align: right;">172.0</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">235.1</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">87.2</td>
        <td style="padding: 10px 15px; text-align: right;">5,160</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">10</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">10</td>
      </tr>
      <tr style="border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Penn Station / Madison Sq. → Times Sq / Theatre District</td>
        <td style="padding: 10px 15px; text-align: right;">244.5</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">272.9</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">188.9</td>
        <td style="padding: 10px 15px; text-align: right;">7,336</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">11</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">11</td>
      </tr>
      <tr style="background-color: #f8f9fa; border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper East Side (North) → Lenox Hill East</td>
        <td style="padding: 10px 15px; text-align: right;">235.4</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">295.2</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">146.0</td>
        <td style="padding: 10px 15px; text-align: right;">7,061</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">12</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">12</td>
      </tr>
      <tr style="border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper East Side (North) → Midtown Center</td>
        <td style="padding: 10px 15px; text-align: right;">234.7</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">335.2</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">84.8</td>
        <td style="padding: 10px 15px; text-align: right;">7,041</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">14</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">14</td>
      </tr>
      <tr style="background-color: #f8f9fa; border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper East Side (South) → Upper East Side (North)</td>
        <td style="padding: 10px 15px; text-align: right;">500.5</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">629.8</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">260.3</td>
        <td style="padding: 10px 15px; text-align: right;">15,014</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">26</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">26</td>
      </tr>
      <tr style="border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper East Side (North) → Upper East Side (South)</td>
        <td style="padding: 10px 15px; text-align: right;">648.6</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">841.7</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">324.9</td>
        <td style="padding: 10px 15px; text-align: right;">19,458</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">34</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">34</td>
      </tr>
      <tr style="background-color: #f8f9fa; border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper East Side (North) → Upper West Side (North)</td>
        <td style="padding: 10px 15px; text-align: right;">215.1</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">256.7</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">163.1</td>
        <td style="padding: 10px 15px; text-align: right;">6,454</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">11</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">11</td>
      </tr>
      <tr style="border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper East Side (South) → Lenox Hill East</td>
        <td style="padding: 10px 15px; text-align: right;">212.3</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">250.9</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">145.9</td>
        <td style="padding: 10px 15px; text-align: right;">6,368</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">11</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">11</td>
      </tr>
      <tr style="background-color: #f8f9fa; border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper East Side (South) → Midtown Center</td>
        <td style="padding: 10px 15px; text-align: right;">311.6</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">439.2</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">140.6</td>
        <td style="padding: 10px 15px; text-align: right;">9,347</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">18</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">18</td>
      </tr>
      <tr style="border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper East Side (South) → Midtown East</td>
        <td style="padding: 10px 15px; text-align: right;">271.0</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">364.7</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">136.6</td>
        <td style="padding: 10px 15px; text-align: right;">8,130</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">15</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">15</td>
      </tr>
      <tr style="background-color: #f8f9fa; border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper East Side (South) → Upper East Side (North)</td>
        <td style="padding: 10px 15px; text-align: right;">749.9</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">964.2</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">395.0</td>
        <td style="padding: 10px 15px; text-align: right;">22,496</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">39</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">39</td>
      </tr>
      <tr style="border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper East Side (North) → Upper East Side (North)</td>
        <td style="padding: 10px 15px; text-align: right;">492.3</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">648.6</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">273.4</td>
        <td style="padding: 10px 15px; text-align: right;">14,769</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">26</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">26</td>
      </tr>
      <tr style="background-color: #f8f9fa; border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper West Side (South) → Upper West Side (North)</td>
        <td style="padding: 10px 15px; text-align: right;">233.9</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">264.6</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">199.3</td>
        <td style="padding: 10px 15px; text-align: right;">7,017</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">11</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">11</td>
      </tr>
      <tr style="border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper West Side (North) → Lincoln Square</td>
        <td style="padding: 10px 15px; text-align: right;">288.6</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">333.7</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">238.3</td>
        <td style="padding: 10px 15px; text-align: right;">8,657</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">14</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">14</td>
      </tr>
      <tr style="background-color: #f8f9fa; border-bottom: 1px solid #eee;">
        <td style="padding: 10px 15px; font-weight: bold; color: #333;">Upper West Side (North) → Upper West Side (South)</td>
        <td style="padding: 10px 15px; text-align: right;">284.8</td>
        <td style="padding: 10px 15px; text-align: right; color: #d9534f; font-weight: bold;">321.0</td>
        <td style="padding: 10px 15px; text-align: right; color: #5cb85c;">232.7</td>
        <td style="padding: 10px 15px; text-align: right;">8,543</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">13</td>
        <td style="padding: 10px 15px; text-align: center; font-weight: bold; background-color: #fcf8e3;">13</td>
      </tr>
    </tbody>
  </table>
</div>

</div>

La proyección de enero de 2026 presente en la tabla 3 permite extraer tres hallazgos clave para la planificación logística y la optimización de recursos de la flota:

#### 1. Concentración de la Demanda en Rutas Internas Residenciales
Las rutas internas del sector **Upper East Side** dominan de manera absoluta el volumen de viajes proyectado. El corredor bidireccional entre **Upper East Side (South) y Upper East Side (North)** (Filas 9, 10, 15 y 16) acumula el mayor tráfico de la red, destacando la ruta de *South a North* (Fila 15) con una proyección de **22,496 viajes mensuales** y un promedio diario de **749.9 viajes**.
*   **Implicación Logística:** Estas rutas representan traslados cortos de alta densidad y frecuencia continua. La asignación de **39 vehículos en horas pico** y su respectiva tripulación debe priorizarse en esta zona, ya que su alta rotación minimiza los tiempos de viaje sin pasajeros y maximiza la rentabilidad por hora del conductor.

#### 2. Corredores Comerciales con Volumen Estable
Las rutas que conectan nodos comerciales o corporativos con distritos residenciales, tales como **Midtown Center → Upper East Side (North)** (Fila 4) y **Lincoln Square → Upper West Side (North)** (Fila 2), muestran volúmenes diarios estables (promedios de **370.3** y **292.4** viajes, respectivamente).
*   **Implicación Logística:** A diferencia de las rutas internas residenciales, estos corredores presentan una marcada sensibilidad horaria (picos en horas de entrada y salida laboral). El modelo proyecta la necesidad de escalar la flota hasta **20 vehículos en el pico máximo diario** para Midtown Center. Sincronizar el inicio de los turnos de los conductores con estas ventanas de congestión permitirá absorber la demanda sin incrementar los tiempos de espera del usuario en la aplicación web.

#### 3. Estabilidad Operativa del Segmento Aeroportuario
La ruta **JFK Airport → Times Sq / Theatre District** (Fila 0), si bien no registra el volumen de los corredores internos de Manhattan (promedio diario de **248.1 viajes**), destaca por su **estabilidad y resiliencia ante los valles de demanda**. Su valor mínimo diario proyectado (**173.7 viajes**) es proporcionalmente más alto en relación con su promedio que el de las rutas residenciales, donde los fines de semana la demanda se reduce drásticamente (por ejemplo, la Fila 8 cae a un mínimo de **84.8 viajes**).
*   **Implicación Logística:** El flujo aeroportuario es menos sensible a la estacionalidad del día de la semana. Requiere una base constante de **12 vehículos y conductores** estacionados de manera permanente. Esta regularidad permite estructurar turnos fijos de larga distancia, garantizando ingresos estables para un subgrupo de la flota y reduciendo la incertidumbre operativa en días de baja demanda generalizada.

## 8. Aspectos Éticos y Creatividad

### Gestión y Privacidad de Datos
El dataset analizado contiene únicamente información agregada de conteo de viajes por ruta y fecha, proveniente de registros públicos de la NYC TLC. Por diseño, la solución se adhiere a la privacidad desde el diseño (*Privacy by Design*), sin recolectar datos de identificación personal (PII) de los usuarios —nombres, identificaciones, números de tarjetas o ubicaciones individuales— garantizando que el modelo analice patrones de flujo netamente impersonales.

### Análisis de Sesgos Geográficos
Es importante reconocer que el criterio de selección de las 20 rutas de mayor volumen concentra el análisis en las zonas de más alta demanda histórica (típicamente el centro de Manhattan). Esto puede invisibilizar patrones de movilidad en zonas periféricas o de menor ingreso. El uso de estas proyecciones para planificación debe complementarse con criterios de equidad de cobertura y no solo de optimización de utilización de flota.

### Enfoque Innovador
El aporte metodológico de este módulo consiste en la construcción de un **modelo global unificado** (un único XGBoost para todas las rutas simultáneamente) mediante One-Hot Encoding de identidades de ruta, en lugar de entrenar modelos individuales por cada par origen–destino. Esto reduce drásticamente la carga de mantenimiento en producción y permite al modelo aprovechar patrones compartidos entre rutas similares.


## 9. Conclusiones

La transición de una asignación empírica a una basada en este modelo predictivo de XGBoost mitiga dos riesgos financieros:
*   **Mitigación del Sub-abastecimiento:** Garantiza que en rutas de alta volatilidad (como *Midtown Center*, donde el pico diario de **484.2** supera por un **30.7%** al promedio diario de **370.3**) se prevea el despacho oportuno de unidades para no perder cuota de mercado.
*   **Reducción de Capacidad Ociosa:** Evita mantener conductores en zonas residenciales durante periodos de mínimo diario (como los domingos), redireccionando proactivamente los esfuerzos de cobertura hacia los flujos aeroportuarios o micro-corredores estables.

Sin embargo, se debe monitorear continuamente el desempeño del modelo en producción, especialmente ante eventos disruptivos (clima extremo, cambios en la infraestructura vial, eventos masivos) que puedan alterar los patrones históricos de movilidad. La capacidad de recalibración rápida del modelo con datos recientes será clave para mantener su relevancia operativa a largo plazo.

## 10. Bibliografía

Chakraborty, S. (2023). Demand Prediction for Public Transport in Nairobi (GitHub Repository). Recuperado de https://github.com/samchak18/Demand-Prediction-for-Public-Transport-in-Nairobi

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.

GeeksforGeeks (2025). Machine Learning: Transport Demand Prediction using Regression. Recuperado de https://www.geeksforgeeks.org/machine-learning/transport-demand-prediction-using-regression/

NYC Taxi & Limousine Commission (2025). TLC Trip Record Data. Recuperado de https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page