<p align="center">
  <img src="https://cdiac.manizales.unal.edu.co/imagenes/LogosMini/un.png" width="250">
  <br>
  <strong>Universidad Nacional de Colombia</strong><br>
  Facultad de Minas - Sede Medellín
</p>

# **Sistema Inteligente Integrado para Predicción, Clasificación y Recomendación en la Empresa de Transporte**

## 👥 Integrantes
* Tomás Acevedo Roldán
* Santiago Cardona Franco
* Jimena Hernández Castillo

**Profesor:** Juan David Ospina  
**Monitor:** Andrés Mauricio Zapata

---

Este repositorio contiene el desarrollo del **Trabajo 3** de la asignatura *Introducción a Redes Neuronales y Algoritmos Bioinspirados* (Semestre 2026-01). El informe técnico completo, detallando el preprocesamiento de los datos, el diseño de los modelos, las métricas de evaluación y los análisis de resultados para los tres módulos, se encuentra desplegado en el siguiente enlace:

<div align="center">

[![Ver Reporte](https://img.shields.io/badge/🚀_Ver_Reporte-GitHub_Pages-blue?style=for-the-badge)](https://jihernandezc.github.io/rnaab_sistema_integrado/)

</div>

## 🛠️ Estructura del Repositorio

De acuerdo con el entorno de desarrollo, el proyecto está estructurado de la siguiente manera:

```text
📦 rnaab_sistema_integrado
 ┣ 📂 data                         # Archivos fuente y conjuntos de datos (CSVs y Parquets)
 ┣ 📂 docs                         # Archivos fuente para el despliegue en GitHub Pages
 ┃ ┣ 📂 assets                     # Recursos estáticos (estilos CSS, imágenes de portada)
 ┃ ┣ 📜 _config.yml                # Archivo de configuración de Jekyll / GitHub Pages
 ┃ ┣ 📜 index.md                   # Reporte principal (Portada, Resumen, Intro, Metodología)
 ┃ ┣ 📜 prediccion.md              # Documentación Módulo 1: Predicción de Demanda
 ┃ ┣ 📜 clasificacion.md           # Documentación Módulo 2: Detección de Conducción Distractiva
 ┃ ┣ 📜 recomendacion.md           # Documentación Módulo 3: Recomendación de Destinos
 ┃ ┗ 📜 sitio-web.md               # Documentación de la Herramienta Web y Videos
 ┣ 📂 output                       # Artefactos y recursos generados por los notebooks
 ┃ ┣ 📂 modulo1                    # Gráficos y modelos exportados de Predicción de Demanda
 ┃ ┣ 📂 modulo2                    # Gráficos y modelos exportados de Conducción Distractiva
 ┃ ┗ 📂 modulo3                    # Gráficos y modelos exportados del Recomendador de Destinos
 ┣ 📜 Modulo1_Prediccion.ipynb     # Notebook de desarrollo del Módulo 1 (Series de Tiempo)
 ┣ 📜 Modulo2_Clasificacion.ipynb   # Notebook de desarrollo del Módulo 2 (CNN / Transfer Learning)
 ┣ 📜 Modulo3_Recomendacion.ipynb  # Notebook de desarrollo del Módulo 3 (Neural CF Híbrido)
 ┗ 📜 README.md                    # Descripción general del repositorio
```

## Solución Final: Sitio Web Interactivo

Para interactuar directamente con los tres modelos integrados (proyectar la demanda de rutas de taxi, subir imágenes para detectar la atención de un conductor o probar el motor de recomendaciones personalizado), accede a nuestra herramienta web:

<div align="center">
  
  <a href="https://jihernandezc-rnaab-sistema-integrado.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Streamlit-Visitar_Sitio_Web-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Visitar Sitio Web">
  </a>
</div>
```