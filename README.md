# Detección de Anomalías en Rodamientos con Machine Learning

Este proyecto se enfoca en la detección de fallos en rodamientos utilizando el dataset del "Case Western Reserve University (CWRU) Bearing Data Center". Se implementa una solución basada en modelos de machine learning clásicos para clasificar las señales de vibración y predecir posibles anomalías.

## Problema

La detección temprana de fallos en componentes mecánicos como los rodamientos es fundamental en el mantenimiento predictivo industrial. Un fallo inesperado puede ocasionar paradas de producción costosas y daños mayores en la maquinaria. El análisis de las señales de vibración es una técnica comúnmente utilizada para diagnosticar la condición de estos componentes. El reto consiste en procesar estas señales y clasificarlas correctamente para identificar patrones que indiquen un fallo incipiente.

## Solución Propuesta

Para abordar este problema, se propone el uso de modelos de machine learning clásicos. La solución consiste en entrenar algoritmos de clasificación para que aprendan a distinguir entre las señales de vibración de un rodamiento en estado normal y aquellas que presentan diferentes tipos de fallos. Se exploran diversos modelos como Regresión Logística, Naive Bayes, Árboles de Decisión, Random Forest, XGBoost, etc., para encontrar el que ofrezca el mejor rendimiento para este caso de uso.

## Componentes y Flujo de Trabajo

![](docs/arquitectura-deteccion-anomalias.drawio.png)

El proyecto está organizado en varios componentes que trabajan en conjunto para ofrecer una solución completa, desde la experimentación hasta la puesta en producción de un modelo.

### Componentes

-   **`notebooks/`**: Contiene una serie de Jupyter Notebooks donde se realiza el análisis exploratorio de datos (EDA), el preprocesamiento de las señales, la extracción de características y el entrenamiento y evaluación de los diferentes modelos de machine learning.
-   **`fast-api/`**: Una aplicación de API REST desarrollada con FastAPI. Su función es servir el modelo de machine learning entrenado, permitiendo realizar predicciones en tiempo real a través de endpoints HTTP.
-   **`mlflow/`**: Se utiliza para la gestión del ciclo de vida de los modelos. MLflow permite registrar experimentos, comparar el rendimiento de diferentes modelos, versionar los modelos y almacenar los artefactos generados durante el entrenamiento.
-   **`minio_client/`**: Un cliente para interactuar con MinIO, que es un servidor de almacenamiento de objetos de alto rendimiento compatible con la API de Amazon S3. En este proyecto, MinIO se utiliza como backend para almacenar los artefactos de los modelos registrados en MLflow.
-   **`docker-compose.yaml`**: Un archivo que orquesta el despliegue de todos los servicios del proyecto (FastAPI, MLflow, MinIO) como contenedores de Docker, facili-tando su configuración y ejecución en cualquier entorno.

### Flujo de Trabajo

1.  **Experimentación y Entrenamiento**: Se utilizan los notebooks para cargar los datos del CWRU, preprocesarlos, extraer características y entrenar los modelos. Cada experimento, con sus parámetros y métricas, se registra en MLflow.
2.  **Selección del Modelo**: A través de la interfaz de MLflow, se comparan los resultados de los experimentos y se selecciona el modelo con el mejor rendimiento.
3.  **Servicio del Modelo**: El modelo seleccionado se carga en la aplicación FastAPI, que expone un endpoint para realizar predicciones.
4.  **Despliegue**: El sistema completo se despliega utilizando `docker-compose`, que levanta todos los servicios necesarios de forma coordinada.

## Entrenamiento de los Modelos

El proceso de entrenamiento de los modelos se detalla en los notebooks y sigue los siguientes pasos:

1.  **Carga de Datos**: Se cargan las señales de vibración del dataset CWRU.
2.  **Extracción de Características**: Se aplican técnicas de procesamiento de señales, como la Transformada Rápida de Fourier (FFT), para extraer características relevantes del dominio de la frecuencia.
3.  **División de Datos**: El dataset se divide en conjuntos de entrenamiento y prueba para poder evaluar el modelo de forma objetiva.
4.  **Entrenamiento del Modelo**: Se instancia y entrena el modelo de machine learning elegido con los datos de entrenamiento.
5.  **Evaluación**: Se evalúa el rendimiento del modelo utilizando el conjunto de prueba y se calculan métricas como `accuracy`, `precision`, `recall` y `F1-score`. Se analiza también la matriz de confusión.
6.  **Registro en MLflow**: Todos los aspectos relevantes del entrenamiento (parámetros, métricas, y el propio modelo) se registran en un experimento de MLflow para su trazabilidad y comparación.

## Cómo Correr el Proyecto

Para ejecutar este proyecto en tu entorno local, necesitas tener instalado **Docker** y **Docker Compose**.

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu_usuario/tu_repositorio.git
    cd tu_repositorio
    ```

    1.1.    **Copia el archivo .env**
    ```bash
    cp sample.env .env
    ```

    1.2.    **Modifica el archivo .env**
    ```bash
    MINIO_ACCESS_KEY=your_minio_access_key
    MINIO_SECRET_KEY=your_minio_secret_key
    MINIO_BUCKET_NAME=your_minio_bucket_name

    POSTGRES_USER=your_postgres_username
    POSTGRES_PASSWORD=your_postgres_password
    POSTGRES_DB=your_postgres_database
    POSTGRES_HOST=your_postgres_host
    POSTGRES_PORT=your_postgres_port 

    MLFLOW_EXPERIMENT_NAME=your_experiment_name

    JUPYTER_TOKEN=your_jupyter_token
    ```


2.  **Levanta los servicios con Docker Compose:**
    ```bash
    docker-compose up -d
    ```
    Este comando construirá las imágenes de Docker necesarias y levantará los siguientes servicios en segundo plano:
    -   **API (FastAPI)**: Accesible en `http://localhost:8000/docs`
    -   **MLflow Tracking Server**: Accesible en `http://localhost:5000`
    -   **MinIO Object Storage**: Accesible en `http://localhost:9001` (consola web)

3.  **Entrena los modelos:**
    Para entrenar los modelos, puedes ejecutar los Jupyter Notebooks que se encuentran en el directorio `notebooks/`. Puedes hacerlo utilizando tu entorno local de Jupyter o configurando un servicio de Jupyter en Docker.

4.  **Interactúa con los servicios:**
    -   Puedes explorar los experimentos y modelos registrados en la interfaz de **MLflow**.
    -   Puedes enviar peticiones a la **API** para obtener predicciones del modelo que se haya cargado.
    -   Puedes gestionar los artefactos almacenados en **MinIO** a través de su consola web.

5.  **Para detener los servicios:**
    ```bash
    docker-compose down
    ```

___________________


[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/jelambrar1)

Made with Love ❤️ by [@jelambrar96](https://github.com/jelambrar96)

