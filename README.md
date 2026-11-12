# PLC Anomaly Lab

Aplicación demostrable de detección de anomalías para un proceso PLC inspirado en TLIGHT. Preparada por **Antonio Robles Meza** para explicar un flujo de investigación reproducible: entrenar detectores con comportamiento normal, inyectar escenarios anómalos controlados y comparar las métricas de detección.

> Alcance honesto: esta es una **réplica metodológica con simulación propia**. No contiene los registros HMI originales ni debe presentar sus resultados como los del artículo. Su propósito es demostrar el método de forma ejecutable y reproducible.

## Qué incluye

- Simulador de trazas TLIGHT: 14 direcciones/señales de proceso, diagnóstico y temporización.
- Siete escenarios de anomalía inspirados en el estudio: conflictos de luces, apagado general, solicitudes denegadas y manipulación temporal.
- Entrenamiento solo con 5,000 estados normales y normalización Min–Max.
- Tres detectores que se entrenan realmente al ejecutar la aplicación:
  - `OneClassSVM` (OCSVM, núcleo RBF)
  - OCNN de una capa oculta (implementación NumPy de objetivo one-class con pérdida hinge)
  - `IsolationForest` (IF)
- Panel web en español con F1, accuracy, precision, recall, matriz de confusión, gráfica comparativa, inspección de memoria PLC y exportación de CSV.
- Archivos de despliegue para Render: `render.yaml`, `Procfile` y `Dockerfile`.

## Ejecutar en esta laptop

La carpeta ya contiene un entorno virtual local preparado. Para abrir el panel, ejecuta:

```powershell
cd C:\Users\ANTONIO\Documents\Codex\2026-11-12\rep\outputs\plc-anomaly-lab
.\.venv\Scripts\python.exe app.py
```

Luego visita [http://127.0.0.1:5000](http://127.0.0.1:5000). También puedes hacer doble clic en `start_local.bat`; si el entorno no existe, lo crea e instala las dependencias.

Para verificar el backend sin abrir el navegador:

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

La corrida de los cinco conjuntos se completa en pocos segundos en la verificación local. La semilla predeterminada (`2026`) permite repetirla; cambiarla genera otra muestra sintética reproducible.

## Diseño experimental

| Componente | Implementación en esta aplicación |
|---|---|
| Proceso | Ciclo de cruce vehicular/peatonal TLIGHT simulado |
| Entrenamiento | 5,000 estados válidos, sin etiquetas de ataque |
| Señales | Luces vehiculares y peatonales, botón, solicitud, interbloqueo, sincronización, retardo, fase, temporizador y diagnósticos de integridad |
| Pruebas | 5 conjuntos de 5,000; 7,000; 13,130; 15,000; y 18,270 muestras |
| Criterio | Una puntuación positiva indica anomalía; se informan TP, TN, FP, FN, accuracy, precision, recall y F1 |

Los diagnósticos de integridad son señales sintéticas añadidas para hacer visible la relación entre coherencia de proceso y alertas. En una instalación real deben sustituirse por direcciones PLC/HMI verificadas y validarse con personal OT.

## Desplegar en GitHub y Render

El repositorio y Render necesitan una sesión conectada con tus propias cuentas antes de publicar. Cuando conectes ambas, estos son los pasos concretos:

1. En GitHub, crea un repositorio vacío llamado `plc-anomaly-lab` (sin README inicial).
2. Desde esta carpeta, publica el código:

   ```powershell
   git init
   git add .
   git commit -m "Implement PLC anomaly detection lab"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/plc-anomaly-lab.git
   git push -u origin main
   ```

3. En Render, elige **New + → Blueprint**, conecta el repositorio y selecciona `render.yaml`.
4. Render instalará `requirements.txt`, iniciará `gunicorn` y comprobará `/health`. Cada `git push` a `main` vuelve a desplegar la aplicación.

No se necesitan secretos ni claves de API para esta versión. Si el usuario GitHub asociado a `aroblesm@undac.edu.pe` no es `TU_USUARIO`, reemplázalo por el nombre real de la cuenta; no se infiere a partir del correo.

## Estructura

```text
app.py              API Flask, simulador y modelos
templates/          panel web
static/             estilos e interacción del dashboard
smoke_test.py       prueba rápida de API
requirements.txt    dependencias reproducibles
render.yaml         configuración de Render
Dockerfile           alternativa de contenedor
```

## Referencia académica

Boateng, E. A., & Bruce, J. W. (2022). *Unsupervised Machine Learning Techniques for Anomaly Detection in PLC Process Control*. Journal of Cybersecurity and Privacy, 2(2), 220–244. [https://doi.org/10.3390/jcp2020012](https://doi.org/10.3390/jcp2020012)

El artículo compara OCSVM, OCNN e Isolation Forest sobre un montaje OpenPLC/ScadaBR TLIGHT; la metodología de entrenamiento normal, los escenarios y los tamaños de prueba orientan esta demostración. Consulta el artículo para los datos y las métricas originales.
