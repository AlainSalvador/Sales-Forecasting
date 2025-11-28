# Sales Forecasting Engine (Time Series Analysis)

## Descripción
Motor de Inteligencia Artificial capaz de predecir ventas futuras basándose en datos históricos. A diferencia de las proyecciones lineales simples, este modelo utiliza **Random Forest** para detectar patrones complejos (días de la semana, estacionalidad mensual) y genera **Intervalos de Confianza del 95%** para la gestión de riesgos en inventarios.

## Características Clave
* **Algoritmo:** Random Forest Regressor (Ensemble Learning).
* **Simulación de Escenarios:** Generación de "Nubes de Probabilidad" basadas en la varianza de 200 árboles de decisión.
* **Feature Engineering:** Transformación de fechas en variables cíclicas comprensibles para la máquina.
* **Realistic Noise:** Inyección de estocasticidad para modelar la incertidumbre del mercado real.

## 🛠 Stack Tecnológico
* Python, Pandas, Numpy.
* Scikit-Learn (Machine Learning).
* Matplotlib (Visualización Financiera).
