import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

#2 años de ventas diarias (simulación)
fechas = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
n_dias = len(fechas)

#tendencia
tendencia = np.linspace(start=50, stop=150, num=n_dias)

# estacionalidad (simular ondas senoidales con picos mensuales)
estacionalidad = 20 * np.sin(np.linspace(0, 3.14 * 8, n_dias))

#dias malos y dias buenos aleatorios
ruido = np.random.normal(0, 10, n_dias)

ventas = tendencia + estacionalidad + ruido

df = pd.DataFrame({'Fecha': fechas, 'Ventas': ventas})
#ventas a enteros y evitar numeros negativos
df['Ventas'] = df['Ventas'].clip(lower=0).astype(int)

print("Primeras 5 filas historico de ventas")
print(df.head())

plt.figure(figsize=(12, 6))
plt.plot(df['Fecha'], df['Ventas'], label='Ventas Reales', color='#1f77b4')
plt.title('Historico de Ventas (Tendencia + Estacionalidad)')
plt.xlabel('Tiempo')
plt.ylabel('Unidades vendidas')
plt.grid(True, alpha=0.3)
plt.show()

#Transformar fechas en numeros
#Crear copia
df_ia = df.copy()

#Fecha a núnmero secuencial
df_ia['Dia_Numerico'] = np.arange(len(df))
#caracteristicas cíclicas
df_ia['Mes'] = df_ia['Fecha'].dt.month
#dia de la semana?
df_ia['Dia_semana'] = df_ia['Fecha'].dt.dayofweek

print("Datos para la IA")
print(df_ia[['Fecha', 'Dia_Numerico', 'Mes', 'Dia_semana']].tail())

from sklearn.ensemble import RandomForestRegressor

# Entrenar modelo
X_total = df_ia[['Dia_Numerico', 'Mes', 'Dia_semana']]
y_total = df_ia['Ventas']

#Usar 200 arboles de decision
modelo_pro = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
modelo_pro.fit(X_total, y_total)

#Generar futuro
futuro_fechas = pd.date_range(start='2025-01-01', end='2025-01-31', freq='D')
df_futuro = pd.DataFrame({'Fecha': futuro_fechas})

ultimo_dia = df_ia['Dia_Numerico'].max()
df_futuro['Dia_Numerico'] = range(ultimo_dia + 1, ultimo_dia + 1 + len(futuro_fechas))
df_futuro['Mes'] = df_futuro['Fecha'].dt.month
df_futuro['Dia_semana'] = df_futuro['Fecha'].dt.dayofweek

X_futuro = df_futuro[['Dia_Numerico', 'Mes', 'Dia_semana']]

#Matriz de deciciosnes aleatorias de los 200 arboles
predicciones_arboles = np.array([tree.predict(X_futuro) for tree in modelo_pro.estimators_])

#Calcular la media (La línea principal)
prediccion_media = np.mean(predicciones_arboles, axis=0)

#Calcular la Desviación Estándar (La incertidumbre/volatilidad)
desviacion = np.std(predicciones_arboles, axis=0)

#Definir los límites: Escenario Pesimista vs Optimista (Intervalo del 95%)
limite_superior = prediccion_media + (2 * desviacion)
limite_inferior = prediccion_media - (2 * desviacion)

#ruido
ruido_realidad = np.random.normal(0, 3, len(prediccion_media)) 
prediccion_media_organica = prediccion_media + ruido_realidad


#visualizar
plt.figure(figsize=(14, 7))

# 1. Histórico Real (Último trimestre)
df_zoom = df_ia[df_ia['Fecha'] >= '2024-10-01'] 
plt.plot(df_zoom['Fecha'], df_zoom['Ventas'], label='Histórico Real', color='#1f77b4', linewidth=2)

# 2. La "Nube" de Incertidumbre (Intervalo de Confianza)
plt.fill_between(df_futuro['Fecha'], limite_inferior, limite_superior, color='red', alpha=0.2, label='Zona de Incertidumbre (95%)')

# 3. La Línea de Predicción
plt.plot(df_futuro['Fecha'], prediccion_media_organica, label='Predicción Enero 2025', color='#cc0000', linewidth=2)

plt.title('Proyección Financiera Enero 2025 (Con Análisis de Riesgo)', fontsize=16)
plt.ylabel('Ventas Proyectadas')
plt.xlabel('Fecha')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.show()