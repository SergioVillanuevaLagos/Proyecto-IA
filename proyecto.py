import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.image as mpimg
import os

# Cargar los datos desde el archivo CSV
file_path = 'housing.csv'  # Asegúrate de que la ruta sea correcta
data = pd.read_csv(file_path)

# Rellenar valores faltantes en 'total_bedrooms' con la mediana
data['total_bedrooms'].fillna(data['total_bedrooms'].median(), inplace=True)

# Visualización exploratoria de correlaciones
plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlación entre características")
plt.show()

# Histograma de la variable objetivo (median_house_value)
plt.figure(figsize=(8, 5))
sns.histplot(data['median_house_value'], kde=True, bins=30, color="blue")
plt.title("Distribución de median_house_value")
plt.xlabel("median_house_value")
plt.ylabel("Frecuencia")
plt.show()

# Verificar si la imagen existe antes de intentar cargarla
if os.path.exists('california.png'):  # Asegúrate de que la ruta sea correcta
    california_img = mpimg.imread('california.png')  # Cambia la ruta si es necesario
else:
    print("El archivo de imagen no se encuentra en la ruta especificada.")
    california_img = None

# Crear el gráfico de dispersión
plt.figure(figsize=(10, 7))
ax = plt.gca()  # Obtener el eje actual
housing_plot = data.copy()  # Hacer una copia de los datos para visualización
housing_plot.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                  s=housing_plot['population'] / 100, label='population', 
                  c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True, ax=ax)

# Si la imagen existe, añadirla al gráfico
if california_img is not None:
    ax.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)

# Etiquetas y leyenda
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.legend()
plt.title("Distribución de viviendas en California")
plt.show()

# Seleccionar características relevantes y la variable objetivo
target_column = "median_house_value"
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income']

X = data[features]
y = data[target_column]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de Regresión Lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación de métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nMétricas del modelo:")
print(f"MAE (Error absoluto medio): {mae:.2f}")
print(f"RMSE (Raíz del error cuadrático medio): {rmse:.2f}")
print(f"R² (Coeficiente de determinación): {r2:.2f}")

# Visualizar resultados
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.7, label="Predicciones")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Ideal")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Regresión Lineal: Resultados")
plt.legend()
plt.show()

# Curva de Aprendizaje
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring="neg_mean_squared_error")
train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)
plt.figure(figsize=(10, 5))
plt.plot(train_sizes, train_scores_mean, 'o-', label="Entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Validación")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Error cuadrático medio (MSE)")
plt.title("Curva de aprendizaje")
plt.legend()
plt.show()
