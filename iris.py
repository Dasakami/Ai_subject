from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame

print("🔎 Первые цветочки:")
print(df.head(), "\n")

# Какие виды есть
print("🌼 Виды ирисов:")
print(df['target'].map(dict(enumerate(iris.target_names))).value_counts(), "\n")

print("📏 Средние размеры по видам (сепал/петал/хуитал):")
print(df.groupby('target').mean(), "\n")

for i, name in enumerate(iris.target_names):
    sub = df[df['target'] == i]
    print(f"🌸 Вид: {name}")
    print(f"   👉 Средняя длина чашелистика: {sub['sepal length (cm)'].mean():.2f}")
    print(f"   👉 Средняя ширина чашелистика: {sub['sepal width (cm)'].mean():.2f}")
    print(f"   👉 Средняя длина лепестка: {sub['petal length (cm)'].mean():.2f}")
    print(f"   👉 Средняя ширина лепестка: {sub['petal width (cm)'].mean():.2f}")
    print()
