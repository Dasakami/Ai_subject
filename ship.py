import pandas as пандос
#Хотел написать пиндос или пидорос, но подумал это черезчур, АХАХААХ
import numpy as нумпи 
import seaborn as sps
import matplotlib.pyplot as плт

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

кораблик = пандос.read_csv("train.csv")
print(кораблик.head())

print("Кто выжил, а кто сдох:")
print(кораблик["Survived"].value_counts(normalize=True))

print("\nВыживаемость по полу (спойлер: девочки живут):")
print(кораблик.groupby("Sex")["Survived"].mean())

print("\nВыживаемость по классу каюты (богатые везунчики):")
print(кораблик.groupby("Pclass")["Survived"].mean())

sps.histplot(кораблик["Age"].dropna(), bins=20, kde=True)
плт.title("Сколько кому лет было")
плт.show()

# чиним пустоты(в этом коде уять слово есть (┬┬﹏┬┬) )
кораблик["Age"] = кораблик["Age"].fillna(кораблик["Age"].median())
кораблик["Embarked"] = кораблик["Embarked"].fillna(кораблик["Embarked"].mode()[0])

кораблик = пандос.get_dummies(кораблик, columns=["Sex", "Embarked"], drop_first=True)

# выкидываем мусорные колонки (Хотел и себя выкинуть)
кораблик = кораблик.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

X = кораблик.drop("Survived", axis=1)
сдохли_или_нет = кораблик["Survived"]

# делим на учебу и контроль  
X_train, X_test, y_train, y_test = train_test_split(X, сдохли_или_нет, test_size=0.2, random_state=42)

# масштабируем чтобы не ругался логистический (Привет братан)
шкала = StandardScaler()
X_train_scaled = шкала.fit_transform(X_train)
X_test_scaled = шкала.transform(X_test)

# логистическая регрессия aka скучная модель
логист = LogisticRegression(max_iter=1000)
логист.fit(X_train_scaled, y_train)
угадай_логист = логист.predict(X_test_scaled)

# рандомный лес 
лес = RandomForestClassifier(n_estimators=100, random_state=42)
лес.fit(X_train, y_train)
угадай_лес = лес.predict(X_test)

print("\n=== Логистическая регрессия (топим за математику, она лучшаяяяя) ===")
print("Accuracy:", accuracy_score(y_test, угадай_логист))
print("F1-score:", f1_score(y_test, угадай_логист))
print(classification_report(y_test, угадай_логист))
# беляя акуарансий писать так сложно
print("\n=== Случайный лес (много деревьев лучше чем одно? прям как я) ===")
print("Accuracy:", accuracy_score(y_test, угадай_лес))
print("F1-score:", f1_score(y_test, угадай_лес))
print(classification_report(y_test, угадай_лес))
