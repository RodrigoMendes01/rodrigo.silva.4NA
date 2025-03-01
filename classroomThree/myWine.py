import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score, roc_curve, auc
)

#Carregando e preparando os dados
wines = load_wine()
dataFrame = pd.DataFrame(wines.data, columns=wines.feature_names)
dataFrame["region"] = wines.target

X = dataFrame.drop(columns="region")
y = dataFrame["region"]

#Dividindo os dados em TREINO e TESTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Padronização dos dados devido a sensibilidade do KNN a escala. Evitando assim o domínio de uma variável sobre a outra.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Criando e treinando o modelo KNN
#Neighbors 5 significa que ele vai olhar para os 5 vizinhos mais próximos para fazer a classificação
modelKnn = KNeighborsClassifier(n_neighbors=5)
modelKnn.fit(X_train, y_train)

#Fazendo predições
y_pred = modelKnn.predict(X_test)

#Exibindo Acurácia e Relatório de Classificação
print("="*50)
print("Resultados da Classificação")
print("="*50)
print(f"🔹 Acurácia do Modelo KNN (k=5): {accuracy_score(y_test, y_pred):.2f}")
print("\n🔹 Relatório de Classificação:\n")
print(classification_report(y_test, y_pred, target_names=wines.target_names))

#Exibindo a matriz de Confusão
print("="*50)
print("Matriz de Confusão:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Exibindo a matriz de Confusão como Gráfico
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False, xticklabels=wines.target_names, yticklabels=wines.target_names)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("🔹 Matriz de Confusão")
plt.show()

#Calculando a Precisão, Recall e F1-Score
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("="*50)
print("Avaliação do Modelo")
print("="*50)
print(f"🎯 Precisão: {precision:.2f}")
print(f"📢 Recall: {recall:.2f}")
print(f"📊 F1-Score: {f1:.2f}")

#Curva ROC e AUC Multiclasse
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_pred_prob = modelKnn.predict_proba(X_test)

plt.figure(figsize=(8, 6))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Classe {wines.target_names[i]} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], "k--") 
plt.xlabel("Falso Positivo (FPR)")
plt.ylabel("Verdadeiro Positivo (TPR)")
plt.title("🔹 Curva ROC para Classificação Multiclasse")
plt.legend()
plt.grid()
plt.show()



