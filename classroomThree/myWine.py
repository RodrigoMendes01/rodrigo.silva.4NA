from sklearn.datasets import load_wine;
from sklearn.model_selection import train_test_split;
import pandas as pd;
import numpy as np;

#Carregando o dataset do scikit-learn
wines = load_wine()

#Criando um dataframe com os dados fornecidos pelo scikit-learn, passando para um padrão do pandas para melhor visualização
dataFrame = pd.DataFrame(wines.data, columns=wines.feature_names)

#Adicionando a coluna região que será o target
dataFrame["region"] = wines.target

#Usando random seed 42 para garantir que a aleatoriedade da geração dos números seja a mesma
np.random.seed(42)

#Simulando um preço para os vinhos baseando-se na região, teor alcoólico, fenóis e etc. O preço é uma combinação linear desses fatores e não está sendo exibido em moeda local nenhuma.
#Caso haja necessidade de um valor monetário real, seria necessária uma base de dados com preços reais de vinhos para uma melhor acurácia e algo mais próximo do real
dataFrame["price"] = (
    (dataFrame["region"] * 30) +
    (dataFrame["alcohol"] * 12) +
    (dataFrame["flavanoids"] * 20) +
    (dataFrame["total_phenols"] * 15) +
    (dataFrame["color_intensity"] * 10) +
    np.random.randint(0, 100, size=len(dataFrame))
)

#Separando features X, dados que contém a compisição química dos vinhos
dataFrameX = dataFrame.drop(columns="price")

#Separando o target Y
dataFrameY = dataFrame["price"]

#Dividindo o conjunto de dados em TREINO e TESTE
trainX, testX, trainY, testY = train_test_split(dataFrameX, dataFrameY, test_size=0.3, random_state=42)
