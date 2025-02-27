from sklearn import load_wine
import pandas as pd

#Carregando dataset de vinhos.
data = load_wine()

#Usando o dataset de vinhos e passando para o padrão do panda para melhorar a manipulação.
data = pd.read_csv()
print(data.head())