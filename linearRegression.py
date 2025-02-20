#Definindo os dados de entrada de X e Y 
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] #Meses do ano sendo 1 Janeiro e 12 Dezembro;

y = [40, 35, 30, 20, 20, 40, 35, 30, 25, 20, 50, 45] #Quantidade de combustível gastos em litros por mês;

#Calculando a média de X e Y para serem usados no cálculo do coeficiente angular B1;
xAverage = sum(x) / len(x)
yAverage = sum(y) / len(x)

# Calculando o numerador e denominador de Beta1

#Σ(xi - x̄)(yi - ȳ) ----> Fórmula
#xi representa cada valor no conjunto de dados X, usamos for i in range (expressão geradora) 
#x̄ é a média de X
#yi representa cada valor no conjunto de dados Y, usamos for i in range (expressão geradora)
#ȳ é a média de Y
#O numerador é a somatória do produto de (xi - x̄) e (yi - ȳ)
betaOneNumerator = sum((x[i] - xAverage) * (y[i] - yAverage) for i in range(len(x)))

#---------------------------------------------------------------------------------------------------------#

#Σ(xi - x̄)² ----> Fórmula
#Produtos notáveis, quadrado da diferença usando (expressão geradora)
betaOneDenominator = sum((x[i] - xAverage) ** 2 for i in range(len(x))) 

#Pegamos o valor do numerador e denominador e fazemos a divisão para encontrar o coeficiente angular B1
betaOne = betaOneNumerator / betaOneDenominator

#Pegamos a média de Y e subtraímos pelo produto da média de X e o coeficiente angular B1 para encontrar o coeficiente linear B0 (interceptor)
#Só é possível calcular o B0 quando já temos o B1
#B0 = ȳ - B1 * x̄ ----> Fórmula
#ȳ é a média de Y
#x̄ é a média de X
#B1 é o coeficiente angular
betaZero = yAverage - (betaOne * xAverage)

#---------------------------------------------------------------------------------------------------------#

print(f"Equação da reta: y = {betaZero:.2f} + {betaOne:.2f}x")
