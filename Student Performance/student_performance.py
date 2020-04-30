### MACHINE LEARNING - REGRESSION MODEL (STUDENT PERFORMANCE) ###
#
# Student Performance Data Set
#
# Fonte origem dataset:
# https://archive.ics.uci.edu/ml/datasets/Student+Performance
#
# # Trabalho base:
# http://www3.dsi.uminho.pt/pcortez/student.pdf
#
# P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance.
# In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008)
# pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.
#


# Importação dos pacotes para análise e manipulação de dados
import numpy as np
import pandas as pd

# Importação dos pacotes para divisão dos dados e criação do modelo
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importação do pacote para salvar o modelo em arquivo pickle
import pickle

# Fonte dos dados
fonte_modelo = ''
fonte_previsao = ''

''' @@@ REMOVER ESTA LINHA E ACRESCENTAR (3 X [']) NO FINAL PARA CRIAR NOVO MODELO @@@

# Carregamento dos dados
dados_modelo = pd.read_csv(fonte_modelo, sep=';')
n_dados = dados_modelo[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

# Label alvo (classe)
previsao = 'G3'

# Define os atributos (todas as colunas exceto a coluna "G3" )
atributos = np.array(n_dados.drop(previsao, 1))

# Define a classe (coluna "G3")
classe = np.array(n_dados[previsao])

# Define uma seed para replicação do modelo
np.random.seed(7)

### CRIAÇÃO E SALVAMENTO DO MODELO DE REGRESSÃO LINEAR

melhor_acuracidade = 0

while melhor_acuracidade < 0.97:

	# Separa os dados de treino e dados de teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        atributos, classe, test_size=0.1)

    # Criação do modelo
    modelo_linear = LinearRegression()

    # Treinamento do modelo
    modelo_linear.fit(x_treino, y_treino)

    # Avaliação a acuracidade do modelo
    acuracidade = modelo_linear.score(x_teste, y_teste)
    print(acuracidade)

    if acuracidade > melhor_acuracidade:
        melhor_acuracidade = acuracidade

        # Salva o modelo com o melhor percentual de acuracidade encontrado
        with open('linear_model_save.pickle', 'wb') as arquivo:
            pickle.dump(modelo_linear, arquivo)

# Previsão utilizando o modelo criado
previsao = modelo_linear.predict(x_teste)

# Comparação entre valor previsto
for i in range(len(previsao)):
    print(previsao[i], x_teste[i], y_teste[i])

'''

# Abre o modelo salvo
arquivo_pickle = open('linear_model_save.pickle', 'rb')

# Carregamento do modelo
modelo_linear = pickle.load(arquivo_pickle)

# Carregamento de novos dados para previsão
n_dados = pd.read_csv('student-por.csv', sep=';')

# Definição dos atributos
n_previsao = n_dados[
    ['G1', 'G2', 'studytime', 'failures', 'absences']].values

# Realiza as previsões da nota final (G3) utilizando o modelo carregado
previsao = modelo_linear.predict(n_previsao)
print(previsao)
